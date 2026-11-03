#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_csi.data.delay_power_residual import (
    ConditionNeighborhoodBatchSampler,
    DelayPowerResidualCacheDataset,
)
from physics_csi.evaluation.scene_diagnostics import paired_map_summary
from physics_csi.models.scene_aod_residual import AoDResidualOutput
from physics_csi.models.scene_delay_power_residual import SceneConditionedDelayPowerResidual
from physics_csi.physics.synthesizer import MIMOOFDMSynthesizer, SystemConfig
from physics_csi.training.aod_residual import AoDDistributionLoss, AoDLossWeights
from physics_csi.training.trainer import move_to_device, resolve_device
from physics_csi.utils.io import atomic_json_dump
from physics_csi.utils.seed import seed_everything


def _subset(payload: Mapping[str, Any], indices: torch.Tensor) -> dict[str, torch.Tensor]:
    sample_count = int(
        next(value for value in payload.values() if isinstance(value, torch.Tensor)).shape[0]
    )
    return {
        key: value[indices]
        for key, value in payload.items()
        if isinstance(value, torch.Tensor) and value.shape[:1] == (sample_count,)
    }


def _without_csi(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Keep path tensors on the metric device without copying large CSI arrays."""
    return {key: value for key, value in payload.items() if key != "csi"}


def _cat(chunks: Mapping[str, list[torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.cat(values) for key, values in chunks.items()}


def _append(chunks: dict[str, list[torch.Tensor]], payload: Mapping[str, torch.Tensor]) -> None:
    for key, value in payload.items():
        if isinstance(value, torch.Tensor):
            chunks.setdefault(key, []).append(value.detach().cpu())


def _wrap_degrees(value: torch.Tensor) -> torch.Tensor:
    radians = torch.deg2rad(value.float())
    return torch.rad2deg(torch.atan2(radians.sin(), radians.cos()))


def _quantiles(value: torch.Tensor, *, absolute: bool = False) -> dict[str, float]:
    value = value.detach().float().flatten()
    if absolute:
        value = value.abs()
    return {
        "mean": float(value.mean()),
        "std": float(value.std(unbiased=False)),
        "p01": float(torch.quantile(value, 0.01)),
        "p05": float(torch.quantile(value, 0.05)),
        "p50": float(torch.quantile(value, 0.50)),
        "p95": float(torch.quantile(value, 0.95)),
        "p99": float(torch.quantile(value, 0.99)),
        "minimum": float(value.min()),
        "maximum": float(value.max()),
    }


def _correlation(first: torch.Tensor, second: torch.Tensor) -> float | None:
    first = first.float().flatten()
    second = second.float().flatten()
    first = first - first.mean()
    second = second - second.mean()
    denominator = first.square().mean().sqrt() * second.square().mean().sqrt()
    if float(denominator) <= 1e-12:
        return None
    return float((first * second).mean() / denominator)


def _center_by_group(values: torch.Tensor, group: torch.Tensor) -> torch.Tensor:
    result = values.clone().float()
    for item in torch.unique(group):
        selected = group == item
        result[selected] = result[selected] - result[selected].mean()
    return result


def _group_mean_r2(values: torch.Tensor, group: torch.Tensor) -> float | None:
    values = values.float()
    prediction = torch.empty_like(values)
    for item in torch.unique(group):
        selected = group == item
        prediction[selected] = values[selected].mean()
    total = (values - values.mean()).square().sum()
    if float(total) <= 1e-12:
        return None
    return float(1.0 - (values - prediction).square().sum() / total)


def _structure_for_angle(
    residual: torch.Tensor,
    mask: torch.Tensor,
    map_ids: torch.Tensor,
    relative_delay: torch.Tensor,
    log_power_fraction: torch.Tensor,
    rank: torch.Tensor,
    original_azimuth: torch.Tensor,
    original_elevation: torch.Tensor,
    direct_path: torch.Tensor,
    effective_limit: float,
) -> dict[str, Any]:
    active = residual[mask].float()
    active_map = map_ids.unsqueeze(-1).expand_as(mask)[mask]
    link_id = torch.arange(mask.shape[0]).unsqueeze(-1).expand_as(mask)[mask]
    active_delay = relative_delay[mask]
    active_power = log_power_fraction[mask]
    active_rank = rank.expand_as(mask)[mask]
    active_azimuth = torch.deg2rad(original_azimuth.float())[mask]
    active_elevation = original_elevation.float()[mask]
    active_direct = direct_path.float()[mask]
    centered = _center_by_group(active, active_map)
    within_link_std = []
    within_map_std: dict[str, float] = {}
    per_map: dict[str, Any] = {}
    for link in range(mask.shape[0]):
        values = residual[link][mask[link]].float()
        within_link_std.append(values.std(unbiased=False))
    for map_id in torch.unique(map_ids, sorted=True):
        selected = active_map == map_id
        values = active[selected]
        within_map_std[str(int(map_id))] = float(values.std(unbiased=False))
        per_map[f"map_{int(map_id)}"] = {
            "paths": int(selected.sum()),
            "signed_mean_deg": float(values.mean()),
            "signed_std_deg": float(values.std(unbiased=False)),
            "positive_fraction": float((values > 1e-6).float().mean()),
            "negative_fraction": float((values < -1e-6).float().mean()),
        }
    absolute_fraction = active.abs() / max(effective_limit, 1e-12)
    features = {
        "relative_delay": active_delay,
        "log_power_fraction": active_power,
        "path_rank": active_rank,
        "original_azimuth_sin": active_azimuth.sin(),
        "original_azimuth_cos": active_azimuth.cos(),
        "original_elevation": active_elevation,
        "direct_path_indicator": active_direct,
    }
    return {
        "effective_limit_deg": effective_limit,
        "signed": _quantiles(active),
        "absolute": _quantiles(active, absolute=True),
        "positive_fraction": float((active > 1e-6).float().mean()),
        "negative_fraction": float((active < -1e-6).float().mean()),
        "near_zero_fraction": float((active.abs() <= 1e-3).float().mean()),
        "fraction_at_least_90_percent_of_limit": float((absolute_fraction >= 0.90).float().mean()),
        "fraction_at_least_95_percent_of_limit": float((absolute_fraction >= 0.95).float().mean()),
        "fraction_at_least_99_percent_of_limit": float((absolute_fraction >= 0.99).float().mean()),
        "map_constant_offset_r2": _group_mean_r2(active, active_map),
        "link_constant_offset_r2": _group_mean_r2(active, link_id),
        "within_link_path_std_deg": _quantiles(torch.stack(within_link_std), absolute=True),
        "within_map_std_deg": _quantiles(
            torch.tensor(list(within_map_std.values())), absolute=True
        ),
        "global_correlations": {
            name: _correlation(active, values) for name, values in features.items()
        },
        "map_centered_correlations": {
            name: _correlation(centered, _center_by_group(values, active_map))
            for name, values in features.items()
        },
        "per_map": per_map,
    }


class MapConstantAoDOffset(nn.Module):
    """Two learned scalars per Map; no path, DP or sample-dependent input."""

    def __init__(
        self,
        map_id_to_index: Mapping[str | int, int],
        *,
        maximum_azimuth_deg: float,
        maximum_elevation_deg: float,
    ) -> None:
        super().__init__()
        self.map_id_to_index = {
            int(map_id): int(index) for map_id, index in map_id_to_index.items()
        }
        self.maximum_azimuth_deg = float(maximum_azimuth_deg)
        self.maximum_elevation_deg = float(maximum_elevation_deg)
        self.raw_azimuth = nn.Parameter(torch.zeros(len(self.map_id_to_index)))
        self.raw_elevation = nn.Parameter(torch.zeros(len(self.map_id_to_index)))

    def map_indices(self, map_id: torch.Tensor) -> torch.Tensor:
        values = [self.map_id_to_index[int(item)] for item in map_id.detach().cpu().tolist()]
        return torch.tensor(values, device=map_id.device, dtype=torch.long)

    def external_map_ids(self, map_index: torch.Tensor) -> torch.Tensor:
        inverse = {index: map_id for map_id, index in self.map_id_to_index.items()}
        return torch.tensor(
            [inverse[int(item)] for item in map_index.detach().cpu().tolist()],
            device=map_index.device,
            dtype=torch.long,
        )

    def forward(
        self, payload: Mapping[str, torch.Tensor], map_index: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], AoDResidualOutput]:
        mask = payload["path_mask"].bool()
        azimuth = self.maximum_azimuth_deg * torch.tanh(self.raw_azimuth[map_index])
        elevation = self.maximum_elevation_deg * torch.tanh(self.raw_elevation[map_index])
        azimuth_residual = azimuth.unsqueeze(-1).expand_as(mask) * mask
        elevation_residual = elevation.unsqueeze(-1).expand_as(mask) * mask
        corrected = dict(payload)
        corrected["aod_az_deg"] = _wrap_degrees(
            payload["aod_az_deg"].float() + azimuth_residual
        ) * mask
        corrected["aod_el_deg"] = (
            payload["aod_el_deg"].float() + elevation_residual
        ).clamp(-90.0, 90.0) * mask
        output = AoDResidualOutput(
            aod_az_deg=corrected["aod_az_deg"],
            aod_el_deg=corrected["aod_el_deg"],
            azimuth_residual_deg=azimuth_residual,
            elevation_residual_deg=elevation_residual,
            azimuth_gate=torch.ones((), device=mask.device),
            elevation_gate=torch.ones((), device=mask.device),
        )
        return corrected, output


def _different_map_indices(
    correct: torch.Tensor, map_count: int, generator: torch.Generator | None = None
) -> torch.Tensor:
    if generator is None:
        offset = 1 + torch.remainder(correct * 37, map_count - 1)
    else:
        offset = torch.randint(1, map_count, (), generator=generator, device=correct.device)
    return (correct + offset) % map_count


@torch.inference_mode()
def _project_dp_split(
    dataset: DelayPowerResidualCacheDataset,
    dp_model: SceneConditionedDelayPowerResidual,
    device: torch.device,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    chunks: dict[str, list[torch.Tensor]] = {}
    for start in range(0, len(dataset), batch_size):
        indices = torch.arange(start, min(start + batch_size, len(dataset)))
        reference = move_to_device(_subset(dataset.reference, indices), device)
        base = move_to_device(_subset(dataset.base, indices), device)
        map_index = dp_model.map_indices(reference["map_id"]).to(device)
        output = dp_model(
            base,
            reference["conditions"],
            reference["condition_validity"],
            map_index=map_index,
        )
        payload = dp_model.apply_to_payload(base, output)
        _append(chunks, payload)
    return _cat(chunks)


def _make_loss(
    checkpoint: Mapping[str, Any], *, seed: int
) -> AoDDistributionLoss:
    values = copy.deepcopy(checkpoint["loss_init"])
    values["seed"] = seed
    values["projections"] = max(128, int(values.get("projections", 64)))
    values["weighted_quantiles"] = max(512, int(values.get("weighted_quantiles", 128)))
    weights = AoDLossWeights(**values.pop("weights"))
    return AoDDistributionLoss(**values, weights=weights)


def _synthesize(payload: Mapping[str, torch.Tensor], model: MIMOOFDMSynthesizer) -> torch.Tensor:
    return model(
        payload["path_delays_ns"],
        payload["path_powers_linear"],
        payload["path_phases_rad"],
        payload["aoa_az_deg"],
        payload["aoa_el_deg"],
        payload["aod_az_deg"],
        payload["aod_el_deg"],
        payload["path_mask"],
    )


@torch.inference_mode()
def _synthesize_batched(
    payload: Mapping[str, torch.Tensor],
    model: MIMOOFDMSynthesizer,
    batch_size: int,
) -> torch.Tensor:
    sample_count = int(payload["path_mask"].shape[0])
    chunks: list[torch.Tensor] = []
    for start in range(0, sample_count, batch_size):
        indices = torch.arange(
            start,
            min(start + batch_size, sample_count),
            device=payload["path_mask"].device,
        )
        chunks.append(_synthesize(_subset(payload, indices), model).cpu())
    return torch.cat(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit AoD residual structure and fit a Map-only constant-offset baseline."
    )
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dp-checkpoint", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-constant-artifact", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=9026)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--train-batches-per-epoch", type=int, default=400)
    parser.add_argument("--validation-batches-per-epoch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--minimum-batch-size", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--projection-batch-size", type=int, default=512)
    parser.add_argument("--minimum-samples-per-map", type=int, default=None)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = resolve_device(args.device)
    artifact = torch.load(args.artifact, map_location="cpu", weights_only=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    dp_checkpoint = torch.load(args.dp_checkpoint, map_location="cpu", weights_only=False)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    if artifact.get("model_kind") != "b3_plus_dp_plus_aod_residual":
        raise ValueError("artifact is not a B3+DP+AoD evaluation")
    if checkpoint.get("model_kind") != "scene_aod_residual":
        raise ValueError("checkpoint is not an AoD residual")
    if dp_checkpoint.get("model_kind") != "scene_delay_power_residual":
        raise ValueError("DP checkpoint has the wrong model kind")
    for contract in (checkpoint["cache_contract"], dp_checkpoint["cache_contract"]):
        if contract["scene_plan_fingerprint"] != cache["scene_plan_fingerprint"]:
            raise ValueError("checkpoint and cache scene plans differ")
        if contract["b3_checkpoint_sha256"] != cache["b3_checkpoint_sha256"]:
            raise ValueError("checkpoint and cache frozen B3 backbones differ")
    if artifact.get("checkpoint") and Path(artifact["checkpoint"]).resolve() != Path(
        args.checkpoint
    ).resolve():
        raise ValueError("--checkpoint does not match the checkpoint recorded in --artifact")
    if artifact.get("dp_checkpoint") and Path(artifact["dp_checkpoint"]).resolve() != Path(
        args.dp_checkpoint
    ).resolve():
        raise ValueError("--dp-checkpoint does not match the checkpoint recorded in --artifact")

    dp = artifact["dp_only"]
    full = artifact["correct_map"]
    reference = artifact["reference"]
    mask = dp["path_mask"].bool()
    azimuth_residual = _wrap_degrees(full["aod_az_deg"] - dp["aod_az_deg"])
    elevation_residual = full["aod_el_deg"].float() - dp["aod_el_deg"].float()
    delay = dp["path_delays_ns"].float()
    first = torch.where(mask, delay, torch.full_like(delay, torch.inf)).amin(-1)
    relative = (delay - first.unsqueeze(-1)).clamp_min(0.0) * mask
    power = dp["path_powers_linear"].float() * mask
    fraction = power / power.sum(-1, keepdim=True).clamp_min(1e-20)
    log_fraction = torch.log(fraction.clamp_min(1e-20))
    rank = torch.arange(mask.shape[1], dtype=torch.float32).unsqueeze(0)
    direct_path = dp.get("direct_path_mask", torch.zeros_like(mask)).bool() & mask
    state = checkpoint.get("ema_state_dict", checkpoint["model_state_dict"])
    azimuth_gate = float(torch.sigmoid(state["azimuth_gate_logit"]))
    elevation_gate = float(torch.sigmoid(state["elevation_gate_logit"]))
    model_config = checkpoint["model_init"]
    minimum_samples_per_map = args.minimum_samples_per_map or int(
        checkpoint.get("config", {}).get("evaluation", {}).get(
            "minimum_samples_per_map", 12
        )
    )
    structure = {
        "azimuth": _structure_for_angle(
            azimuth_residual,
            mask,
            reference["map_id"].long(),
            relative,
            log_fraction,
            rank,
            dp["aod_az_deg"],
            dp["aod_el_deg"],
            direct_path,
            azimuth_gate * float(model_config["maximum_azimuth_residual_deg"]),
        ),
        "elevation": _structure_for_angle(
            elevation_residual,
            mask,
            reference["map_id"].long(),
            relative,
            log_fraction,
            rank,
            dp["aod_az_deg"],
            dp["aod_el_deg"],
            direct_path,
            elevation_gate * float(model_config["maximum_elevation_residual_deg"]),
        ),
    }

    train_dataset = DelayPowerResidualCacheDataset(cache, "train")
    validation_dataset = DelayPowerResidualCacheDataset(cache, "validation")
    dp_model = SceneConditionedDelayPowerResidual(**dp_checkpoint["model_init"]).to(device)
    dp_model.load_state_dict(dp_checkpoint.get("ema_state_dict", dp_checkpoint["model_state_dict"]))
    dp_model.requires_grad_(False).eval()
    print(
        json.dumps({"event": "project_dp_train_start", "samples": len(train_dataset)}),
        flush=True,
    )
    dp_train = _project_dp_split(train_dataset, dp_model, device, args.projection_batch_size)
    print(
        json.dumps(
            {"event": "project_dp_validation_start", "samples": len(validation_dataset)}
        ),
        flush=True,
    )
    dp_validation = _project_dp_split(
        validation_dataset, dp_model, device, args.projection_batch_size
    )

    sampler_kwargs = {
        "batch_size": args.batch_size,
        "minimum_batch_size": args.minimum_batch_size,
        "condition_std": cache["b3_statistics"]["condition_std"],
    }
    train_sampler = ConditionNeighborhoodBatchSampler(
        train_dataset,
        batches_per_epoch=args.train_batches_per_epoch,
        seed=args.seed,
        shuffle=True,
        **sampler_kwargs,
    )
    validation_sampler = ConditionNeighborhoodBatchSampler(
        validation_dataset,
        batches_per_epoch=args.validation_batches_per_epoch,
        seed=args.seed + 100000,
        shuffle=False,
        **sampler_kwargs,
    )
    model = MapConstantAoDOffset(
        cache["map_id_to_index"],
        # Match the effective range of the trained residual.  Giving this
        # ablation the ungated architectural maximum would make the capacity
        # comparison needlessly asymmetric.
        maximum_azimuth_deg=structure["azimuth"]["effective_limit_deg"],
        maximum_elevation_deg=structure["elevation"]["effective_limit_deg"],
    ).to(device)
    training_loss = _make_loss(checkpoint, seed=args.seed).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_validation = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    patience = 0
    history: list[dict[str, Any]] = []

    def run_epoch(epoch: int, *, training: bool) -> dict[str, float]:
        sampler = train_sampler if training else validation_sampler
        sampler.set_epoch(epoch)
        dp_payload = dp_train if training else dp_validation
        dataset = train_dataset if training else validation_dataset
        model.train(training)
        totals: dict[str, float] = defaultdict(float)
        samples = 0
        generator = torch.Generator(device=device).manual_seed(
            args.seed + epoch * 1009 + (0 if training else 500000)
        )
        context = torch.enable_grad if training else torch.no_grad
        with context():
            for indices in sampler:
                index = torch.tensor(indices, dtype=torch.long)
                current_dp = move_to_device(_subset(dp_payload, index), device)
                current_reference = move_to_device(_subset(dataset.reference, index), device)
                correct_index = model.map_indices(current_reference["map_id"])
                wrong_index = _different_map_indices(
                    correct_index, len(model.map_id_to_index), generator
                )
                if training:
                    optimizer.zero_grad(set_to_none=True)
                correct, correct_output = model(current_dp, correct_index)
                shuffled, _ = model(current_dp, wrong_index)
                metrics = training_loss(
                    correct_payload=correct,
                    shuffled_payload=shuffled,
                    dp_payload=current_dp,
                    reference_payload=current_reference,
                    correct_output=correct_output,
                )
                if training:
                    metrics["loss"].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                count = len(indices)
                for key, value in metrics.items():
                    totals[key] += float(value.detach()) * count
                samples += count
        return {key: value / samples for key, value in totals.items()}

    started = time.monotonic()
    for epoch in range(args.epochs):
        train_metrics = run_epoch(epoch, training=True)
        validation_metrics = run_epoch(epoch, training=False)
        record = {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        history.append(record)
        print(json.dumps({"event": "constant_offset_epoch", **record}, sort_keys=True), flush=True)
        if validation_metrics["loss"] < best_validation:
            best_validation = validation_metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
        if patience >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("constant-offset baseline produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()

    diagnostic_loss = _make_loss(checkpoint, seed=args.seed + 700000).to(device).eval()
    dp_test = move_to_device(_without_csi(dp), device)
    reference_test = move_to_device(_without_csi(reference), device)
    full_test = move_to_device(_without_csi(full), device)
    correct_index = model.map_indices(reference_test["map_id"])
    wrong_index = _different_map_indices(correct_index, len(model.map_id_to_index))
    with torch.inference_mode():
        constant_test, _ = model(dp_test, correct_index)
        wrong_constant_test, _ = model(dp_test, wrong_index)
    per_map: dict[str, Any] = {}
    metric_names = (
        "path_power_aod_swd",
        "delay_power_aod_swd",
        "link_power_aod_swd",
        "power_weighted_aod_swd",
        "distribution_score",
    )
    with torch.inference_mode():
        for map_id in sorted(int(item) for item in torch.unique(reference_test["map_id"])):
            index = torch.nonzero(reference_test["map_id"] == map_id, as_tuple=False).flatten()
            if index.numel() < minimum_samples_per_map:
                per_map[f"map_{map_id}"] = {
                    "map_id": map_id,
                    "samples": int(index.numel()),
                    "status": "insufficient_samples",
                }
                continue
            current_reference = _subset(reference_test, index)
            models = {}
            for label, payload in (
                ("dp_only", dp_test),
                ("constant_offset", constant_test),
                ("full_aod_residual", full_test),
                ("shuffled_constant_offset", wrong_constant_test),
            ):
                terms = diagnostic_loss.distribution_terms(
                    _subset(payload, index), current_reference
                )
                models[label] = {key: float(value) for key, value in terms.items()}
            per_map[f"map_{map_id}"] = {
                "map_id": map_id,
                "samples": int(index.numel()),
                "status": "ok",
                "models": models,
            }

    maps = [item for item in per_map.values() if item.get("status") == "ok"]
    if not maps:
        raise RuntimeError("no Map has enough samples for the constant-offset audit")
    comparisons = {
        "full_aod_residual_vs_constant_offset": ("full_aod_residual", "constant_offset"),
        "constant_offset_vs_dp_only": ("constant_offset", "dp_only"),
        "full_aod_residual_vs_dp_only": ("full_aod_residual", "dp_only"),
        "constant_offset_vs_shuffled_constant_offset": (
            "constant_offset",
            "shuffled_constant_offset",
        ),
    }
    summaries: dict[str, Any] = {}
    for comparison_number, (name, (candidate, baseline)) in enumerate(comparisons.items()):
        summaries[name] = {}
        for metric_number, metric in enumerate(metric_names):
            summaries[name][metric] = paired_map_summary(
                [item["models"][candidate][metric] for item in maps],
                [item["models"][baseline][metric] for item in maps],
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed + comparison_number * 1000 + metric_number,
            )

    constant_cpu = {key: value.detach().cpu() for key, value in constant_test.items()}
    wrong_constant_cpu = {
        key: value.detach().cpu() for key, value in wrong_constant_test.items()
    }
    synthesizer = MIMOOFDMSynthesizer(
        SystemConfig.from_mapping(checkpoint["system"])
    ).to(device)
    constant_cpu["csi"] = _synthesize_batched(
        constant_test, synthesizer, args.projection_batch_size
    )
    wrong_constant_cpu["csi"] = _synthesize_batched(
        wrong_constant_test, synthesizer, args.projection_batch_size
    )
    constant_cpu["map_id"] = reference["map_id"]
    wrong_constant_cpu["map_id"] = reference["map_id"]
    constant_cpu["conditioned_map_id"] = reference["map_id"]
    wrong_constant_cpu["conditioned_map_id"] = model.external_map_ids(
        wrong_index
    ).detach().cpu()
    constant_artifact = {
        "format_version": 1,
        "model_kind": "b3_dp_map_constant_aod_offset_diagnostic",
        "reference": reference,
        "physics_only_baseline": artifact["dp_only"],
        "dp_only": artifact["dp_only"],
        "correct_map": constant_cpu,
        "shuffled_map": wrong_constant_cpu,
        "original_b3": artifact["physics_only_baseline"],
        "full_aod_residual": artifact["correct_map"],
        "shuffled_conditioned_map_id": wrong_constant_cpu["conditioned_map_id"],
        "scene_plan_fingerprint": artifact["scene_plan_fingerprint"],
        "source_aod_artifact": str(Path(args.artifact).resolve()),
        "diagnostic_baseline": "physics_only_baseline is DP-only in this artifact",
    }
    constant_path = Path(args.output_constant_artifact)
    constant_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = constant_path.with_suffix(constant_path.suffix + ".tmp")
    torch.save(constant_artifact, temporary)
    temporary.replace(constant_path)

    learned_offsets = {
        f"map_{map_id}": {
            "azimuth_deg": float(
                model.maximum_azimuth_deg
                * torch.tanh(model.raw_azimuth[index]).detach().cpu()
            ),
            "elevation_deg": float(
                model.maximum_elevation_deg
                * torch.tanh(model.raw_elevation[index]).detach().cpu()
            ),
        }
        for map_id, index in sorted(model.map_id_to_index.items())
    }
    report = {
        "format_version": 1,
        "diagnostic": "aod_residual_structure_and_constant_offset_ablation",
        "samples": int(reference["map_id"].numel()),
        "maps_evaluated": len(maps),
        "minimum_samples_per_map": minimum_samples_per_map,
        "residual_structure": structure,
        "constant_offset_training": {
            "epochs_completed": len(history),
            "best_validation_loss": best_validation,
            "seconds": time.monotonic() - started,
            "history": history,
            "learned_offsets": learned_offsets,
        },
        "independent_metric_configuration": {
            "seed": args.seed + 700000,
            "projections": diagnostic_loss.path_directions.shape[0],
            "weighted_quantiles": diagnostic_loss.quantiles.numel(),
        },
        "summaries": summaries,
        "per_map": per_map,
        "constant_offset_artifact": str(constant_path.resolve()),
        "decision_rule": (
            "The full AoD residual must beat the Map-only constant offset, especially on "
            "power-weighted and link-level Power--AoD. High Map-constant R2 or saturation with "
            "no full-model advantage means the claimed per-path conditional mechanism is unsupported."
        ),
    }
    atomic_json_dump(report, args.output)
    print(
        json.dumps(
            {
                "event": "aod_residual_structure_diagnostic_complete",
                "output": str(Path(args.output).resolve()),
                "constant_artifact": str(constant_path.resolve()),
                "maps": len(maps),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
