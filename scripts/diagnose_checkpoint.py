#!/usr/bin/env python3
"""Localize reconstruction bottlenecks in a trained physics-conditioned CVAE.

The script is read-only: it never updates model parameters or checkpoint files.
It evaluates a bounded number of prepared validation/test batches and reports:

* recent training-history trends and plateau detection;
* sampled-posterior versus posterior-mean CSI reconstruction;
* path-parameter errors and latent-variable usage;
* oracle ablations that replace one predicted physical parameter family with
  its ground-truth value before re-synthesizing CSI;
* CSI-only gradient finiteness/norms for one batch.

The oracle ablations are the main localization tool.  For example, a large
drop from ``posterior_mean`` NMSE to ``oracle_phase`` NMSE means that phase
prediction is the dominant source of complex-CSI error.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_csi.data.dataset import PreparedShardDataset
from physics_csi.models.conditional_vae import PhysicsConditionedVAE
from physics_csi.physics.synthesizer import SystemConfig
from physics_csi.training.trainer import resolve_device
from physics_csi.utils.io import atomic_json_dump
from physics_csi.utils.seed import seed_everything


PREDICTED_PATH_KEYS = (
    "path_delays_ns",
    "path_powers_linear",
    "phases_rad",
    "aoa_az_deg",
    "aoa_el_deg",
    "aod_az_deg",
    "aod_el_deg",
)


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    return value


def complex_nmse_per_sample(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    numerator = (prediction - target).abs().square().flatten(1).sum(-1)
    denominator = target.abs().square().flatten(1).sum(-1).clamp_min(eps)
    return numerator / denominator


def circular_difference_radians(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    difference = prediction - target
    return torch.atan2(torch.sin(difference), torch.cos(difference))


def circular_difference_degrees(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    return torch.rad2deg(
        circular_difference_radians(
            torch.deg2rad(prediction), torch.deg2rad(target)
        )
    )


def posterior_reconstruction(
    model: PhysicsConditionedVAE,
    batch: Mapping[str, torch.Tensor],
    *,
    use_mean: bool,
) -> dict[str, torch.Tensor]:
    """Run the model with either q(z|x)'s mean or one posterior sample."""
    mean, log_variance, condition_embedding = model.encode(batch)
    latent = mean if use_mean else model.reparameterize(mean, log_variance)
    decoded = model.decoder(latent, condition_embedding)
    path_mask = batch["path_mask"].bool()
    constrained = model.constraints(
        decoded["delays_ns"],
        decoded["positive_power_weights"],
        path_mask,
        batch["conditions"],
        target_condition_validity=batch.get("condition_validity"),
        direct_path_mask=batch.get("direct_path_mask"),
    )
    csi = model._synthesize(decoded, constrained, path_mask)
    return {
        **decoded,
        "path_mask": path_mask,
        "path_delays_ns": constrained.delays_ns,
        "path_powers_linear": constrained.powers_linear,
        "direct_path_mask": constrained.direct_path_mask,
        "actual_conditions": constrained.actual_conditions,
        "constraint_feasible": constrained.feasible,
        "csi": csi,
        "latent_mean": mean,
        "latent_log_variance": log_variance,
        "latent": latent,
    }


def synthesize_oracle(
    model: PhysicsConditionedVAE,
    prediction: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    replacements: Iterable[str],
) -> torch.Tensor:
    replace = set(replacements)

    def choose(group: str, predicted_key: str, target_key: str) -> torch.Tensor:
        return target[target_key] if group in replace else prediction[predicted_key]

    def choose_angle(
        angle_group: str, predicted_key: str, target_key: str
    ) -> torch.Tensor:
        use_target = "angles" in replace or angle_group in replace
        return target[target_key] if use_target else prediction[predicted_key]

    return model.synthesizer(
        choose("delays", "path_delays_ns", "path_delays_ns"),
        choose("powers", "path_powers_linear", "path_powers_linear"),
        choose("phase", "phases_rad", "path_phases_rad"),
        choose_angle("aoa", "aoa_az_deg", "aoa_az_deg"),
        choose_angle("aoa", "aoa_el_deg", "aoa_el_deg"),
        choose_angle("aod", "aod_az_deg", "aod_az_deg"),
        choose_angle("aod", "aod_el_deg", "aod_el_deg"),
        target["path_mask"].bool(),
    )


def global_phase_aligned(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    flat_prediction = prediction.flatten(1)
    flat_target = target.flatten(1)
    inner = (flat_prediction.conj() * flat_target).sum(-1)
    rotation = torch.polar(torch.ones_like(inner.real), torch.angle(inner))
    shape = (prediction.shape[0],) + (1,) * (prediction.ndim - 1)
    return prediction * rotation.reshape(shape)


def complex_alignment_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-12,
) -> dict[str, torch.Tensor]:
    flat_prediction = prediction.flatten(1)
    flat_target = target.flatten(1)
    prediction_energy = flat_prediction.abs().square().sum(-1)
    target_energy = flat_target.abs().square().sum(-1).clamp_min(eps)
    # Clamp each energy before taking the geometric mean.  Clamping their
    # product directly at ``eps`` is too aggressive for low-power wireless
    # channels because both energies can legitimately be much smaller than 1.
    normalization = (
        prediction_energy.clamp_min(eps).sqrt()
        * target_energy.clamp_min(eps).sqrt()
    )
    inner = (flat_prediction.conj() * flat_target).sum(-1)
    return {
        "energy_ratio": prediction_energy / target_energy,
        "correlation_magnitude": inner.abs() / normalization,
        "correlation_real": inner.real / normalization,
    }


class MeanAccumulator:
    def __init__(self) -> None:
        self.sums: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)

    def add(self, name: str, values: torch.Tensor) -> None:
        detached = values.detach().float().reshape(-1)
        finite = torch.isfinite(detached)
        if not bool(finite.all()):
            raise FloatingPointError(f"Non-finite diagnostic values in {name}")
        self.sums[name] += float(detached.sum().cpu())
        self.counts[name] += int(detached.numel())

    def result(self, name: str) -> float:
        count = self.counts[name]
        return self.sums[name] / count if count else float("nan")

    def results(self) -> dict[str, float]:
        return {key: self.result(key) for key in sorted(self.sums)}


def add_path_errors(
    accumulator: MeanAccumulator,
    prediction: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
) -> None:
    mask = target["path_mask"].bool()

    def active(values: torch.Tensor) -> torch.Tensor:
        return values[mask]

    accumulator.add(
        "delay_mae_ns",
        active((prediction["path_delays_ns"] - target["path_delays_ns"]).abs()),
    )
    predicted_power_db = 10.0 * torch.log10(
        prediction["path_powers_linear"].clamp_min(1e-20)
    )
    target_power_db = 10.0 * torch.log10(
        target["path_powers_linear"].clamp_min(1e-20)
    )
    accumulator.add(
        "power_mae_db", active((predicted_power_db - target_power_db).abs())
    )
    phase_error = circular_difference_radians(
        prediction["phases_rad"], target["path_phases_rad"]
    )
    accumulator.add("phase_mae_deg", active(torch.rad2deg(phase_error).abs()))

    for key in ("aoa_az_deg", "aod_az_deg"):
        error = circular_difference_degrees(prediction[key], target[key]).abs()
        accumulator.add(f"{key}_mae_deg", active(error))
    for key in ("aoa_el_deg", "aod_el_deg"):
        accumulator.add(
            f"{key}_mae_deg", active((prediction[key] - target[key]).abs())
        )


def summarize_history(history: list[dict[str, Any]]) -> dict[str, Any]:
    if not history:
        return {"epochs": 0, "available": False}
    fields = ("loss", "csi_nmse", "phase", "angle", "delay", "power")
    rows: list[dict[str, Any]] = []
    for record in history:
        rows.append(
            {
                "epoch": int(record["epoch"]) + 1,
                "beta": float(record["beta"]),
                "learning_rate": float(record["learning_rate"]),
                "train": {
                    field: float(record["train"][field])
                    for field in fields
                    if field in record["train"]
                },
                "validation": {
                    field: float(record["validation"][field])
                    for field in fields
                    if field in record["validation"]
                },
            }
        )
    window = min(5, len(rows))
    recent = rows[-window:]
    trends: dict[str, Any] = {"window_epochs": window}
    for split in ("train", "validation"):
        for field in ("loss", "csi_nmse", "phase", "angle"):
            if field not in recent[0][split] or field not in recent[-1][split]:
                continue
            first = recent[0][split][field]
            last = recent[-1][split][field]
            relative_improvement = (first - last) / max(abs(first), 1e-12)
            trends[f"{split}_{field}"] = {
                "first": first,
                "last": last,
                "relative_improvement": relative_improvement,
                "plateau_under_1_percent": abs(relative_improvement) < 0.01,
            }
    return {
        "available": True,
        "epochs": len(rows),
        "recent": recent,
        "trends": trends,
    }


def parameter_group(name: str) -> str:
    prefixes = (
        "condition_encoder",
        "encoder.path_network",
        "encoder.posterior",
        "decoder.trunk",
        "decoder.path_count_head",
        "decoder.validity_head",
        "decoder.delay_delta_head",
        "decoder.power_head",
        "decoder.phase_head",
        "decoder.azimuth_head",
        "decoder.elevation_head",
        "decoder.direct_path_head",
    )
    for prefix in prefixes:
        if name.startswith(prefix):
            return prefix
    return name.split(".", 1)[0]


def tensor_gradient_summary(tensor: torch.Tensor) -> dict[str, Any]:
    gradient = tensor.grad
    if gradient is None:
        return {"present": False}
    finite = torch.isfinite(gradient)
    finite_values = gradient[finite]
    return {
        "present": True,
        "all_finite": bool(finite.all()),
        "finite_fraction": float(finite.float().mean()),
        "l2_norm": float(finite_values.float().norm()) if finite_values.numel() else 0.0,
        "abs_max": float(finite_values.abs().max()) if finite_values.numel() else 0.0,
    }


def gradient_diagnosis(
    model: PhysicsConditionedVAE,
    batch: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    model.zero_grad(set_to_none=True)
    output = posterior_reconstruction(model, batch, use_mean=True)
    watched = {
        key: output[key]
        for key in PREDICTED_PATH_KEYS
        if output[key].requires_grad
    }
    for tensor in watched.values():
        tensor.retain_grad()
    loss = complex_nmse_per_sample(output["csi"], batch["csi"]).mean()
    loss.backward()

    group_squared_norm: dict[str, float] = defaultdict(float)
    group_all_finite: dict[str, bool] = defaultdict(lambda: True)
    group_parameter_count: dict[str, int] = defaultdict(int)
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        group = parameter_group(name)
        gradient = parameter.grad.detach()
        group_all_finite[group] &= bool(torch.isfinite(gradient).all())
        finite_gradient = torch.nan_to_num(gradient.float())
        group_squared_norm[group] += float(finite_gradient.square().sum().cpu())
        group_parameter_count[group] += parameter.numel()

    parameter_groups = {
        group: {
            "l2_norm": math.sqrt(group_squared_norm[group]),
            "all_finite": group_all_finite[group],
            "parameter_count": group_parameter_count[group],
        }
        for group in sorted(group_squared_norm)
    }
    result = {
        "csi_nmse": float(loss.detach().cpu()),
        "output_gradients": {
            key: tensor_gradient_summary(tensor) for key, tensor in watched.items()
        },
        "parameter_groups": parameter_groups,
        "all_parameter_gradients_finite": all(
            item["all_finite"] for item in parameter_groups.values()
        ),
    }
    model.zero_grad(set_to_none=True)
    return result


def build_diagnosis(report: Mapping[str, Any]) -> list[str]:
    messages: list[str] = []
    reconstruction = report["reconstruction"]
    oracle = report["oracle_nmse"]
    baseline = float(reconstruction["posterior_mean_nmse"])

    history = report["history"]
    trend = history.get("trends", {}).get("validation_csi_nmse")
    if trend and trend["plateau_under_1_percent"]:
        messages.append(
            "Validation CSI NMSE changed by less than 1% over the recent history window; "
            "the reconstruction metric is on a plateau."
        )

    if reconstruction["posterior_sample_nmse"] - baseline > 0.05:
        messages.append(
            "Posterior sampling is materially worse than using the posterior mean; "
            "latent sampling noise is a significant reconstruction-error source."
        )

    aligned = float(reconstruction["global_phase_aligned_nmse"])
    if baseline - aligned > 0.2 * max(baseline, 1e-12):
        messages.append(
            "Removing one global phase offset per sample reduces NMSE by more than 20%; "
            "global phase ambiguity is a major error source."
        )

    single_groups = ("phase", "aoa", "aod", "delays", "powers")
    reductions = {
        group: baseline - float(oracle[f"oracle_{group}"])
        for group in single_groups
    }
    dominant = max(reductions, key=reductions.get)
    if reductions[dominant] > 0.05:
        messages.append(
            f"The largest single oracle improvement comes from replacing {dominant}: "
            f"NMSE decreases by {reductions[dominant]:.4f}."
        )
    else:
        messages.append(
            "No single parameter family produces a large oracle improvement; "
            "the CSI error is distributed across several coupled path parameters."
        )

    oracle_all = float(oracle["oracle_all"])
    if oracle_all > 1e-5:
        messages.append(
            "Ground-truth path parameters do not reproduce stored CSI closely; check "
            "preprocessing and synthesizer consistency before changing the model."
        )
    else:
        messages.append(
            "Ground-truth path parameters reproduce stored CSI, so preprocessing and "
            "the differentiable synthesizer are mutually consistent."
        )

    latent = report["latent"]
    latent_dim = int(latent["latent_dim"])
    active_units = int(latent["active_units_variance_gt_1e-2"])
    if active_units < max(2, latent_dim // 4):
        messages.append(
            f"Only {active_units}/{latent_dim} latent dimensions are active by the "
            "variance criterion; posterior collapse or weak latent usage is likely."
        )

    gradient = report.get("gradient")
    if gradient and not gradient["all_parameter_gradients_finite"]:
        messages.append(
            "At least one CSI-only model-parameter gradient is non-finite."
        )
    elif gradient:
        messages.append("CSI-only gradients are finite in the inspected batch.")
    return messages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose where a trained physics-conditioned CVAE loses CSI accuracy."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prepared-root", default=None)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-batches", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output", default="artifacts/diagnostics/checkpoint_diagnosis.json"
    )
    parser.add_argument("--skip-gradient", action="store_true")
    args = parser.parse_args()

    if args.batch_size <= 0 or args.num_batches <= 0 or args.num_workers < 0:
        parser.error("batch-size and num-batches must be positive; num-workers cannot be negative")

    seed_everything(args.seed)
    device = resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    system = SystemConfig.from_mapping(config["data"]["system"])
    model = PhysicsConditionedVAE(config["model"], system).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    prepared_root = args.prepared_root or config["data"]["prepared_root"]
    dataset = PreparedShardDataset(prepared_root, args.split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    scalar = MeanAccumulator()
    path_errors = MeanAccumulator()
    oracle_sums: dict[str, float] = defaultdict(float)
    oracle_counts: dict[str, int] = defaultdict(int)
    path_count_sums: dict[int, float] = defaultdict(float)
    path_count_counts: dict[int, int] = defaultdict(int)
    latent_means: list[torch.Tensor] = []
    latent_log_variances: list[torch.Tensor] = []
    evaluated_samples = 0

    oracle_definitions = {
        "oracle_phase": ("phase",),
        "oracle_aoa": ("aoa",),
        "oracle_aod": ("aod",),
        "oracle_angles": ("angles",),
        "oracle_delays": ("delays",),
        "oracle_powers": ("powers",),
        "oracle_phase_angles": ("phase", "angles"),
        "oracle_phase_aod": ("phase", "aod"),
        "oracle_phase_delays": ("phase", "delays"),
        "oracle_aod_delays": ("aod", "delays"),
        "oracle_phase_aod_delays": ("phase", "aod", "delays"),
        "oracle_phase_angles_delays": ("phase", "angles", "delays"),
        "oracle_phase_aod_delays_powers": (
            "phase",
            "aod",
            "delays",
            "powers",
        ),
        "oracle_all": ("phase", "angles", "delays", "powers"),
    }

    with torch.no_grad():
        for batch_index, cpu_batch in enumerate(loader):
            if batch_index >= args.num_batches:
                break
            batch = move_to_device(cpu_batch, device)
            target_csi = batch["csi"]
            sampled = posterior_reconstruction(model, batch, use_mean=False)
            mean_output = posterior_reconstruction(model, batch, use_mean=True)

            sampled_nmse = complex_nmse_per_sample(sampled["csi"], target_csi)
            mean_nmse = complex_nmse_per_sample(mean_output["csi"], target_csi)
            aligned_nmse = complex_nmse_per_sample(
                global_phase_aligned(mean_output["csi"], target_csi), target_csi
            )
            scalar.add("posterior_sample_nmse", sampled_nmse)
            scalar.add("posterior_mean_nmse", mean_nmse)
            scalar.add("global_phase_aligned_nmse", aligned_nmse)

            for name, values in complex_alignment_metrics(
                mean_output["csi"], target_csi
            ).items():
                scalar.add(name, values)

            add_path_errors(path_errors, mean_output, batch)
            latent_means.append(mean_output["latent_mean"].detach().cpu())
            latent_log_variances.append(
                mean_output["latent_log_variance"].detach().cpu()
            )

            for name, replacements in oracle_definitions.items():
                oracle_csi = synthesize_oracle(
                    model, mean_output, batch, replacements
                )
                values = complex_nmse_per_sample(oracle_csi, target_csi)
                oracle_sums[name] += float(values.sum().cpu())
                oracle_counts[name] += int(values.numel())

            for count in torch.unique(batch["path_count"]).tolist():
                count = int(count)
                selected = batch["path_count"] == count
                path_count_sums[count] += float(mean_nmse[selected].sum().cpu())
                path_count_counts[count] += int(selected.sum())
            evaluated_samples += int(target_csi.shape[0])

    if evaluated_samples == 0:
        raise RuntimeError("The diagnostic data loader produced no samples")

    means = torch.cat(latent_means, dim=0).float()
    log_variances = torch.cat(latent_log_variances, dim=0).float()
    variance_of_means = means.var(dim=0, unbiased=False)
    kl_per_dimension = 0.5 * (
        means.square() + log_variances.exp() - 1.0 - log_variances
    ).mean(dim=0)
    posterior_std = (0.5 * log_variances).exp()

    reconstruction = scalar.results()
    oracle_nmse = {
        name: oracle_sums[name] / oracle_counts[name]
        for name in sorted(oracle_sums)
    }
    oracle_nmse["posterior_mean"] = reconstruction["posterior_mean_nmse"]

    report: dict[str, Any] = {
        "format_version": 1,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)) + 1,
        "split": args.split,
        "prepared_root": str(Path(prepared_root).resolve()),
        "device": str(device),
        "batch_size": args.batch_size,
        "batches": min(args.num_batches, math.ceil(len(dataset) / args.batch_size)),
        "samples": evaluated_samples,
        "history": summarize_history(list(checkpoint.get("history", []))),
        "reconstruction": reconstruction,
        "oracle_nmse": oracle_nmse,
        "oracle_reduction_from_posterior_mean": {
            name: reconstruction["posterior_mean_nmse"] - value
            for name, value in oracle_nmse.items()
            if name != "posterior_mean"
        },
        "path_parameter_errors": path_errors.results(),
        "nmse_by_path_count": {
            str(count): {
                "nmse": path_count_sums[count] / path_count_counts[count],
                "samples": path_count_counts[count],
            }
            for count in sorted(path_count_sums)
        },
        "latent": {
            "latent_dim": int(means.shape[-1]),
            "mean_total_kl": float(kl_per_dimension.sum()),
            "mean_kl_per_dimension": float(kl_per_dimension.mean()),
            "mean_posterior_std": float(posterior_std.mean()),
            "mean_variance_of_latent_means": float(variance_of_means.mean()),
            "max_variance_of_latent_means": float(variance_of_means.max()),
            "active_units_variance_gt_1e-2": int(
                (variance_of_means > 1e-2).sum()
            ),
            "variance_of_latent_means": variance_of_means.tolist(),
            "kl_per_dimension": kl_per_dimension.tolist(),
        },
    }

    if not args.skip_gradient:
        first_batch = move_to_device(next(iter(loader)), device)
        report["gradient"] = gradient_diagnosis(model, first_batch)

    report["diagnosis"] = build_diagnosis(report)
    output_path = Path(args.output)
    atomic_json_dump(report, output_path)

    concise = {
        "checkpoint_epoch": report["checkpoint_epoch"],
        "split": report["split"],
        "samples": report["samples"],
        "reconstruction": report["reconstruction"],
        "oracle_nmse": report["oracle_nmse"],
        "oracle_reduction_from_posterior_mean": report[
            "oracle_reduction_from_posterior_mean"
        ],
        "path_parameter_errors": report["path_parameter_errors"],
        "latent": {
            key: value
            for key, value in report["latent"].items()
            if key not in {"variance_of_latent_means", "kl_per_dimension"}
        },
        "diagnosis": report["diagnosis"],
        "full_report": str(output_path.resolve()),
    }
    print(json.dumps(concise, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
