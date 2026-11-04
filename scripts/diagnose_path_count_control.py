#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_csi.data.physical_labels import compute_physical_conditions
from physics_csi.evaluation.conditions import condition_accuracy
from physics_csi.evaluation.diffusion_comparison import compare_csi_distributions
from physics_csi.evaluation.distribution import (
    full_multipath_features,
    sliced_wasserstein,
)
from physics_csi.evaluation.research_diagnostics import (
    direction_vectors,
    normalized_path_power,
)
from physics_csi.evaluation.scene_diagnostics import spatial_correlation_matrix
from physics_csi.training.trainer import resolve_device
from physics_csi.utils.io import atomic_json_dump


MODEL_SECTIONS = {
    "b3": "physics_only_baseline",
    "dp_only": "dp_only",
    "current": "correct_map",
}
PHASE_PROBE_COUNT = 16
PHASE_PROBE_SEED = 314159


def _validate_artifact(source: Mapping[str, Any]) -> None:
    if source.get("model_kind") != "b3_plus_dp_plus_aod_residual":
        raise ValueError("artifact is not a B3+DP+AoD residual evaluation")
    if source.get("aod_architecture") != "link_global":
        raise ValueError("artifact is not a Link-global AoD evaluation")
    for section in ("reference", *MODEL_SECTIONS.values()):
        if section not in source:
            raise ValueError(f"artifact is missing {section}")
        payload = source[section]
        for key in (
            "csi",
            "path_mask",
            "path_delays_ns",
            "path_powers_linear",
            "path_phases_rad",
            "aod_az_deg",
            "aod_el_deg",
            "direct_path_mask",
        ):
            if key not in payload:
                raise ValueError(f"artifact is missing {section}.{key}")
    for key in ("map_id", "conditions", "condition_validity"):
        if key not in source["reference"]:
            raise ValueError(f"artifact is missing reference.{key}")
    sample_count = int(source["reference"]["path_mask"].shape[0])
    for section in ("reference", *MODEL_SECTIONS.values()):
        if int(source[section]["path_mask"].shape[0]) != sample_count:
            raise ValueError("artifact sections do not contain the same samples")


def _select(
    payload: Mapping[str, Any], indices: torch.Tensor, device: torch.device
) -> dict[str, torch.Tensor]:
    sample_count = int(payload["path_mask"].shape[0])
    result: dict[str, torch.Tensor] = {}
    for key, value in payload.items():
        if isinstance(value, torch.Tensor) and value.shape[:1] == (sample_count,):
            result[key] = value[indices].to(device)
    return result


def _balanced_indices(
    indices: torch.Tensor,
    map_ids: torch.Tensor,
    maximum: int,
    seed: int,
) -> torch.Tensor:
    """Round-robin Maps so one well-supported Map cannot dominate an L group."""
    if indices.numel() <= maximum:
        return indices
    generator = torch.Generator().manual_seed(seed)
    pools: dict[int, list[int]] = {}
    for map_id in sorted(int(value) for value in torch.unique(map_ids[indices])):
        local = indices[map_ids[indices] == map_id]
        order = torch.randperm(local.numel(), generator=generator)
        pools[map_id] = [int(value) for value in local[order]]
    selected: list[int] = []
    offset = 0
    while len(selected) < maximum:
        added = False
        for map_id in sorted(pools):
            if offset < len(pools[map_id]):
                selected.append(pools[map_id][offset])
                added = True
                if len(selected) == maximum:
                    break
        if not added:
            break
        offset += 1
    return torch.tensor(selected, dtype=torch.long)


def _safe_summary(values: torch.Tensor) -> dict[str, Any]:
    values = values.detach().float().cpu()
    if not values.numel():
        return {"count": 0}
    quantiles = torch.quantile(
        values, torch.tensor([0.0, 0.05, 0.5, 0.95, 1.0])
    )
    return {
        "count": int(values.numel()),
        "minimum": float(quantiles[0]),
        "p05": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p95": float(quantiles[3]),
        "maximum": float(quantiles[4]),
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
    }


def _support_audit(cache: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split, split_payload in cache["splits"].items():
        reference = split_payload["reference"]
        mask = reference["path_mask"].bool()
        count = mask.sum(-1).long()
        conditions = reference["conditions"].float()
        validity = reference["condition_validity"].bool()
        map_ids = reference["map_id"].long()
        per_l: dict[str, Any] = {}
        for path_count in range(1, mask.shape[1] + 1):
            selected = count == path_count
            k_selected = selected & validity[:, 2]
            per_l[str(path_count)] = {
                "samples": int(selected.sum()),
                "maps": int(torch.unique(map_ids[selected]).numel()),
                "los_samples": int((selected & (conditions[:, 3] >= 0.5)).sum()),
                "nlos_samples": int((selected & (conditions[:, 3] < 0.5)).sum()),
                "valid_k_samples": int(k_selected.sum()),
                "total_power_db": _safe_summary(conditions[selected, 0]),
                "delay_spread_ns": _safe_summary(conditions[selected, 1]),
                "k_factor_db_valid_only": _safe_summary(conditions[k_selected, 2]),
            }
        result[split] = {
            "samples": int(mask.shape[0]),
            "path_count_histogram": {
                str(key): int(value)
                for key, value in sorted(Counter(count.tolist()).items())
            },
            "per_l": per_l,
        }
    return result


def _condition_control(
    payload: Mapping[str, torch.Tensor],
    target: torch.Tensor,
    target_validity: torch.Tensor,
    requested_l: int,
) -> dict[str, Any]:
    mask = payload["path_mask"].bool()
    actual_l = mask.sum(-1).long()
    actual, actual_validity = compute_physical_conditions(
        payload["path_powers_linear"].float(),
        payload["path_delays_ns"].float(),
        mask,
        payload["direct_path_mask"].bool(),
    )
    validity = target_validity.bool() & actual_validity.bool()
    accuracy = condition_accuracy(target.float(), actual, validity)
    errors: dict[str, Any] = {}
    for index, name in enumerate(
        ("total_power_db", "delay_spread_ns", "k_factor_db")
    ):
        valid = validity[:, index]
        absolute = (target[valid, index] - actual[valid, index]).abs()
        errors[f"{name}_absolute_error"] = _safe_summary(absolute)
    inactive = ~mask
    inactive_power = payload["path_powers_linear"].float()[inactive]
    inactive_delay = payload["path_delays_ns"].float()[inactive]
    return {
        "requested_l": requested_l,
        "samples": int(actual_l.numel()),
        "path_count_accuracy": float((actual_l == requested_l).float().mean()),
        "actual_path_count_histogram": {
            str(key): int(value)
            for key, value in sorted(Counter(actual_l.cpu().tolist()).items())
        },
        "finite_csi_rate": float(
            (torch.isfinite(payload["csi"].real) & torch.isfinite(payload["csi"].imag))
            .flatten(1)
            .all(-1)
            .float()
            .mean()
        ),
        "inactive_power_maximum_absolute": (
            float(inactive_power.abs().max()) if inactive_power.numel() else 0.0
        ),
        "inactive_delay_maximum_absolute": (
            float(inactive_delay.abs().max()) if inactive_delay.numel() else 0.0
        ),
        "condition_accuracy": accuracy,
        "condition_absolute_error_audit": errors,
    }


def _standardized_swd(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    projections: int,
    seed: int,
) -> float:
    reference = reference.float()
    candidate = candidate.float()
    mean = reference.mean(0, keepdim=True)
    raw_scale = reference.std(0, unbiased=False, keepdim=True)
    positive = raw_scale[torch.isfinite(raw_scale) & (raw_scale > 0)]
    typical = positive.median() if positive.numel() else torch.ones((), device=reference.device)
    floor = torch.maximum(typical * 0.05, torch.tensor(1e-6, device=reference.device))
    scale = torch.nan_to_num(raw_scale).clamp_min(floor)
    return float(
        sliced_wasserstein(
            (candidate - mean) / scale,
            (reference - mean) / scale,
            projections=projections,
            seed=seed,
        )
    )


def _relative_delay_power(payload: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    mask = payload["path_mask"].bool()
    delay = payload["path_delays_ns"].float()
    first = torch.where(mask, delay, torch.full_like(delay, torch.inf)).amin(-1)
    relative = (delay - first.unsqueeze(-1)).clamp_min(0.0) * mask
    power = payload["path_powers_linear"].float().clamp_min(0.0) * mask
    fraction = power / power.sum(-1, keepdim=True).clamp_min(1e-20)
    return mask, relative, fraction


def _delay_power_path(payload: Mapping[str, torch.Tensor]) -> torch.Tensor:
    mask, delay, fraction = _relative_delay_power(payload)
    return torch.stack(
        [delay, 10.0 * torch.log10(fraction.clamp_min(1e-20))], dim=-1
    )[mask]


def _delay_power_link(payload: Mapping[str, torch.Tensor]) -> torch.Tensor:
    mask, delay, fraction = _relative_delay_power(payload)
    order = torch.argsort(
        torch.where(mask, payload["path_delays_ns"].float(), torch.full_like(delay, torch.inf)),
        dim=-1,
        stable=True,
    )
    ordered_mask = mask.gather(1, order)
    return torch.cat(
        [
            ordered_mask.float(),
            delay.gather(1, order) * ordered_mask,
            fraction.gather(1, order) * ordered_mask,
        ],
        dim=-1,
    )


def _power_aod_path(payload: Mapping[str, torch.Tensor]) -> torch.Tensor:
    mask = payload["path_mask"].bool()
    fraction = normalized_path_power(payload)
    direction = direction_vectors(payload["aod_az_deg"], payload["aod_el_deg"])
    power_db = 10.0 * torch.log10(fraction.clamp_min(1e-20))
    return torch.cat([direction, power_db.unsqueeze(-1)], dim=-1)[mask]


def _power_aod_link(payload: Mapping[str, torch.Tensor]) -> torch.Tensor:
    mask = payload["path_mask"].bool()
    fraction = normalized_path_power(payload)
    direction = direction_vectors(payload["aod_az_deg"], payload["aod_el_deg"])
    order = torch.argsort(
        fraction.masked_fill(~mask, -torch.inf), dim=-1, descending=True, stable=True
    )
    ordered_mask = mask.gather(1, order)
    ordered_fraction = fraction.gather(1, order) * ordered_mask
    ordered_direction = direction.gather(
        1, order.unsqueeze(-1).expand(-1, -1, direction.shape[-1])
    ) * ordered_mask.unsqueeze(-1)
    return torch.cat(
        [ordered_mask.float(), ordered_fraction, ordered_direction.flatten(1)], dim=-1
    )


def _phase_normalize(csi: torch.Tensor) -> torch.Tensor:
    flat = csi.flatten(1)
    generator = torch.Generator(device=csi.device).manual_seed(PHASE_PROBE_SEED)
    real = torch.randn(
        PHASE_PROBE_COUNT,
        flat.shape[1],
        generator=generator,
        device=csi.device,
        dtype=csi.real.dtype,
    )
    imag = torch.randn(
        PHASE_PROBE_COUNT,
        flat.shape[1],
        generator=generator,
        device=csi.device,
        dtype=csi.real.dtype,
    )
    probes = torch.complex(real, imag)
    probes = probes / probes.abs().square().sum(-1, keepdim=True).sqrt().clamp_min(1e-20)
    anchors = flat @ probes.conj().transpose(0, 1)
    anchor = anchors.gather(1, anchors.abs().argmax(-1, keepdim=True))
    unit = anchor / anchor.abs().clamp_min(1e-20)
    shape = (csi.shape[0],) + (1,) * (csi.ndim - 1)
    return csi * unit.conj().reshape(shape)


def _energy_normalized_features(csi: torch.Tensor) -> torch.Tensor:
    flat = csi.flatten(1)
    flat = flat / flat.abs().square().sum(-1, keepdim=True).sqrt().clamp_min(1e-20)
    return torch.cat([flat.real, flat.imag], dim=-1).float()


def _projected_csi_swd(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    phase_invariant: bool,
    dimension: int,
    projections: int,
    seeds: tuple[int, ...],
) -> float:
    if phase_invariant:
        candidate = _phase_normalize(candidate)
        reference = _phase_normalize(reference)
    first = _energy_normalized_features(candidate)
    second = _energy_normalized_features(reference)
    mean = second.mean(0, keepdim=True)
    raw_scale = second.std(0, unbiased=False, keepdim=True)
    positive = raw_scale[torch.isfinite(raw_scale) & (raw_scale > 0)]
    typical = positive.median() if positive.numel() else torch.ones((), device=second.device)
    floor = torch.maximum(typical * 0.05, torch.tensor(1e-6, device=second.device))
    scale = torch.nan_to_num(raw_scale).clamp_min(floor)
    first = (first - mean) / scale
    second = (second - mean) / scale
    values: list[float] = []
    for seed in seeds:
        generator = torch.Generator(device=second.device).manual_seed(seed)
        signs = torch.randint(
            0,
            2,
            (second.shape[1], dimension),
            generator=generator,
            device=second.device,
            dtype=torch.int8,
        ).float()
        basis = (signs * 2.0 - 1.0) / math.sqrt(second.shape[1])
        values.append(
            float(
                sliced_wasserstein(
                    first @ basis,
                    second @ basis,
                    projections=projections,
                    seed=seed + 1000,
                )
            )
        )
    return sum(values) / len(values)


def _tx_spatial_relative_frobenius(
    candidate: torch.Tensor, reference: torch.Tensor
) -> float:
    first = spatial_correlation_matrix(candidate)
    second = spatial_correlation_matrix(reference)
    return float(
        torch.linalg.matrix_norm(first - second)
        / torch.linalg.matrix_norm(second).clamp_min(1e-20)
    )


def _fidelity_metrics(
    candidate: Mapping[str, torch.Tensor],
    reference: Mapping[str, torch.Tensor],
    *,
    swd_projections: int,
    projection_dimension: int,
    projection_seeds: tuple[int, ...],
    seed: int,
) -> dict[str, float]:
    csi = compare_csi_distributions(candidate["csi"], reference["csi"])
    result = {
        "path_delay_power_swd": _standardized_swd(
            _delay_power_path(candidate),
            _delay_power_path(reference),
            projections=swd_projections,
            seed=seed,
        ),
        "link_delay_power_swd": _standardized_swd(
            _delay_power_link(candidate),
            _delay_power_link(reference),
            projections=swd_projections,
            seed=seed + 1,
        ),
        "path_power_aod_swd": _standardized_swd(
            _power_aod_path(candidate),
            _power_aod_path(reference),
            projections=swd_projections,
            seed=seed + 2,
        ),
        "link_power_aod_swd": _standardized_swd(
            _power_aod_link(candidate),
            _power_aod_link(reference),
            projections=swd_projections,
            seed=seed + 3,
        ),
        "full_multipath_swd": _standardized_swd(
            full_multipath_features(dict(candidate)),
            full_multipath_features(dict(reference)),
            projections=swd_projections,
            seed=seed + 4,
        ),
        "raw_full_csi_projected_swd": _projected_csi_swd(
            candidate["csi"],
            reference["csi"],
            phase_invariant=False,
            dimension=projection_dimension,
            projections=swd_projections,
            seeds=projection_seeds,
        ),
        "phase_invariant_full_csi_projected_swd": _projected_csi_swd(
            candidate["csi"],
            reference["csi"],
            phase_invariant=True,
            dimension=projection_dimension,
            projections=swd_projections,
            seeds=projection_seeds,
        ),
        "tx_spatial_correlation_relative_frobenius": _tx_spatial_relative_frobenius(
            candidate["csi"], reference["csi"]
        ),
        "csi_summary_swd": csi["csi_summary_sliced_wasserstein"],
        "pdp_swd": csi["pdp_sliced_wasserstein"],
        "generated_pairwise_diversity": csi["generated_pairwise_diversity"],
        "reference_pairwise_diversity": csi["reference_pairwise_diversity"],
    }
    reference_diversity = result["reference_pairwise_diversity"]
    result["csi_diversity_ratio_to_real"] = (
        result["generated_pairwise_diversity"] / reference_diversity
        if math.isfinite(reference_diversity) and reference_diversity > 1e-20
        else float("nan")
    )
    return result


def _relative_improvement(
    current: Mapping[str, float], baseline: Mapping[str, float]
) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for metric, baseline_value in baseline.items():
        if metric in {
            "generated_pairwise_diversity",
            "reference_pairwise_diversity",
            "csi_diversity_ratio_to_real",
        }:
            continue
        current_value = current[metric]
        result[metric] = (
            (baseline_value - current_value) / abs(baseline_value)
            if math.isfinite(baseline_value)
            and math.isfinite(current_value)
            and abs(baseline_value) > 1e-20
            else None
        )
    return result


def _real_real_floor(
    reference: Mapping[str, torch.Tensor],
    *,
    swd_projections: int,
    projection_dimension: int,
    projection_seeds: tuple[int, ...],
    seed: int,
) -> dict[str, float] | None:
    count = int(reference["path_mask"].shape[0])
    half = count // 2
    if half < 8:
        return None
    generator = torch.Generator(device=reference["path_mask"].device).manual_seed(seed)
    order = torch.randperm(count, generator=generator, device=reference["path_mask"].device)
    first_indices = order[:half]
    second_indices = order[half : half * 2]
    first = {key: value[first_indices] for key, value in reference.items()}
    second = {key: value[second_indices] for key, value in reference.items()}
    return _fidelity_metrics(
        first,
        second,
        swd_projections=swd_projections,
        projection_dimension=projection_dimension,
        projection_seeds=projection_seeds,
        seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit forced path-count control and per-L generation fidelity."
    )
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--maximum-samples-per-l", type=int, default=1024)
    parser.add_argument("--minimum-samples-per-l", type=int, default=32)
    parser.add_argument("--swd-projections", type=int, default=128)
    parser.add_argument("--projection-dimension", type=int, default=64)
    parser.add_argument("--projection-seeds", default="0,1,2")
    args = parser.parse_args()
    if args.maximum_samples_per_l < 2:
        parser.error("maximum samples per L must be at least two")
    if args.minimum_samples_per_l < 2:
        parser.error("minimum samples per L must be at least two")
    projection_seeds = tuple(
        int(value.strip()) for value in args.projection_seeds.split(",") if value.strip()
    )
    if not projection_seeds:
        parser.error("at least one projection seed is required")

    device = resolve_device(args.device)
    artifact = torch.load(args.artifact, map_location="cpu", weights_only=False)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    _validate_artifact(artifact)
    if cache.get("scene_plan_fingerprint") != artifact.get("scene_plan_fingerprint"):
        raise ValueError("artifact and cache use different scene plans")

    reference_cpu = artifact["reference"]
    reference_count = reference_cpu["path_mask"].bool().sum(-1).long()
    map_ids = reference_cpu["map_id"].long()
    max_paths = int(reference_cpu["path_mask"].shape[1])
    per_l: dict[str, Any] = {}
    with torch.inference_mode():
        for path_count in range(1, max_paths + 1):
            all_indices = torch.nonzero(
                reference_count == path_count, as_tuple=False
            ).flatten()
            selected_indices = _balanced_indices(
                all_indices,
                map_ids,
                args.maximum_samples_per_l,
                args.seed + path_count,
            )
            section: dict[str, Any] = {
                "available_test_samples": int(all_indices.numel()),
                "available_test_maps": int(torch.unique(map_ids[all_indices]).numel()),
                "evaluated_samples": int(selected_indices.numel()),
                "selection": "Map-balanced round-robin without replacement",
            }
            if selected_indices.numel() < args.minimum_samples_per_l:
                section["status"] = "insufficient_support"
                section["minimum_required"] = args.minimum_samples_per_l
                per_l[str(path_count)] = section
                continue

            reference = _select(reference_cpu, selected_indices, device)
            target = reference["conditions"].float()
            target_validity = reference["condition_validity"].bool()
            model_payloads = {
                label: _select(artifact[source], selected_indices, device)
                for label, source in MODEL_SECTIONS.items()
            }
            control = {
                label: _condition_control(
                    payload, target, target_validity, path_count
                )
                for label, payload in model_payloads.items()
            }
            # Reuse identical SWD directions for every candidate so B3/current
            # differences cannot be caused by Monte Carlo projection noise.
            fidelity = {
                label: _fidelity_metrics(
                    payload,
                    reference,
                    swd_projections=args.swd_projections,
                    projection_dimension=args.projection_dimension,
                    projection_seeds=projection_seeds,
                    seed=args.seed + path_count * 100,
                )
                for label, payload in model_payloads.items()
            }
            section.update(
                {
                    "status": "evaluated",
                    "control": control,
                    "fidelity_to_real": fidelity,
                    "real_vs_real_finite_sample_floor": _real_real_floor(
                        reference,
                        swd_projections=args.swd_projections,
                        projection_dimension=args.projection_dimension,
                        projection_seeds=projection_seeds,
                        seed=args.seed + path_count * 1000,
                    ),
                    "current_relative_improvement_over_b3": _relative_improvement(
                        fidelity["current"], fidelity["b3"]
                    ),
                }
            )
            per_l[str(path_count)] = section
            print(
                {
                    "event": "path_count_group_complete",
                    "path_count": path_count,
                    "samples": int(selected_indices.numel()),
                    "maps": int(torch.unique(map_ids[selected_indices]).numel()),
                },
                flush=True,
            )

    report = {
        "format_version": 1,
        "experiment": "forced_path_count_control_and_fidelity",
        "artifact": str(Path(args.artifact).resolve()),
        "cache": str(Path(args.cache).resolve()),
        "artifact_split": artifact.get("split"),
        "artifact_samples": int(reference_count.numel()),
        "max_paths": max_paths,
        "metric_direction": (
            "All fidelity distances are lower-is-better; diversity ratio is best near one. "
            "Positive current_relative_improvement_over_b3 favors the current model."
        ),
        "support_audit": _support_audit(cache),
        "per_l": per_l,
        "success_criteria": {
            "control": (
                "For every physically legal L, path_count_accuracy and finite_csi_rate must be 1, "
                "with negligible physical-condition error and no inactive-slot leakage."
            ),
            "fidelity": (
                "Assess every L separately. Current-model Delay--Power, Power--AoD, PDP, spatial "
                "and full-CSI distances should not regress materially from B3 and should be judged "
                "relative to the Real--Real finite-sample floor."
            ),
            "support": (
                "Do not claim reliable control for L/Map/physical-condition regions with sparse "
                "training support, even when the structural path-count constraint is exact."
            ),
        },
        "cautions": [
            "This evaluates L values that occur naturally in the held-out test anchors; it does not prove arbitrary cross-combinations are in distribution.",
            "The evaluation artifact used one frozen B3 generation seed. Repeat with independent generation seeds before making a final stability claim.",
            "Real--Real halves are independent samples but are only approximately balanced in their remaining physical conditions.",
        ],
    }
    atomic_json_dump(report, args.output)
    print(
        {
            "event": "path_count_control_diagnostic_complete",
            "output": str(Path(args.output).resolve()),
            "evaluated_l": [
                int(value)
                for value, item in per_l.items()
                if item.get("status") == "evaluated"
            ],
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
