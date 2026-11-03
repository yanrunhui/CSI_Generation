#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_csi.config import load_yaml
from physics_csi.data.physical_labels import compute_physical_conditions
from physics_csi.evaluation.distribution import sliced_wasserstein
from physics_csi.evaluation.research_diagnostics import (
    aod_statistical_features,
    direction_vectors,
    normalized_path_power,
)
from physics_csi.evaluation.scene_diagnostics import spatial_correlation_matrix
from physics_csi.physics.synthesizer import MIMOOFDMSynthesizer, SystemConfig
from physics_csi.training.trainer import resolve_device
from physics_csi.utils.io import atomic_json_dump


PATH_KEYS = (
    "path_mask",
    "path_delays_ns",
    "path_powers_linear",
    "path_phases_rad",
    "aoa_az_deg",
    "aoa_el_deg",
    "aod_az_deg",
    "aod_el_deg",
)

REPAIR_GROUPS = {
    "aod": ("aod_az_deg", "aod_el_deg"),
    "aoa": ("aoa_az_deg", "aoa_el_deg"),
    "all_angles": ("aoa_az_deg", "aoa_el_deg", "aod_az_deg", "aod_el_deg"),
    "power": ("path_powers_linear",),
    "delay": ("path_delays_ns",),
    "delay_power_joint": ("path_delays_ns", "path_powers_linear"),
    "power_aod_joint": ("path_powers_linear", "aod_az_deg", "aod_el_deg"),
    "path_phase": ("path_phases_rad",),
    "all_except_phase": (
        "path_delays_ns",
        "path_powers_linear",
        "aoa_az_deg",
        "aoa_el_deg",
        "aod_az_deg",
        "aod_el_deg",
    ),
}


def _subset(payload: dict[str, Any], indices: torch.Tensor, device: torch.device) -> dict[str, Any]:
    count = int(next(value for value in payload.values() if isinstance(value, torch.Tensor)).shape[0])
    return {
        key: value[indices].to(device) if isinstance(value, torch.Tensor) and value.shape[:1] == (count,) else value
        for key, value in payload.items()
    }


def _standardized_swd(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    mean = reference.mean(0, keepdim=True)
    std = reference.std(0, unbiased=False, keepdim=True).clamp_min(1e-6)
    return float(sliced_wasserstein((candidate - mean) / std, (reference - mean) / std))


def _delay_power_points(payload: dict[str, torch.Tensor]) -> torch.Tensor:
    mask = payload["path_mask"].bool()
    fraction = normalized_path_power(payload)
    delay = payload["path_delays_ns"].float()
    first = torch.where(mask, delay, torch.full_like(delay, torch.inf)).amin(-1)
    relative_delay = (delay - first.unsqueeze(-1)).clamp_min(0.0)
    points = torch.stack(
        [relative_delay, 10.0 * torch.log10(fraction.clamp_min(1e-20))], dim=-1
    )
    return points[mask]


def _circular_path_features(payload: dict[str, torch.Tensor], key: str, degrees: bool) -> torch.Tensor:
    mask = payload["path_mask"].bool()
    values = payload[key][mask].float()
    if degrees:
        values = torch.deg2rad(values)
    return torch.stack([values.sin(), values.cos()], dim=-1)


def _delay_phase_points(payload: dict[str, torch.Tensor]) -> torch.Tensor:
    mask = payload["path_mask"].bool()
    delays = payload["path_delays_ns"].float()
    first = torch.where(mask, delays, torch.full_like(delays, torch.inf)).amin(-1)
    relative_delay = (delays - first.unsqueeze(-1)).clamp_min(0.0)
    phase = payload["path_phases_rad"].float()
    return torch.stack([relative_delay, phase.sin(), phase.cos()], dim=-1)[mask]


def _complex_frequency_correlation(csi: torch.Tensor, lags: tuple[int, ...]) -> torch.Tensor:
    samples = csi.flatten(0, 2)
    energy = samples.abs().square().mean().clamp_min(1e-20)
    return torch.stack(
        [(samples[:, :-lag].conj() * samples[:, lag:]).mean() / energy for lag in lags if lag < csi.shape[-1]]
    )


def _complex_distance(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    magnitude_mae = (candidate.abs() - reference.abs()).abs().mean()
    valid = reference.abs() > 1e-3
    phase_mae = (
        torch.angle(candidate[valid] * reference[valid].conj()).abs().mean()
        if bool(valid.any())
        else torch.zeros((), device=reference.device)
    )
    return {"magnitude_mae": float(magnitude_mae), "phase_mae_rad": float(phase_mae)}


def _spatial_metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    candidate_matrix = spatial_correlation_matrix(candidate)
    reference_matrix = spatial_correlation_matrix(reference)
    difference = candidate_matrix - reference_matrix
    off_diagonal = ~torch.eye(
        reference_matrix.shape[0], dtype=torch.bool, device=reference_matrix.device
    )
    phase_valid = off_diagonal & (reference_matrix.abs() > 1e-3)
    phase_error = (
        torch.angle(candidate_matrix[phase_valid] * reference_matrix[phase_valid].conj()).abs().mean()
        if bool(phase_valid.any())
        else torch.zeros((), device=reference_matrix.device)
    )
    return {
        "relative_frobenius": float(
            torch.linalg.matrix_norm(difference)
            / torch.linalg.matrix_norm(reference_matrix).clamp_min(1e-12)
        ),
        "magnitude_mae": float((candidate_matrix.abs() - reference_matrix.abs()).abs().mean()),
        "phase_mae_rad": float(phase_error),
    }


def _parameter_metrics(
    candidate: dict[str, torch.Tensor], reference: dict[str, torch.Tensor]
) -> dict[str, Any]:
    candidate_mask = candidate["path_mask"].bool()
    reference_mask = reference["path_mask"].bool()
    candidate_aod = direction_vectors(candidate["aod_az_deg"], candidate["aod_el_deg"])[candidate_mask]
    reference_aod = direction_vectors(reference["aod_az_deg"], reference["aod_el_deg"])[reference_mask]
    candidate_power_aod = aod_statistical_features(candidate)
    reference_power_aod = aod_statistical_features(reference)
    candidate_delay_power = _delay_power_points(candidate)
    reference_delay_power = _delay_power_points(reference)
    candidate_delay_phase = _delay_phase_points(candidate)
    reference_delay_phase = _delay_phase_points(reference)
    frequency_candidate = _complex_frequency_correlation(candidate["csi"], (1, 2, 4, 8, 16))
    frequency_reference = _complex_frequency_correlation(reference["csi"], (1, 2, 4, 8, 16))
    return {
        "path_count_total_variation": float(
            0.5
            * (
                torch.bincount(candidate_mask.sum(-1).long(), minlength=candidate_mask.shape[1] + 1).float()
                / candidate_mask.shape[0]
                - torch.bincount(reference_mask.sum(-1).long(), minlength=reference_mask.shape[1] + 1).float()
                / reference_mask.shape[0]
            ).abs().sum()
        ),
        "aod_unweighted_direction_sliced_wasserstein": float(
            sliced_wasserstein(candidate_aod, reference_aod)
        ),
        "aod_power_weighted_link_statistics_standardized_sliced_wasserstein": _standardized_swd(
            candidate_power_aod, reference_power_aod
        ),
        "within_link_aod_clustering_standardized_sliced_wasserstein": _standardized_swd(
            candidate_power_aod[:, -3:], reference_power_aod[:, -3:]
        ),
        "delay_power_joint_standardized_sliced_wasserstein": _standardized_swd(
            candidate_delay_power, reference_delay_power
        ),
        "path_phase_circular_sliced_wasserstein": float(
            sliced_wasserstein(
                _circular_path_features(candidate, "path_phases_rad", False),
                _circular_path_features(reference, "path_phases_rad", False),
            )
        ),
        "delay_phase_joint_standardized_sliced_wasserstein": _standardized_swd(
            candidate_delay_phase, reference_delay_phase
        ),
        "frequency_correlation": _complex_distance(frequency_candidate, frequency_reference),
        "spatial_correlation": _spatial_metrics(candidate["csi"], reference["csi"]),
        "note": (
            "Spatial-correlation phase is computed from the aggregate Tx correlation matrix. "
            "It is not identified with marginal path-phase error."
        ),
    }


def _condition_view(payload: dict[str, torch.Tensor], generated: bool) -> tuple[torch.Tensor, torch.Tensor]:
    if generated:
        return payload["actual_conditions"].float(), payload["actual_condition_validity"].bool()
    return payload["conditions"].float(), payload["condition_validity"].bool()


def _match_reference_donors(
    candidate: dict[str, torch.Tensor], reference: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    candidate_conditions, candidate_validity = _condition_view(candidate, True)
    reference_conditions, reference_validity = _condition_view(reference, False)
    pooled = torch.cat([candidate_conditions[:, :3], reference_conditions[:, :3]])
    scale = pooled.std(0, unbiased=False).clamp_min(1e-6)
    candidate_count = candidate["path_mask"].sum(-1)
    reference_count = reference["path_mask"].sum(-1)
    candidate_los = candidate_conditions[:, 3] >= 0.5
    reference_los = reference_conditions[:, 3] >= 0.5
    candidate_normalized = candidate_conditions[:, :3] / scale
    reference_normalized = reference_conditions[:, :3] / scale
    candidate_normalized[:, 2] = torch.where(
        candidate_validity[:, 2], candidate_normalized[:, 2], 0.0
    )
    reference_normalized[:, 2] = torch.where(
        reference_validity[:, 2], reference_normalized[:, 2], 0.0
    )
    matched_chunks: list[torch.Tensor] = []
    donor_chunks: list[torch.Tensor] = []
    distance_chunks: list[torch.Tensor] = []
    groups = sorted(
        {
            (int(candidate_count[index]), bool(candidate_los[index]), bool(candidate_validity[index, 2]))
            for index in range(candidate_count.numel())
        }
    )
    for path_count, los, k_valid in groups:
        candidate_indices = torch.nonzero(
            (candidate_count == path_count)
            & (candidate_los == los)
            & (candidate_validity[:, 2] == k_valid),
            as_tuple=False,
        ).flatten()
        reference_indices = torch.nonzero(
            (reference_count == path_count)
            & (reference_los == los)
            & (reference_validity[:, 2] == k_valid),
            as_tuple=False,
        ).flatten()
        if reference_indices.numel() == 0:
            continue
        distance = torch.cdist(
            candidate_normalized[candidate_indices], reference_normalized[reference_indices]
        )
        nearest_distance, nearest_local = distance.min(dim=1)
        matched_chunks.append(candidate_indices)
        donor_chunks.append(reference_indices[nearest_local])
        distance_chunks.append(nearest_distance)
    if not matched_chunks:
        empty = torch.empty(0, dtype=torch.long, device=candidate_conditions.device)
        return empty, empty, torch.empty(0, device=candidate_conditions.device)
    matched = torch.cat(matched_chunks)
    order = torch.argsort(matched)
    return (
        matched[order],
        torch.cat(donor_chunks)[order],
        torch.cat(distance_chunks)[order],
    )


def _synthesize(
    payload: dict[str, torch.Tensor], synthesizer: MIMOOFDMSynthesizer, batch_size: int
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for start in range(0, payload["path_mask"].shape[0], batch_size):
        stop = min(start + batch_size, payload["path_mask"].shape[0])
        chunks.append(
            synthesizer(
                payload["path_delays_ns"][start:stop],
                payload["path_powers_linear"][start:stop],
                payload["path_phases_rad"][start:stop],
                payload["aoa_az_deg"][start:stop],
                payload["aoa_el_deg"][start:stop],
                payload["aod_az_deg"][start:stop],
                payload["aod_el_deg"][start:stop],
                payload["path_mask"][start:stop],
            )
        )
    return torch.cat(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Separate real/generated path-parameter discrepancies and attribute spatial error "
            "with physics-synthesizer counterfactual repairs."
        )
    )
    parser.add_argument("--generated", required=True)
    parser.add_argument("--candidate", default="correct_map")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--maximum-samples", type=int, default=2048)
    parser.add_argument("--minimum-per-map", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min(args.maximum_samples, args.minimum_per_map, args.batch_size) < 1:
        parser.error("sample limits and batch size must be positive")

    device = resolve_device(args.device)
    source = torch.load(args.generated, map_location="cpu", weights_only=False)
    if "reference" not in source or args.candidate not in source:
        raise ValueError(f"artifact must contain reference and {args.candidate!r}")
    reference_cpu = source["reference"]
    candidate_cpu = source[args.candidate]
    if not isinstance(reference_cpu, dict) or not isinstance(candidate_cpu, dict):
        raise ValueError("reference and selected candidate must both be tensor dictionaries")
    required_reference = set(PATH_KEYS) | {"csi", "conditions", "condition_validity", "map_id"}
    required_candidate = set(PATH_KEYS) | {
        "csi",
        "actual_conditions",
        "actual_condition_validity",
        "direct_path_mask",
        "target_conditions",
        "target_condition_validity",
    }
    if missing := sorted(required_reference - reference_cpu.keys()):
        raise ValueError(f"reference payload is missing {missing}")
    if missing := sorted(required_candidate - candidate_cpu.keys()):
        raise ValueError(f"candidate payload is missing {missing}")
    sample_count = reference_cpu["csi"].shape[0]
    if candidate_cpu["csi"].shape[0] != sample_count:
        raise ValueError("reference and candidate sample counts differ")
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.arange(sample_count)
    if sample_count > args.maximum_samples:
        indices = torch.randperm(sample_count, generator=generator)[: args.maximum_samples].sort().values
    reference = _subset(reference_cpu, indices, device)
    candidate = _subset(candidate_cpu, indices, device)

    overall = _parameter_metrics(candidate, reference)
    per_map: dict[str, Any] = {}
    for map_id in torch.unique(reference["map_id"].long(), sorted=True):
        keep = reference["map_id"].long() == map_id
        if int(keep.sum()) < args.minimum_per_map:
            per_map[f"map_{int(map_id)}"] = {
                "samples": int(keep.sum()), "status": "insufficient_samples"
            }
            continue
        selected = torch.nonzero(keep, as_tuple=False).flatten()
        per_map[f"map_{int(map_id)}"] = {
            "samples": int(keep.sum()),
            "metrics": _parameter_metrics(
                _subset(candidate, selected, device), _subset(reference, selected, device)
            ),
        }

    matched_candidate, donor_indices, match_distances = _match_reference_donors(candidate, reference)
    if matched_candidate.numel() < 2:
        raise ValueError("fewer than two same-path-count/LoS/K-validity donor matches")
    candidate_matched = _subset(candidate, matched_candidate, device)
    donor_reference = _subset(reference, donor_indices, device)
    system = SystemConfig.from_mapping(load_yaml(args.data_config)["data"]["system"])
    synthesizer = MIMOOFDMSynthesizer(system).to(device).eval()
    baseline_spatial = _spatial_metrics(candidate_matched["csi"], donor_reference["csi"])
    repairs: dict[str, Any] = {}
    with torch.inference_mode():
        for name, keys in REPAIR_GROUPS.items():
            repaired = {key: value for key, value in candidate_matched.items()}
            for key in keys:
                repaired[key] = donor_reference[key].clone()
            repaired["csi"] = _synthesize(repaired, synthesizer, args.batch_size)
            spatial = _spatial_metrics(repaired["csi"], donor_reference["csi"])
            actual_conditions, _ = compute_physical_conditions(
                repaired["path_powers_linear"],
                repaired["path_delays_ns"],
                repaired["path_mask"],
                repaired["direct_path_mask"],
            )
            target_conditions = candidate_matched["target_conditions"]
            valid = candidate_matched["target_condition_validity"].bool()
            drift: dict[str, float | None] = {}
            for index, field in enumerate(("total_power_db", "delay_spread_ns", "k_factor_db")):
                selected_valid = valid[:, index]
                drift[field + "_mae"] = (
                    float((actual_conditions[selected_valid, index] - target_conditions[selected_valid, index]).abs().mean())
                    if bool(selected_valid.any())
                    else None
                )
            repairs[name] = {
                "replaced_parameters": list(keys),
                "spatial_correlation": spatial,
                "improvement_baseline_minus_repaired": {
                    key: baseline_spatial[key] - spatial[key] for key in baseline_spatial
                },
                "physical_condition_drift_after_repair": drift,
            }

    ranked = sorted(
        (
            {
                "repair": name,
                "relative_frobenius_improvement": values["improvement_baseline_minus_repaired"][
                    "relative_frobenius"
                ],
            }
            for name, values in repairs.items()
        ),
        key=lambda item: item["relative_frobenius_improvement"],
        reverse=True,
    )
    report = {
        "format_version": 1,
        "diagnostic": "physical_parameter_distribution_and_spatial_attribution",
        "source": str(Path(args.generated).resolve()),
        "candidate": args.candidate,
        "samples": int(indices.numel()),
        "device": str(device),
        "real_vs_generated_parameter_metrics": overall,
        "per_map": per_map,
        "counterfactual_repair": {
            "matching": {
                "rule": (
                    "nearest real condition with exact path count, LoS, and K-validity; donor reuse allowed"
                ),
                "matched_samples": int(matched_candidate.numel()),
                "match_rate": float(matched_candidate.numel() / indices.numel()),
                "condition_distance_mean": float(match_distances.mean()),
                "condition_distance_p90": float(torch.quantile(match_distances, 0.9)),
            },
            "baseline_spatial_correlation": baseline_spatial,
            "repairs": repairs,
            "ranking_by_relative_frobenius_improvement": ranked,
            "interpretation": (
                "A positive repair improvement means that replacing that generated parameter group "
                "with condition/path-count-matched real values moved aggregate spatial correlation "
                "closer to real data. Joint repairs are interactions, not additive causal effects."
            ),
            "phase_caution": (
                "The path_phase repair tests path phase. Spatial-correlation phase also depends on "
                "AoD, power, array convention, and their joint structure, so the two errors are not equivalent."
            ),
        },
    }
    atomic_json_dump(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
