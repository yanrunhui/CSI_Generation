#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from diagnose_scene_joint_decomposition import (
    PATH_KEYS,
    _align_donor_paths,
    _canonicalize_path_order,
    _path_integrity,
    _projection_changes,
    _synthesize,
)
from evaluate_csi_absolute_quality import _make_map_pairs, _sample_tensors, _select
from physics_csi.config import load_yaml
from physics_csi.evaluation.scene_diagnostics import paired_map_summary, spatial_correlation_matrix
from physics_csi.physics.constraints import PhysicsConstraintLayer
from physics_csi.physics.synthesizer import MIMOOFDMSynthesizer, SystemConfig
from physics_csi.training.trainer import resolve_device
from physics_csi.utils.io import atomic_json_dump


MODEL_KEYS = {
    "real_vs_real": "reference",
    "b3_vs_real": "b3",
    "current_vs_real": "current",
}


def _parse_ints(value: str) -> tuple[int, ...]:
    result = tuple(sorted({int(item) for item in value.split(",") if item}))
    if not result or min(result) < 1:
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive integers")
    return result


def _validate_artifact(source: Mapping[str, Any]) -> None:
    if source.get("model_kind") != "b3_plus_dp_plus_aod_residual":
        raise ValueError("artifact is not a B3+DP+AoD residual evaluation")
    if source.get("aod_architecture") != "link_global":
        raise ValueError("artifact is not a Link-global AoD evaluation")
    for section in ("reference", "physics_only_baseline", "correct_map"):
        if section not in source or "csi" not in source[section]:
            raise ValueError(f"artifact is missing {section}.csi")


def _complex_distance(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    difference = candidate - reference
    valid = reference.abs() > 1e-3
    phase = (
        torch.angle(candidate[valid] * reference[valid].conj()).abs().mean()
        if bool(valid.any())
        else torch.zeros((), device=reference.device)
    )
    return {
        "relative_l2": float(
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(reference).clamp_min(1e-20)
        ),
        "complex_mae": float(difference.abs().mean()),
        "magnitude_mae": float((candidate.abs() - reference.abs()).abs().mean()),
        "phase_mae_rad": float(phase),
    }


def _frequency_correlation(csi: torch.Tensor, lags: tuple[int, ...]) -> torch.Tensor:
    # csi: [sample, rx, tx, subcarrier].  Aggregate over links and antennas.
    samples = csi.flatten(0, 2)
    energy = samples.abs().square().mean().clamp_min(1e-20)
    return torch.stack(
        [
            (samples[:, :-lag].conj() * samples[:, lag:]).mean() / energy
            for lag in lags
            if lag < csi.shape[-1]
        ]
    )


def _lag_errors(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for index in range(candidate.numel()):
        result[str(index)] = {
            "candidate_real": float(candidate[index].real),
            "candidate_imag": float(candidate[index].imag),
            "reference_real": float(reference[index].real),
            "reference_imag": float(reference[index].imag),
            "complex_absolute_error": float((candidate[index] - reference[index]).abs()),
            "magnitude_absolute_error": float((candidate[index].abs() - reference[index].abs()).abs()),
            "phase_absolute_error_rad": float(
                torch.angle(candidate[index] * reference[index].conj()).abs()
            ) if float(reference[index].abs()) > 1e-3 else 0.0,
        }
    return result


def _tx_spacing_groups(system: SystemConfig, device: torch.device) -> dict[int, dict[str, Any]]:
    rows, columns = system.tx_array_shape
    row = torch.arange(rows, device=device).repeat_interleave(columns)
    column = torch.arange(columns, device=device).repeat(rows)
    first, second = torch.triu_indices(rows * columns, rows * columns, offset=1, device=device)
    squared_grid_distance = (row[first] - row[second]).square() + (
        column[first] - column[second]
    ).square()
    groups: dict[int, dict[str, Any]] = {}
    for squared in torch.unique(squared_grid_distance, sorted=True).tolist():
        selected = squared_grid_distance == squared
        groups[int(squared)] = {
            "first": first[selected],
            "second": second[selected],
            "pairs": int(selected.sum()),
            "distance_wavelengths": math.sqrt(float(squared)) * system.tx_spacing_wavelengths,
        }
    return groups


def _tx_spacing_errors(
    candidate_csi: torch.Tensor,
    reference_csi: torch.Tensor,
    groups: Mapping[int, Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    candidate = spatial_correlation_matrix(candidate_csi)
    reference = spatial_correlation_matrix(reference_csi)
    output: dict[str, dict[str, float]] = {}
    for squared, group in groups.items():
        first = group["first"]
        second = group["second"]
        candidate_values = candidate[first, second]
        reference_values = reference[first, second]
        valid = reference_values.abs() > 1e-3
        phase = (
            torch.angle(candidate_values[valid] * reference_values[valid].conj()).abs().mean()
            if bool(valid.any())
            else torch.zeros((), device=reference.device)
        )
        output[str(squared)] = {
            "distance_wavelengths": float(group["distance_wavelengths"]),
            "antenna_pairs": int(group["pairs"]),
            "relative_l2": float(
                torch.linalg.vector_norm(candidate_values - reference_values)
                / torch.linalg.vector_norm(reference_values).clamp_min(1e-20)
            ),
            "complex_mae": float((candidate_values - reference_values).abs().mean()),
            "magnitude_mae": float((candidate_values.abs() - reference_values.abs()).abs().mean()),
            "phase_mae_rad": float(phase),
        }
    return output


def _gap_summary(
    model: list[float],
    floor: list[float],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    model_tensor = torch.tensor(model, dtype=torch.float64)
    floor_tensor = torch.tensor(floor, dtype=torch.float64)
    gap = model_tensor - floor_tensor
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(
        gap.numel(), (bootstrap_samples, gap.numel()), generator=generator
    )
    interval = torch.quantile(
        gap[indices].mean(-1), torch.tensor([0.025, 0.975], dtype=torch.float64)
    )
    floor_mean = float(floor_tensor.mean())
    return {
        "maps": int(gap.numel()),
        "model_mean_error": float(model_tensor.mean()),
        "real_real_mean_error": floor_mean,
        "model_to_real_real_ratio": (
            float(model_tensor.mean()) / floor_mean if floor_mean > 1e-20 else None
        ),
        "mean_excess_error": float(gap.mean()),
        "bootstrap_95_percent_ci": [float(interval[0]), float(interval[1])],
        "status": (
            "detectably_above_real_real_floor"
            if float(interval[0]) > 0
            else "inconclusive_vs_real_real_floor"
        ),
    }


def _decomposition(
    payloads: Mapping[str, Mapping[str, torch.Tensor]],
    pairs: Mapping[int, Mapping[str, torch.Tensor]],
    *,
    frequency_lags: tuple[int, ...],
    tx_groups: Mapping[int, Mapping[str, Any]],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    all_candidate = torch.cat([value["candidate_indices"] for value in pairs.values()])
    all_reference = torch.cat([value["reference_indices"] for value in pairs.values()])
    overall: dict[str, Any] = {}
    per_map: dict[str, Any] = {}
    map_vectors: dict[str, list[torch.Tensor]] = {label: [] for label in MODEL_KEYS}
    reference_vectors: list[torch.Tensor] = []
    for output_label, payload_label in MODEL_KEYS.items():
        candidate_csi = payloads[payload_label]["csi"][all_candidate]
        reference_csi = payloads["reference"]["csi"][all_reference]
        candidate_frequency = _frequency_correlation(candidate_csi, frequency_lags)
        reference_frequency = _frequency_correlation(reference_csi, frequency_lags)
        lag_values = _lag_errors(candidate_frequency, reference_frequency)
        for index, lag in enumerate(frequency_lags[: candidate_frequency.numel()]):
            lag_values[str(lag)] = lag_values.pop(str(index))
            lag_values[str(lag)]["lag"] = lag
        overall[output_label] = {
            "frequency_vector": _complex_distance(candidate_frequency, reference_frequency),
            "frequency_by_lag": lag_values,
            "tx_by_spacing": _tx_spacing_errors(candidate_csi, reference_csi, tx_groups),
        }

    map_records: list[dict[str, Any]] = []
    for map_id, pair in sorted(pairs.items()):
        record: dict[str, Any] = {"map_id": map_id, "models": {}}
        reference_csi = payloads["reference"]["csi"][pair["reference_indices"]]
        reference_frequency = _frequency_correlation(reference_csi, frequency_lags)
        reference_vectors.append(reference_frequency)
        for output_label, payload_label in MODEL_KEYS.items():
            candidate_csi = payloads[payload_label]["csi"][pair["candidate_indices"]]
            candidate_frequency = _frequency_correlation(candidate_csi, frequency_lags)
            map_vectors[output_label].append(candidate_frequency - reference_frequency)
            frequency_by_lag = {
                str(lag): float((candidate_frequency[index] - reference_frequency[index]).abs())
                for index, lag in enumerate(frequency_lags[: candidate_frequency.numel()])
            }
            tx_by_spacing = _tx_spacing_errors(candidate_csi, reference_csi, tx_groups)
            record["models"][output_label] = {
                "frequency_vector": _complex_distance(candidate_frequency, reference_frequency),
                "frequency_by_lag_complex_absolute_error": frequency_by_lag,
                "tx_by_spacing": tx_by_spacing,
            }
        map_records.append(record)

    frequency_summary: dict[str, Any] = {}
    for lag_number, lag in enumerate(frequency_lags):
        real = [item["models"]["real_vs_real"]["frequency_by_lag_complex_absolute_error"][str(lag)] for item in map_records]
        b3 = [item["models"]["b3_vs_real"]["frequency_by_lag_complex_absolute_error"][str(lag)] for item in map_records]
        current = [item["models"]["current_vs_real"]["frequency_by_lag_complex_absolute_error"][str(lag)] for item in map_records]
        frequency_summary[str(lag)] = {
            "b3_vs_real_real_floor": _gap_summary(b3, real, bootstrap_samples=bootstrap_samples, seed=seed + lag_number),
            "current_vs_real_real_floor": _gap_summary(current, real, bootstrap_samples=bootstrap_samples, seed=seed + 100 + lag_number),
            "current_vs_b3": paired_map_summary(current, b3, bootstrap_samples=bootstrap_samples, seed=seed + 200 + lag_number),
        }

    spacing_summary: dict[str, Any] = {}
    for spacing_number, (squared, group) in enumerate(tx_groups.items()):
        metric = "complex_mae"
        real = [item["models"]["real_vs_real"]["tx_by_spacing"][str(squared)][metric] for item in map_records]
        b3 = [item["models"]["b3_vs_real"]["tx_by_spacing"][str(squared)][metric] for item in map_records]
        current = [item["models"]["current_vs_real"]["tx_by_spacing"][str(squared)][metric] for item in map_records]
        spacing_summary[str(squared)] = {
            "distance_wavelengths": float(group["distance_wavelengths"]),
            "antenna_pairs": int(group["pairs"]),
            "b3_vs_real_real_floor": _gap_summary(b3, real, bootstrap_samples=bootstrap_samples, seed=seed + 1000 + spacing_number),
            "current_vs_real_real_floor": _gap_summary(current, real, bootstrap_samples=bootstrap_samples, seed=seed + 1100 + spacing_number),
            "current_vs_b3": paired_map_summary(current, b3, bootstrap_samples=bootstrap_samples, seed=seed + 1200 + spacing_number),
        }

    systematic: dict[str, Any] = {}
    for label, vectors in map_vectors.items():
        errors = torch.stack(vectors)
        coherent = errors.mean(0).abs()
        typical = errors.abs().mean(0).clamp_min(1e-20)
        systematic[label] = {
            str(lag): {
                "coherent_mean_error": float(coherent[index]),
                "mean_map_error": float(typical[index]),
                "systematic_bias_fraction": float((coherent / typical)[index]),
            }
            for index, lag in enumerate(frequency_lags[: errors.shape[1]])
        }

    for label in MODEL_KEYS:
        macro_frequency = sum(
            item["models"][label]["frequency_vector"]["relative_l2"]
            for item in map_records
        ) / len(map_records)
        per_map[label] = {
            "macro_frequency_relative_l2": macro_frequency,
            "pooled_frequency_relative_l2": overall[label]["frequency_vector"]["relative_l2"],
            "pooled_to_macro_ratio": overall[label]["frequency_vector"]["relative_l2"] / max(macro_frequency, 1e-20),
        }
    return {
        "overall": overall,
        "pooled_vs_per_map_frequency": per_map,
        "frequency_systematic_bias": systematic,
        "frequency_per_lag_map_summary": frequency_summary,
        "tx_per_spacing_map_summary": spacing_summary,
        "per_map": map_records,
    }


def _make_counterfactual_pairs(
    reference: Mapping[str, torch.Tensor],
    b3: Mapping[str, torch.Tensor],
    current: Mapping[str, torch.Tensor],
    *,
    seed: int,
    maximum_distance: float,
    maximum_samples: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    conditions = reference["conditions"].float()
    validity = reference["condition_validity"].bool()
    map_ids = reference["map_id"].long()
    path_count = reference["path_mask"].sum(-1).long()
    b3_count = b3["path_mask"].sum(-1).long()
    current_count = current["path_mask"].sum(-1).long()
    common_count = (path_count == b3_count) & (path_count == current_count)
    center = conditions[:, :3].mean(0)
    scale = conditions[:, :3].std(0, unbiased=False).clamp_min(1e-6)
    features = (conditions[:, :3] - center) / scale
    features[:, 2] = torch.where(validity[:, 2], features[:, 2], torch.zeros_like(features[:, 2]))
    los = conditions[:, 3] >= 0.5
    generator = torch.Generator().manual_seed(seed)
    candidate_values: list[int] = []
    donor_values: list[int] = []
    distance_values: list[float] = []
    strata = sorted(
        {
            (int(map_ids[index]), int(path_count[index]), bool(los[index]), bool(validity[index, 2]))
            for index in torch.nonzero(common_count, as_tuple=False).flatten().tolist()
        }
    )
    for map_id, count, is_los, k_valid in strata:
        indices = torch.nonzero(
            common_count
            & (map_ids == map_id)
            & (path_count == count)
            & (los == is_los)
            & (validity[:, 2] == k_valid),
            as_tuple=False,
        ).flatten()
        if indices.numel() < 2:
            continue
        permutation = indices[torch.randperm(indices.numel(), generator=generator).to(indices.device)]
        midpoint = permutation.numel() // 2
        first = permutation[:midpoint]
        second = permutation[midpoint : midpoint * 2]
        matrix = torch.cdist(features[first], features[second]).cpu()
        used = torch.zeros(second.numel(), dtype=torch.bool)
        for row in torch.argsort(matrix.amin(1)).tolist():
            for column in torch.argsort(matrix[row]).tolist():
                distance = float(matrix[row, column])
                if distance > maximum_distance:
                    break
                if not bool(used[column]):
                    used[column] = True
                    candidate_values.append(int(first[row]))
                    donor_values.append(int(second[column]))
                    distance_values.append(distance)
                    break
    if not candidate_values:
        raise RuntimeError("no exact-path-count counterfactual pairs were found")
    candidate = torch.tensor(candidate_values, dtype=torch.long, device=conditions.device)
    donor = torch.tensor(donor_values, dtype=torch.long, device=conditions.device)
    distance = torch.tensor(distance_values, dtype=torch.float32, device=conditions.device)
    available = int(candidate.numel())
    if candidate.numel() > maximum_samples:
        selected = torch.randperm(candidate.numel(), generator=generator)[:maximum_samples].to(candidate.device)
        candidate = candidate[selected]
        donor = donor[selected]
        distance = distance[selected]
    return candidate, donor, distance, {
        "available_pairs_before_subsampling": available,
        "selected_pairs": int(candidate.numel()),
        "selected_maps": int(torch.unique(map_ids[candidate]).numel()),
        "same_Map": True,
        "exact_path_count": True,
        "exact_LoS": True,
        "exact_K_validity": True,
        "unique_donor_within_each_stratum": True,
        "maximum_condition_distance": maximum_distance,
        "condition_distance_mean": float(distance.mean()),
        "condition_distance_p90": float(torch.quantile(distance, 0.90)),
        "condition_distance_maximum": float(distance.max()),
        "samples_excluded_because_model_path_counts_differed_from_real": int((~common_count).sum()),
    }


def _random_active_permutation(mask: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=mask.device).manual_seed(seed)
    scores = torch.rand(mask.shape, generator=generator, device=mask.device)
    scores = torch.where(mask, scores, torch.full_like(scores, 2.0))
    return torch.argsort(scores, dim=-1)


def _permute_path_value(value: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    gather = order
    while gather.ndim < value.ndim:
        gather = gather.unsqueeze(-1)
    return value.gather(1, gather.expand_as(value))


def _circular_centroid_deg(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    radians = torch.deg2rad(values)
    sine = (weights * radians.sin()).sum(-1)
    cosine = (weights * radians.cos()).sum(-1)
    return torch.rad2deg(torch.atan2(sine, cosine))


def _wrap_degrees(values: torch.Tensor) -> torch.Tensor:
    return torch.remainder(values + 180.0, 360.0) - 180.0


def _angle_centers(payload: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    mask = payload["path_mask"].bool()
    power = payload["path_powers_linear"].float() * mask
    weights = power / power.sum(-1, keepdim=True).clamp_min(1e-20)
    azimuth = _circular_centroid_deg(payload["aod_az_deg"].float(), weights)
    elevation = (weights * payload["aod_el_deg"].float()).sum(-1)
    return azimuth, elevation


def _aod_global_orientation_repair(
    generated: Mapping[str, torch.Tensor], donor: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    generated_azimuth, generated_elevation = _angle_centers(generated)
    donor_azimuth, donor_elevation = _angle_centers(donor)
    result = dict(generated)
    result["aod_az_deg"] = _wrap_degrees(
        generated["aod_az_deg"]
        + _wrap_degrees(donor_azimuth - generated_azimuth).unsqueeze(-1)
    )
    result["aod_el_deg"] = (
        generated["aod_el_deg"]
        + (donor_elevation - generated_elevation).unsqueeze(-1)
    ).clamp(-89.9, 89.9)
    return result


def _aod_relative_structure_repair(
    generated: Mapping[str, torch.Tensor], donor: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    generated_azimuth, generated_elevation = _angle_centers(generated)
    donor_azimuth, donor_elevation = _angle_centers(donor)
    result = dict(generated)
    provisional_azimuth = _wrap_degrees(
        generated_azimuth.unsqueeze(-1)
        + _wrap_degrees(donor["aod_az_deg"] - donor_azimuth.unsqueeze(-1))
    )
    provisional_elevation = (
        generated_elevation.unsqueeze(-1)
        + donor["aod_el_deg"]
        - donor_elevation.unsqueeze(-1)
    ).clamp(-89.9, 89.9)
    provisional = dict(generated)
    provisional["aod_az_deg"] = provisional_azimuth
    provisional["aod_el_deg"] = provisional_elevation
    actual_azimuth, actual_elevation = _angle_centers(provisional)
    result["aod_az_deg"] = _wrap_degrees(
        provisional_azimuth
        + _wrap_degrees(generated_azimuth - actual_azimuth).unsqueeze(-1)
    )
    result["aod_el_deg"] = (
        provisional_elevation
        + (generated_elevation - actual_elevation).unsqueeze(-1)
    ).clamp(-89.9, 89.9)
    return result


def _relative_angle_audit(
    before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]
) -> dict[str, float]:
    mask = before["path_mask"].bool()
    azimuth_errors: list[torch.Tensor] = []
    elevation_errors: list[torch.Tensor] = []
    for row in range(mask.shape[0]):
        count = int(mask[row].sum())
        if count < 2:
            continue
        first, second = torch.triu_indices(count, count, offset=1, device=mask.device)
        before_az = _wrap_degrees(before["aod_az_deg"][row, first] - before["aod_az_deg"][row, second]).abs()
        after_az = _wrap_degrees(after["aod_az_deg"][row, first] - after["aod_az_deg"][row, second]).abs()
        before_el = before["aod_el_deg"][row, first] - before["aod_el_deg"][row, second]
        after_el = after["aod_el_deg"][row, first] - after["aod_el_deg"][row, second]
        azimuth_errors.append((after_az - before_az).abs())
        elevation_errors.append((after_el - before_el).abs())
    return {
        "pairwise_azimuth_gap_mae_deg": float(torch.cat(azimuth_errors).mean()),
        "pairwise_elevation_gap_mae_deg": float(torch.cat(elevation_errors).mean()),
    }


def _project_delay_phase(
    payload: Mapping[str, torch.Tensor],
    constraint: PhysicsConstraintLayer,
) -> tuple[dict[str, Any], Any, dict[str, torch.Tensor]]:
    ordered = _canonicalize_path_order(payload)
    before = {key: ordered[key].clone() for key in PATH_KEYS}
    result = constraint(
        ordered["path_delays_ns"],
        ordered["path_powers_linear"].clamp_min(1e-20),
        ordered["path_mask"],
        ordered["target_conditions"],
        target_condition_validity=ordered["target_condition_validity"],
        direct_path_mask=ordered["direct_path_mask"],
    )
    ordered["path_delays_ns"] = result.delays_ns
    ordered["path_powers_linear"] = result.powers_linear
    ordered["direct_path_mask"] = result.direct_path_mask
    ordered["actual_conditions"] = result.actual_conditions
    ordered["actual_condition_validity"] = result.condition_validity
    return ordered, result, before


def _correlation_metrics(
    candidate_csi: torch.Tensor,
    reference_csi: torch.Tensor,
    *,
    frequency_lags: tuple[int, ...],
    tx_groups: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    candidate_frequency = _frequency_correlation(candidate_csi, frequency_lags)
    reference_frequency = _frequency_correlation(reference_csi, frequency_lags)
    lag = _lag_errors(candidate_frequency, reference_frequency)
    renamed: dict[str, Any] = {}
    for index, value in lag.items():
        actual = frequency_lags[int(index)]
        renamed[str(actual)] = value | {"lag": actual}
    candidate_tx = spatial_correlation_matrix(candidate_csi)
    reference_tx = spatial_correlation_matrix(reference_csi)
    return {
        "frequency_vector": _complex_distance(candidate_frequency, reference_frequency),
        "frequency_by_lag": renamed,
        "tx_matrix": _complex_distance(candidate_tx.flatten(), reference_tx.flatten()),
        "tx_by_spacing": _tx_spacing_errors(candidate_csi, reference_csi, tx_groups),
    }


def _counterfactual_suite(
    generated: Mapping[str, torch.Tensor],
    donor: Mapping[str, torch.Tensor],
    *,
    constraint: PhysicsConstraintLayer,
    synthesizer: MIMOOFDMSynthesizer,
    batch_size: int,
    frequency_lags: tuple[int, ...],
    tx_groups: Mapping[int, Mapping[str, Any]],
    seed: int,
) -> dict[str, Any]:
    generated = _canonicalize_path_order(generated)
    donor = _canonicalize_path_order(donor)
    aligned, alignment_cost = _align_donor_paths(generated, donor)
    variants: dict[str, dict[str, Any]] = {"control": dict(generated)}

    phase = dict(generated)
    phase["path_phases_rad"] = aligned["path_phases_rad"].clone()
    variants["phase_only"] = phase

    phase_permuted = dict(generated)
    order = _random_active_permutation(generated["path_mask"].bool(), seed)
    phase_permuted["path_phases_rad"] = _permute_path_value(generated["path_phases_rad"], order)
    variants["phase_within_link_permutation_control"] = phase_permuted

    delay_phase_raw = dict(generated)
    delay_phase_raw["path_delays_ns"] = aligned["path_delays_ns"].clone()
    delay_phase_raw["path_phases_rad"] = aligned["path_phases_rad"].clone()
    delay_phase, projection_result, projection_before = _project_delay_phase(delay_phase_raw, constraint)
    variants["delay_phase_joint"] = delay_phase

    variants["aod_global_orientation_only"] = _aod_global_orientation_repair(generated, aligned)
    variants["aod_relative_structure_only"] = _aod_relative_structure_repair(generated, aligned)
    full_aod = dict(generated)
    full_aod["aod_az_deg"] = aligned["aod_az_deg"].clone()
    full_aod["aod_el_deg"] = aligned["aod_el_deg"].clone()
    variants["aod_full_donor"] = full_aod

    results: dict[str, Any] = {}
    synthesized: dict[str, torch.Tensor] = {}
    for name, payload in variants.items():
        csi = _synthesize(payload, synthesizer, batch_size)
        synthesized[name] = csi
        results[name] = {
            "metrics": _correlation_metrics(
                csi,
                donor["csi"],
                frequency_lags=frequency_lags,
                tx_groups=tx_groups,
            ),
            "path_integrity": _path_integrity(payload),
        }
    control_frequency = results["control"]["metrics"]["frequency_vector"]["relative_l2"]
    control_tx = results["control"]["metrics"]["tx_matrix"]["relative_l2"]
    for name, values in results.items():
        values["gain_control_minus_variant"] = {
            "frequency_relative_l2": control_frequency - values["metrics"]["frequency_vector"]["relative_l2"],
            "tx_relative_l2": control_tx - values["metrics"]["tx_matrix"]["relative_l2"],
        }
    results["delay_phase_joint"]["projection_audit"] = {
        "feasible_rate": float(projection_result.feasible.float().mean()),
        "changes": _projection_changes(
            projection_before,
            delay_phase,
            replaced_parameters=("path_delays_ns", "path_phases_rad"),
        ),
        "before_projection_path_integrity": _path_integrity(projection_before),
    }
    for name in ("aod_global_orientation_only", "aod_relative_structure_only", "aod_full_donor"):
        results[name]["relative_angle_change_vs_control"] = _relative_angle_audit(generated, variants[name])
    artifact_control_error = (
        (generated["csi"] - synthesized["control"]).abs().square().sum()
        / generated["csi"].abs().square().sum().clamp_min(1e-20)
    )
    return {
        "path_alignment_cost_mean": float(alignment_cost.mean()),
        "path_alignment_cost_p90": float(torch.quantile(alignment_cost, 0.90)),
        "artifact_vs_resynthesized_control_relative_mse": float(artifact_control_error),
        "variants": results,
    }


def _counterfactual(
    payloads: Mapping[str, Mapping[str, torch.Tensor]],
    candidate_indices: torch.Tensor,
    donor_indices: torch.Tensor,
    *,
    constraint: PhysicsConstraintLayer,
    synthesizer: MIMOOFDMSynthesizer,
    batch_size: int,
    frequency_lags: tuple[int, ...],
    tx_groups: Mapping[int, Mapping[str, Any]],
    seed: int,
) -> dict[str, Any]:
    donor = _select(payloads["reference"], donor_indices)
    real_candidate = _select(payloads["reference"], candidate_indices)
    real_floor = _correlation_metrics(
        real_candidate["csi"], donor["csi"], frequency_lags=frequency_lags, tx_groups=tx_groups
    )
    models: dict[str, Any] = {}
    for model_number, label in enumerate(("b3", "current")):
        models[label] = _counterfactual_suite(
            _select(payloads[label], candidate_indices),
            donor,
            constraint=constraint,
            synthesizer=synthesizer,
            batch_size=batch_size,
            frequency_lags=frequency_lags,
            tx_groups=tx_groups,
            seed=seed + model_number * 1000,
        )
    return {"real_vs_real_floor": real_floor, "models": models}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Targeted frequency-lag, Tx-spacing, Phase and relative-AoD diagnostics."
    )
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--frequency-lags", type=_parse_ints, default=(1, 2, 4, 8, 16, 32))
    parser.add_argument("--minimum-pairs-per-map", type=int, default=12)
    parser.add_argument("--maximum-condition-distance", type=float, default=2.5)
    parser.add_argument("--maximum-counterfactual-distance", type=float, default=1.0)
    parser.add_argument("--maximum-counterfactual-samples", type=int, default=4096)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    device = resolve_device(args.device)
    source = torch.load(args.artifact, map_location="cpu", weights_only=False)
    _validate_artifact(source)
    payloads = {
        "reference": _sample_tensors(source["reference"], device),
        "b3": _sample_tensors(source["physics_only_baseline"], device),
        "current": _sample_tensors(source["correct_map"], device),
    }
    if payloads["reference"]["csi"].shape[1] != 1:
        raise ValueError("this targeted diagnostic assumes Rx=1; AoA omission is otherwise invalid")
    data = load_yaml(args.data_config)["data"]
    system = SystemConfig.from_mapping(data["system"])
    synthesizer = MIMOOFDMSynthesizer(system).to(device).eval()
    constraint = PhysicsConstraintLayer(
        enforce_total_power=True,
        enforce_delay_spread=True,
        enforce_k_factor=True,
        enforce_los_direct_path=True,
        epsilon=1e-20,
        invalid_k_fill_db=float(data.get("k_factor", {}).get("invalid_fill_db", 0.0)),
    ).to(device).eval()
    tx_groups = _tx_spacing_groups(system, device)

    pairs, pairing_audit = _make_map_pairs(
        payloads["reference"],
        seed=args.seed,
        maximum_distance=args.maximum_condition_distance,
        minimum_pairs=args.minimum_pairs_per_map,
    )
    decomposition = _decomposition(
        payloads,
        pairs,
        frequency_lags=args.frequency_lags,
        tx_groups=tx_groups,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    candidate_indices, donor_indices, distances, counterfactual_audit = _make_counterfactual_pairs(
        payloads["reference"],
        payloads["b3"],
        payloads["current"],
        seed=args.seed + 1,
        maximum_distance=args.maximum_counterfactual_distance,
        maximum_samples=args.maximum_counterfactual_samples,
    )
    with torch.inference_mode():
        counterfactual = _counterfactual(
            payloads,
            candidate_indices,
            donor_indices,
            constraint=constraint,
            synthesizer=synthesizer,
            batch_size=args.batch_size,
            frequency_lags=args.frequency_lags,
            tx_groups=tx_groups,
            seed=args.seed + 2,
        )

    report = {
        "format_version": 1,
        "diagnostic": "csi_remaining_frequency_and_spatial_bottlenecks",
        "source_artifact": str(Path(args.artifact).resolve()),
        "samples": int(payloads["reference"]["csi"].shape[0]),
        "maps": len(pairs),
        "configuration": {
            "frequency_lags": list(args.frequency_lags),
            "tx_array_shape": list(system.tx_array_shape),
            "tx_spacing_wavelengths": system.tx_spacing_wavelengths,
            "rx_array_shape": list(system.rx_array_shape),
            "seed": args.seed,
            "bootstrap_samples": args.bootstrap_samples,
        },
        "pairing_audit": pairing_audit,
        "frequency_and_tx_decomposition": decomposition,
        "counterfactual_selection_audit": counterfactual_audit,
        "counterfactual": counterfactual,
        "interpretation": {
            "frequency": "Compare per-lag model error with the Real-vs-Real floor. A high systematic-bias fraction means Map-level errors align instead of cancelling when pooled.",
            "tx": "Tx correlations are grouped by physical planar-array separation, not flattened antenna index.",
            "phase": "Phase-only transfers aligned donor phase; Delay-Phase transfers both and then applies the physical projection before CSI synthesis.",
            "aod": "Global-orientation repair preserves generated relative azimuth/elevation gaps; relative-structure repair preserves the generated power-weighted global direction while importing donor path offsets.",
        },
        "cautions": [
            "Counterfactual donors are independent same-Map links with exact path count, LoS and K-validity, but they remain matched rather than paired ground truth.",
            "Projection coupling is reported for Delay-Phase and counterfactual gains are not independent causal effects.",
            "The same repairs are applied to B3 and the current model to expose shared backbone limitations.",
            "AoA counterfactual is intentionally omitted because Rx=1 makes the synthesized CSI invariant to AoA under the current normalized array model.",
        ],
    }
    atomic_json_dump(report, args.output)
    print(
        json.dumps(
            {
                "event": "csi_remaining_bottlenecks_complete",
                "output": str(Path(args.output).resolve()),
                "maps": len(pairs),
                "counterfactual_samples": int(candidate_indices.numel()),
                "counterfactual_condition_distance_mean": float(distances.mean()),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
