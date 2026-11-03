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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_csi.evaluation.distribution import rbf_mmd, sliced_wasserstein
from physics_csi.evaluation.scene_diagnostics import paired_map_summary
from physics_csi.training.trainer import resolve_device
from physics_csi.utils.io import atomic_json_dump


MODEL_LABELS = ("b3", "b3_scene_id")
PHASE_PROBE_COUNT = 16
PHASE_PROBE_SEED = 314159


def _subset_tensor(value: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return value[indices.to(value.device)]


def _energy_normalized_features(csi: torch.Tensor) -> torch.Tensor:
    flat = csi.flatten(1)
    flat = flat / flat.abs().square().mean(-1, keepdim=True).sqrt().clamp_min(1e-20)
    return torch.cat([flat.real, flat.imag], dim=-1).float()


def _phase_normalize(csi: torch.Tensor) -> torch.Tensor:
    """Remove one link-wise U(1) phase using stable equivariant probe anchors."""
    flat = csi.flatten(1)
    generator = torch.Generator(device=csi.device).manual_seed(PHASE_PROBE_SEED)
    probe_real = torch.randn(
        PHASE_PROBE_COUNT,
        flat.shape[1],
        generator=generator,
        device=csi.device,
        dtype=csi.real.dtype,
    )
    probe_imag = torch.randn(
        PHASE_PROBE_COUNT,
        flat.shape[1],
        generator=generator,
        device=csi.device,
        dtype=csi.real.dtype,
    )
    probes = torch.complex(probe_real, probe_imag)
    probes = probes / probes.abs().square().sum(-1, keepdim=True).sqrt().clamp_min(
        1e-20
    )
    candidate_anchors = flat @ probes.conj().transpose(0, 1)
    anchor_index = candidate_anchors.abs().argmax(-1, keepdim=True)
    anchor = candidate_anchors.gather(1, anchor_index)
    unit = anchor / anchor.abs().clamp_min(1e-20)
    broadcast_shape = (csi.shape[0],) + (1,) * (csi.ndim - 1)
    return csi * unit.conj().reshape(broadcast_shape)


def _random_global_phase(
    csi: torch.Tensor, *, generator: torch.Generator
) -> torch.Tensor:
    phase = (
        torch.rand(
            csi.shape[0], generator=generator, device=csi.device, dtype=csi.real.dtype
        )
        * (2.0 * math.pi)
        - math.pi
    )
    broadcast_shape = (csi.shape[0],) + (1,) * (csi.ndim - 1)
    return csi * torch.exp(1j * phase).reshape(broadcast_shape)


def _relative_error(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    numerator = (candidate - reference).abs().square().flatten(1).sum(-1)
    denominator = reference.abs().square().flatten(1).sum(-1).clamp_min(1e-20)
    return float((numerator / denominator).mean())


class ProjectionBasis:
    def __init__(
        self,
        reference_csi: torch.Tensor,
        dimensions: tuple[int, ...],
        seeds: tuple[int, ...],
    ) -> None:
        self.dimensions = dimensions
        self.seeds = seeds
        reference = _energy_normalized_features(reference_csi)
        self.mean = reference.mean(0, keepdim=True)
        raw_scale = reference.std(0, unbiased=False, keepdim=True)
        positive = raw_scale[torch.isfinite(raw_scale) & (raw_scale > 0)]
        typical = (
            positive.median()
            if positive.numel()
            else torch.ones((), device=reference.device)
        )
        self.scale_floor = torch.maximum(
            typical * 0.05, torch.tensor(1e-6, device=reference.device)
        )
        self.scale = torch.where(
            torch.isfinite(raw_scale), raw_scale, torch.zeros_like(raw_scale)
        ).clamp_min(self.scale_floor)
        self.maximum_dimension = max(dimensions)
        self.projections: dict[int, torch.Tensor] = {}
        for seed in seeds:
            generator = torch.Generator(device=reference.device).manual_seed(seed)
            projection = torch.randint(
                0,
                2,
                (reference.shape[1], self.maximum_dimension),
                generator=generator,
                device=reference.device,
                dtype=torch.int8,
            ).float()
            self.projections[seed] = (projection * 2.0 - 1.0) / math.sqrt(
                reference.shape[1]
            )

    def project(self, csi: torch.Tensor) -> dict[int, torch.Tensor]:
        standardized = (_energy_normalized_features(csi) - self.mean) / self.scale
        return {
            seed: standardized @ projection
            for seed, projection in self.projections.items()
        }

    def audit(self) -> dict[str, Any]:
        return {
            "dimensions": list(self.dimensions),
            "seeds": list(self.seeds),
            "normalization_scale_floor": float(self.scale_floor),
            "maximum_projection_dimension": self.maximum_dimension,
        }


def _metric(
    candidate: torch.Tensor, reference: torch.Tensor
) -> dict[str, float]:
    return {
        "mmd_rbf": float(rbf_mmd(candidate, reference, [2.0, 4.0, 8.0, 16.0])),
        "sliced_wasserstein": float(sliced_wasserstein(candidate, reference)),
    }


def _projection_metrics(
    candidate: Mapping[int, torch.Tensor],
    reference: Mapping[int, torch.Tensor],
    dimensions: tuple[int, ...],
    indices: torch.Tensor | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for dimension in dimensions:
        runs = []
        for seed in candidate:
            candidate_values = candidate[seed][:, :dimension]
            reference_values = reference[seed][:, :dimension]
            if indices is not None:
                candidate_values = _subset_tensor(candidate_values, indices)
                reference_values = _subset_tensor(reference_values, indices)
            runs.append({"seed": seed, **_metric(candidate_values, reference_values)})
        result[str(dimension)] = {
            "runs": runs,
            "mmd_mean": sum(item["mmd_rbf"] for item in runs) / len(runs),
            "mmd_min": min(item["mmd_rbf"] for item in runs),
            "mmd_max": max(item["mmd_rbf"] for item in runs),
            "swd_mean": sum(item["sliced_wasserstein"] for item in runs) / len(runs),
            "swd_min": min(item["sliced_wasserstein"] for item in runs),
            "swd_max": max(item["sliced_wasserstein"] for item in runs),
        }
    return result


def _primary_metric(
    candidate: Mapping[int, torch.Tensor],
    reference: Mapping[int, torch.Tensor],
    *,
    dimension: int,
    indices: torch.Tensor,
) -> dict[str, float]:
    values = []
    for seed in candidate:
        values.append(
            _metric(
                _subset_tensor(candidate[seed][:, :dimension], indices),
                _subset_tensor(reference[seed][:, :dimension], indices),
            )
        )
    return {
        "mmd_rbf": sum(item["mmd_rbf"] for item in values) / len(values),
        "sliced_wasserstein": sum(
            item["sliced_wasserstein"] for item in values
        )
        / len(values),
    }


def _condition_features(
    conditions: torch.Tensor, validity: torch.Tensor
) -> torch.Tensor:
    values = conditions[:, :3].float().clone()
    mean = values.mean(0)
    scale = values.std(0, unbiased=False).clamp_min(1e-6)
    if bool(validity[:, 2].any()):
        mean[2] = values[validity[:, 2], 2].mean()
        scale[2] = values[validity[:, 2], 2].std(unbiased=False).clamp_min(1e-6)
    values = (values - mean) / scale
    values[:, 2] = torch.where(validity[:, 2], values[:, 2], torch.zeros_like(values[:, 2]))
    return values


def _balanced_condition_groups(
    map_id: torch.Tensor,
    conditions: torch.Tensor,
    validity: torch.Tensor,
    *,
    minimum_samples: int,
    maximum_groups_per_stratum: int,
) -> list[dict[str, Any]]:
    features = _condition_features(conditions, validity)
    los = conditions[:, 3] >= 0.5
    k_valid = validity[:, 2].bool()
    groups: list[dict[str, Any]] = []
    group_number = 0
    strata = sorted(
        {
            (int(map_id[index]), bool(los[index]), bool(k_valid[index]))
            for index in range(map_id.numel())
        }
    )
    for current_map, current_los, current_k_valid in strata:
        indices = torch.nonzero(
            (map_id == current_map)
            & (los == current_los)
            & (k_valid == current_k_valid),
            as_tuple=False,
        ).flatten()
        if indices.numel() < minimum_samples:
            continue
        group_count = min(
            maximum_groups_per_stratum,
            max(1, int(indices.numel()) // minimum_samples),
        )
        centered = features[indices] - features[indices].mean(0, keepdim=True)
        if group_count == 1 or indices.numel() < 2:
            ordered = indices
        else:
            _, _, right = torch.linalg.svd(centered, full_matrices=False)
            score = centered @ right[0]
            ordered = indices[torch.argsort(score)]
        for chunk in torch.tensor_split(ordered, group_count):
            if chunk.numel() < minimum_samples:
                continue
            local = features[chunk]
            center = local.mean(0, keepdim=True)
            radius = (local - center).square().sum(-1).sqrt()
            groups.append(
                {
                    "name": f"group_{group_number}",
                    "map_id": current_map,
                    "los": current_los,
                    "k_valid": current_k_valid,
                    "indices": chunk,
                    "samples": int(chunk.numel()),
                    "condition_radius_mean": float(radius.mean()),
                    "condition_radius_maximum": float(radius.max()),
                }
            )
            group_number += 1
    return groups


def _paired_summary(
    values: Mapping[str, Mapping[str, Mapping[str, float]]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    names = sorted(values)
    if not names:
        return {"status": "no_groups_meeting_minimum_sample_requirement"}
    for metric_index, metric in enumerate(("mmd_rbf", "sliced_wasserstein")):
        scene = [values[name]["b3_scene_id"][metric] for name in names]
        baseline = [values[name]["b3"][metric] for name in names]
        result[metric] = paired_map_summary(
            scene,
            baseline,
            bootstrap_samples=bootstrap_samples,
            seed=seed + metric_index,
        )
    return result


def _evaluate_mode(
    reference_csi: torch.Tensor,
    model_csi: Mapping[str, torch.Tensor],
    map_id: torch.Tensor,
    condition_groups: list[dict[str, Any]],
    *,
    dimensions: tuple[int, ...],
    projection_seeds: tuple[int, ...],
    minimum_per_map: int,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    basis = ProjectionBasis(reference_csi, dimensions, projection_seeds)
    reference_projected = basis.project(reference_csi)
    model_projected = {
        model: basis.project(csi) for model, csi in model_csi.items()
    }
    overall = {
        model: _projection_metrics(
            projected, reference_projected, dimensions
        )
        for model, projected in model_projected.items()
    }
    primary_dimension = max(dimensions)
    per_map: dict[str, Any] = {}
    for current_map in torch.unique(map_id, sorted=True):
        indices = torch.nonzero(map_id == current_map, as_tuple=False).flatten()
        if indices.numel() < minimum_per_map:
            continue
        name = f"map_{int(current_map)}"
        per_map[name] = {
            "samples": int(indices.numel()),
            **{
                model: _primary_metric(
                    projected,
                    reference_projected,
                    dimension=primary_dimension,
                    indices=indices,
                )
                for model, projected in model_projected.items()
            },
        }
    map_values = {
        name: {model: values[model] for model in MODEL_LABELS}
        for name, values in per_map.items()
    }
    condition_values: dict[str, Any] = {}
    for group in condition_groups:
        condition_values[group["name"]] = {
            key: value for key, value in group.items() if key != "indices"
        } | {
            model: _primary_metric(
                projected,
                reference_projected,
                dimension=primary_dimension,
                indices=group["indices"],
            )
            for model, projected in model_projected.items()
        }
    condition_for_pairing = {
        name: {model: values[model] for model in MODEL_LABELS}
        for name, values in condition_values.items()
    }
    return {
        "basis": basis.audit(),
        "primary_dimension_for_grouped_metrics": primary_dimension,
        "overall": overall,
        "per_map": per_map,
        "per_map_paired_summary": _paired_summary(
            map_values, bootstrap_samples=bootstrap_samples, seed=seed
        ),
        "map_condition_groups": condition_values,
        "map_condition_paired_summary": _paired_summary(
            condition_for_pairing,
            bootstrap_samples=bootstrap_samples,
            seed=seed + 100,
        ),
    }


def _aggregate_random_modes(
    evaluations: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    dimensions = evaluations[0]["overall"]["b3"].keys()
    overall: dict[str, Any] = {}
    for model in MODEL_LABELS:
        overall[model] = {}
        for dimension in dimensions:
            runs = [
                {"repeat": repeat, **run}
                for repeat, evaluation in enumerate(evaluations)
                for run in evaluation["overall"][model][dimension]["runs"]
            ]
            overall[model][dimension] = {
                "repeat_seed_runs": runs,
                "mmd_mean": sum(item["mmd_rbf"] for item in runs) / len(runs),
                "mmd_min": min(item["mmd_rbf"] for item in runs),
                "mmd_max": max(item["mmd_rbf"] for item in runs),
                "swd_mean": sum(item["sliced_wasserstein"] for item in runs) / len(runs),
                "swd_min": min(item["sliced_wasserstein"] for item in runs),
                "swd_max": max(item["sliced_wasserstein"] for item in runs),
            }

    def aggregate_groups(section: str) -> tuple[dict[str, Any], dict[str, Any]]:
        names = sorted(
            set.intersection(
                *(set(evaluation[section]) for evaluation in evaluations)
            )
        )
        groups: dict[str, Any] = {}
        paired_input: dict[str, Any] = {}
        for name in names:
            template = evaluations[0][section][name]
            groups[name] = {
                key: value
                for key, value in template.items()
                if key not in MODEL_LABELS
            }
            paired_input[name] = {}
            for model in MODEL_LABELS:
                averaged = {
                    metric: sum(
                        evaluation[section][name][model][metric]
                        for evaluation in evaluations
                    )
                    / len(evaluations)
                    for metric in ("mmd_rbf", "sliced_wasserstein")
                }
                groups[name][model] = averaged
                paired_input[name][model] = averaged
        return groups, _paired_summary(
            paired_input,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )

    per_map, per_map_summary = aggregate_groups("per_map")
    condition, condition_summary = aggregate_groups("map_condition_groups")
    return {
        "repeats": len(evaluations),
        "overall": overall,
        "per_map": per_map,
        "per_map_paired_summary": per_map_summary,
        "map_condition_groups": condition,
        "map_condition_paired_summary": condition_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare raw, randomized-global-phase and phase-invariant Full CSI distances."
    )
    parser.add_argument("--generated", required=True)
    parser.add_argument("--maximum-samples", type=int, default=2048)
    parser.add_argument("--projection-dimensions", default="32,64,128")
    parser.add_argument("--projection-seeds", default="0,1,2")
    parser.add_argument("--random-phase-repeats", type=int, default=5)
    parser.add_argument("--minimum-per-map", type=int, default=8)
    parser.add_argument("--minimum-condition-group", type=int, default=6)
    parser.add_argument("--maximum-condition-groups-per-stratum", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    dimensions = tuple(
        int(value) for value in args.projection_dimensions.split(",") if value.strip()
    )
    projection_seeds = tuple(
        int(value) for value in args.projection_seeds.split(",") if value.strip()
    )
    if not dimensions or min(dimensions) < 1:
        parser.error("projection dimensions must contain positive integers")
    if not projection_seeds:
        parser.error("at least one projection seed is required")
    if min(
        args.maximum_samples,
        args.random_phase_repeats,
        args.minimum_per_map,
        args.minimum_condition_group,
        args.maximum_condition_groups_per_stratum,
        args.bootstrap_samples,
    ) < 1:
        parser.error("sample, repeat, group and bootstrap counts must be positive")

    device = resolve_device(args.device)
    source = torch.load(args.generated, map_location="cpu", weights_only=False)
    for key in ("reference", "correct_map", "physics_only_baseline"):
        if source.get(key) is None:
            raise ValueError(f"generated artifact must contain {key}")
    sample_count = source["reference"]["csi"].shape[0]
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.arange(sample_count)
    if sample_count > args.maximum_samples:
        indices = torch.randperm(sample_count, generator=generator)[
            : args.maximum_samples
        ].sort().values
    reference_payload = source["reference"]
    reference_csi = reference_payload["csi"][indices].to(device)
    model_csi = {
        "b3": source["physics_only_baseline"]["csi"][indices].to(device),
        "b3_scene_id": source["correct_map"]["csi"][indices].to(device),
    }
    map_id = reference_payload["map_id"][indices].long().to(device)
    conditions = reference_payload["conditions"][indices].float().to(device)
    validity = reference_payload["condition_validity"][indices].bool().to(device)
    condition_groups = _balanced_condition_groups(
        map_id,
        conditions,
        validity,
        minimum_samples=args.minimum_condition_group,
        maximum_groups_per_stratum=args.maximum_condition_groups_per_stratum,
    )

    raw = _evaluate_mode(
        reference_csi,
        model_csi,
        map_id,
        condition_groups,
        dimensions=dimensions,
        projection_seeds=projection_seeds,
        minimum_per_map=args.minimum_per_map,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )

    randomized_runs = []
    for repeat in range(args.random_phase_repeats):
        generators = {
            label: torch.Generator(device=device).manual_seed(
                args.seed + 10000 * (repeat + 1) + offset
            )
            for offset, label in enumerate(("rayverse", *MODEL_LABELS))
        }
        randomized_reference = _random_global_phase(
            reference_csi, generator=generators["rayverse"]
        )
        randomized_models = {
            model: _random_global_phase(csi, generator=generators[model])
            for model, csi in model_csi.items()
        }
        evaluation = _evaluate_mode(
            randomized_reference,
            randomized_models,
            map_id,
            condition_groups,
            dimensions=dimensions,
            projection_seeds=projection_seeds,
            minimum_per_map=args.minimum_per_map,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + 1000 + repeat,
        )
        evaluation["repeat"] = repeat
        randomized_runs.append(evaluation)
    randomized = _aggregate_random_modes(
        randomized_runs,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed + 2000,
    )

    invariant_reference = _phase_normalize(reference_csi)
    invariant_models = {
        model: _phase_normalize(csi) for model, csi in model_csi.items()
    }
    invariant = _evaluate_mode(
        invariant_reference,
        invariant_models,
        map_id,
        condition_groups,
        dimensions=dimensions,
        projection_seeds=projection_seeds,
        minimum_per_map=args.minimum_per_map,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed + 3000,
    )
    validation_inputs = {"rayverse": reference_csi, **model_csi}
    invariant_errors: dict[str, float] = {}
    for offset, (label, csi) in enumerate(validation_inputs.items()):
        validation_generator = torch.Generator(device=device).manual_seed(
            args.seed + 99999 + offset
        )
        rotated = _random_global_phase(csi, generator=validation_generator)
        invariant_errors[label] = _relative_error(
            _phase_normalize(rotated), _phase_normalize(csi)
        )
    maximum_invariant_error = max(invariant_errors.values())

    report = {
        "format_version": 1,
        "diagnostic": "global_phase_full_csi_three_way_comparison",
        "source": str(Path(args.generated).resolve()),
        "samples": int(indices.numel()),
        "maps": int(torch.unique(map_id).numel()),
        "map_condition_groups": len(condition_groups),
        "selection": {
            "same_source_indices_for_all_models_and_modes": True,
            "source_indices": indices.tolist(),
            "selection_seed": args.seed,
        },
        "artifact_labels": {
            "reference": "RayVerse",
            "physics_only_baseline": "B3",
            "correct_map": "B3-S-ID",
        },
        "configuration": {
            "projection_dimensions": list(dimensions),
            "projection_seeds": list(projection_seeds),
            "phase_probe_count": PHASE_PROBE_COUNT,
            "phase_probe_seed": PHASE_PROBE_SEED,
            "random_phase_repeats": args.random_phase_repeats,
            "minimum_per_map": args.minimum_per_map,
            "minimum_condition_group": args.minimum_condition_group,
            "maximum_condition_groups_per_stratum": (
                args.maximum_condition_groups_per_stratum
            ),
            "bootstrap_samples": args.bootstrap_samples,
        },
        "A_raw_full_csi": raw,
        "B_independent_random_global_phase": randomized,
        "C_global_phase_invariant": invariant,
        "phase_invariance_validation": {
            "relative_error_by_source": invariant_errors,
            "maximum_relative_error_after_random_rotation_and_canonicalization": (
                maximum_invariant_error
            ),
            "status": "ok" if maximum_invariant_error < 1e-10 else "investigate",
        },
        "definitions": {
            "random_phase": (
                "RayVerse, B3 and B3-S-ID receive independent link-wise Uniform(-pi, pi) "
                "global phases in every repeat."
            ),
            "phase_invariant": (
                "Each link is divided by the phase of its strongest response across 16 fixed "
                "complex probes. Probe selection is itself global-phase invariant, avoiding "
                "ties among equal-magnitude CSI coefficients while preserving relative phase, "
                "magnitude, frequency and spatial structure."
            ),
            "map_condition_groups": (
                "Exact Map/LoS/K-validity strata are split into balanced neighborhoods along "
                "their leading standardized physical-condition direction."
            ),
        },
        "interpretation": (
            "Lower distance is better. Paired summaries define positive improvement as "
            "B3 distance minus B3-S-ID distance, so positive values favor B3-S-ID."
        ),
    }
    atomic_json_dump(report, args.output)
    print(
        json.dumps(
            {
                "event": "global_phase_diagnostic_complete",
                "output": str(Path(args.output).resolve()),
                "samples": int(indices.numel()),
                "condition_groups": len(condition_groups),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
