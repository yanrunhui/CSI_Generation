#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_csi.evaluation.distribution import rbf_mmd, sliced_wasserstein
from physics_csi.evaluation.scene_diagnostics import (
    paired_map_summary,
    spatial_correlation_matrix,
)
from physics_csi.training.trainer import resolve_device
from physics_csi.utils.io import atomic_json_dump


PHASE_PROBE_COUNT = 16
PHASE_PROBE_SEED = 314159
PRIMARY_METRICS = (
    "pdp_sliced_wasserstein",
    "raw_full_csi_projected_mmd_rbf",
    "raw_full_csi_projected_sliced_wasserstein",
    "phase_invariant_full_csi_projected_mmd_rbf",
    "phase_invariant_full_csi_projected_sliced_wasserstein",
    "tx_spatial_correlation_relative_frobenius",
    "tx_spatial_correlation_magnitude_mae",
    "tx_spatial_correlation_phase_mae_rad",
)


def _parse_comparison(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("comparison must be LABEL=ARTIFACT")
    label, raw_path = value.split("=", 1)
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", label):
        raise argparse.ArgumentTypeError("comparison label contains unsupported characters")
    return label, Path(raw_path)


def _same_tensor(first: torch.Tensor, second: torch.Tensor) -> bool:
    if first.shape != second.shape:
        return False
    if first.is_floating_point() or first.is_complex():
        return bool(
            torch.allclose(first, second, atol=1e-7, rtol=1e-6, equal_nan=True)
        )
    return bool(torch.equal(first, second))


def _validate_artifact(source: Mapping[str, Any]) -> None:
    if source.get("model_kind") != "b3_plus_dp_plus_aod_residual":
        raise ValueError("artifact is not a B3+DP+AoD residual evaluation")
    for section in ("reference", "dp_only", "correct_map"):
        if section not in source or "csi" not in source[section]:
            raise ValueError(f"artifact is missing {section}.csi")
        csi = source[section]["csi"]
        if csi.ndim != 4 or not csi.is_complex():
            raise ValueError(f"{section}.csi must be complex [sample, rx, tx, subcarrier]")
    if "map_id" not in source["reference"]:
        raise ValueError("reference payload has no map_id")


def _assert_same_reference(
    candidate: Mapping[str, Any], comparison: Mapping[str, Any]
) -> None:
    if candidate.get("scene_plan_fingerprint") != comparison.get(
        "scene_plan_fingerprint"
    ):
        raise ValueError("artifacts use different scene plans")
    for key in (
        "map_id",
        "conditions",
        "condition_validity",
        "path_mask",
        "path_delays_ns",
        "path_powers_linear",
        "aod_az_deg",
        "aod_el_deg",
        "csi",
    ):
        if not _same_tensor(candidate["reference"][key], comparison["reference"][key]):
            raise ValueError(f"artifacts do not contain identical reference.{key}")
    for key in (
        "path_mask",
        "path_delays_ns",
        "path_powers_linear",
        "aod_az_deg",
        "aod_el_deg",
        "csi",
    ):
        if not _same_tensor(candidate["dp_only"][key], comparison["dp_only"][key]):
            raise ValueError(f"artifacts do not contain identical dp_only.{key}")


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
    probes = probes / probes.abs().square().sum(-1, keepdim=True).sqrt().clamp_min(
        1e-20
    )
    anchors = flat @ probes.conj().transpose(0, 1)
    anchor = anchors.gather(1, anchors.abs().argmax(-1, keepdim=True))
    unit = anchor / anchor.abs().clamp_min(1e-20)
    shape = (csi.shape[0],) + (1,) * (csi.ndim - 1)
    return csi * unit.conj().reshape(shape)


def _energy_normalized_features(csi: torch.Tensor) -> torch.Tensor:
    flat = csi.flatten(1)
    flat = flat / flat.abs().square().sum(-1, keepdim=True).sqrt().clamp_min(1e-20)
    return torch.cat([flat.real, flat.imag], dim=-1).float()


def _normalized_pdp(csi: torch.Tensor) -> torch.Tensor:
    pdp = torch.fft.ifft(csi, dim=-1, norm="ortho").abs().square().mean((1, 2))
    return pdp / pdp.sum(-1, keepdim=True).clamp_min(1e-20)


def _robust_reference_standardize(
    values: Mapping[str, torch.Tensor], reference_label: str
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    reference = values[reference_label].float()
    mean = reference.mean(0, keepdim=True)
    raw_scale = reference.std(0, unbiased=False, keepdim=True)
    positive = raw_scale[torch.isfinite(raw_scale) & (raw_scale > 0)]
    typical = (
        positive.median()
        if positive.numel()
        else torch.ones((), device=reference.device)
    )
    floor = torch.maximum(typical * 0.05, torch.tensor(1e-6, device=reference.device))
    scale = torch.where(
        torch.isfinite(raw_scale), raw_scale, torch.zeros_like(raw_scale)
    ).clamp_min(floor)
    standardized = {
        label: (item.float() - mean) / scale for label, item in values.items()
    }
    return standardized, {
        "scale_floor": float(floor),
        "dimensions_at_floor": int((raw_scale <= floor).sum()),
        "dimensions": int(raw_scale.numel()),
    }


class FixedProjection:
    def __init__(
        self,
        reference: torch.Tensor,
        *,
        dimension: int,
        seeds: tuple[int, ...],
    ) -> None:
        self.dimension = int(dimension)
        self.seeds = seeds
        mean = reference.mean(0, keepdim=True)
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
        self.mean = mean
        self.scale = torch.where(
            torch.isfinite(raw_scale), raw_scale, torch.zeros_like(raw_scale)
        ).clamp_min(self.scale_floor)
        self.directions: dict[int, torch.Tensor] = {}
        for seed in seeds:
            generator = torch.Generator(device=reference.device).manual_seed(seed)
            signs = torch.randint(
                0,
                2,
                (reference.shape[1], self.dimension),
                generator=generator,
                device=reference.device,
                dtype=torch.int8,
            ).float()
            self.directions[seed] = (signs * 2.0 - 1.0) / math.sqrt(
                reference.shape[1]
            )

    def transform(self, features: torch.Tensor) -> dict[int, torch.Tensor]:
        standardized = (features - self.mean) / self.scale
        return {
            seed: standardized @ direction for seed, direction in self.directions.items()
        }

    def audit(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "seeds": list(self.seeds),
            "input_dimensions": int(self.mean.shape[1]),
            "scale_floor": float(self.scale_floor),
            "dimensions_at_floor": int((self.scale <= self.scale_floor).sum()),
        }


def _projection_metrics(
    candidate: Mapping[int, torch.Tensor],
    reference: Mapping[int, torch.Tensor],
    indices: torch.Tensor,
) -> tuple[float, float]:
    mmd_values: list[float] = []
    swd_values: list[float] = []
    for seed in sorted(candidate):
        first = candidate[seed][indices]
        second = reference[seed][indices]
        mmd_values.append(float(rbf_mmd(first, second, [2.0, 4.0, 8.0, 16.0])))
        swd_values.append(
            float(sliced_wasserstein(first, second, projections=128, seed=seed + 1000))
        )
    return sum(mmd_values) / len(mmd_values), sum(swd_values) / len(swd_values)


def _spatial_metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    candidate_matrix = spatial_correlation_matrix(candidate)
    reference_matrix = spatial_correlation_matrix(reference)
    difference = candidate_matrix - reference_matrix
    size = reference_matrix.shape[0]
    off_diagonal = ~torch.eye(size, dtype=torch.bool, device=reference_matrix.device)
    valid_phase = off_diagonal & (reference_matrix.abs() > 1e-3)
    phase_error = (
        torch.angle(
            candidate_matrix[valid_phase] * reference_matrix[valid_phase].conj()
        )
        .abs()
        .mean()
        if bool(valid_phase.any())
        else torch.zeros((), device=reference_matrix.device)
    )
    return {
        "tx_spatial_correlation_relative_frobenius": float(
            torch.linalg.matrix_norm(difference)
            / torch.linalg.matrix_norm(reference_matrix).clamp_min(1e-20)
        ),
        "tx_spatial_correlation_magnitude_mae": float(
            (candidate_matrix.abs() - reference_matrix.abs()).abs().mean()
        ),
        "tx_spatial_correlation_phase_mae_rad": float(phase_error),
    }


def _annotate_noninferiority(
    summary: dict[str, Any], relative_margin: float
) -> dict[str, Any]:
    baseline_mean = float(summary["comparison_mean_distance"])
    allowed_regression = relative_margin * baseline_mean
    lower, upper = (float(item) for item in summary["bootstrap_95_percent_ci"])
    if lower > 0:
        status = "superior"
    elif lower > -allowed_regression:
        status = "noninferior_within_margin"
    elif upper < -allowed_regression:
        status = "regressed_beyond_margin"
    else:
        status = "inconclusive_noninferiority"
    return summary | {
        "relative_noninferiority_margin": relative_margin,
        "absolute_allowed_regression": allowed_regression,
        "noninferiority_threshold_for_improvement": -allowed_regression,
        "noninferiority_status": status,
        "passes_noninferiority_gate": status
        in {"superior", "noninferior_within_margin"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired PDP, Full-CSI and Tx-spatial non-regression test for AoD residuals."
    )
    parser.add_argument("--candidate-artifact", required=True)
    parser.add_argument("--candidate-label", default="link_global")
    parser.add_argument(
        "--comparison",
        action="append",
        default=[],
        type=_parse_comparison,
        metavar="LABEL=ARTIFACT",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--minimum-per-map", type=int, default=12)
    parser.add_argument("--projection-dimension", type=int, default=64)
    parser.add_argument("--projection-seeds", default="0,1,2")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=22026)
    parser.add_argument("--relative-noninferiority-margin", type=float, default=0.05)
    args = parser.parse_args()
    if not 0 <= args.relative_noninferiority_margin < 1:
        parser.error("--relative-noninferiority-margin must be in [0, 1)")
    seeds = tuple(int(item) for item in args.projection_seeds.split(",") if item)
    if not seeds:
        parser.error("--projection-seeds must contain at least one integer")

    device = resolve_device(args.device)
    candidate_source = torch.load(
        args.candidate_artifact, map_location="cpu", weights_only=False
    )
    _validate_artifact(candidate_source)
    reference = candidate_source["reference"]
    model_csi: dict[str, torch.Tensor] = {
        args.candidate_label: candidate_source["correct_map"]["csi"],
        "dp_only": candidate_source["dp_only"]["csi"],
    }
    source_paths = {
        args.candidate_label: str(Path(args.candidate_artifact).resolve()),
        "dp_only": str(Path(args.candidate_artifact).resolve()),
    }
    for label, path in args.comparison:
        if label in model_csi:
            raise ValueError(f"duplicate comparison label: {label}")
        source = torch.load(path, map_location="cpu", weights_only=False)
        _validate_artifact(source)
        _assert_same_reference(candidate_source, source)
        model_csi[label] = source["correct_map"]["csi"]
        source_paths[label] = str(path.resolve())
        del source
    del candidate_source

    print(
        json.dumps(
            {
                "event": "feature_projection_start",
                "samples": int(reference["map_id"].numel()),
                "models": sorted(model_csi),
                "device": str(device),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    all_csi = {"reference": reference["csi"], **model_csi}
    device_csi = {label: value.to(device) for label, value in all_csi.items()}

    pdp_values = {label: _normalized_pdp(csi) for label, csi in device_csi.items()}
    pdp_values, pdp_audit = _robust_reference_standardize(pdp_values, "reference")

    raw_features = {
        label: _energy_normalized_features(csi) for label, csi in device_csi.items()
    }
    raw_basis = FixedProjection(
        raw_features["reference"], dimension=args.projection_dimension, seeds=seeds
    )
    raw_projected = {
        label: raw_basis.transform(values) for label, values in raw_features.items()
    }
    del raw_features

    invariant_features = {
        label: _energy_normalized_features(_phase_normalize(csi))
        for label, csi in device_csi.items()
    }
    invariant_basis = FixedProjection(
        invariant_features["reference"],
        dimension=args.projection_dimension,
        seeds=seeds,
    )
    invariant_projected = {
        label: invariant_basis.transform(values)
        for label, values in invariant_features.items()
    }
    del invariant_features

    map_ids = reference["map_id"].long().to(device)
    per_map: dict[str, Any] = {}
    for map_number, map_id in enumerate(torch.unique(map_ids, sorted=True), start=1):
        indices = torch.nonzero(map_ids == map_id, as_tuple=False).flatten()
        if indices.numel() < args.minimum_per_map:
            per_map[f"map_{int(map_id)}"] = {
                "map_id": int(map_id),
                "samples": int(indices.numel()),
                "status": "insufficient_samples",
            }
            continue
        models: dict[str, Any] = {}
        for label in model_csi:
            raw_mmd, raw_swd = _projection_metrics(
                raw_projected[label], raw_projected["reference"], indices
            )
            invariant_mmd, invariant_swd = _projection_metrics(
                invariant_projected[label], invariant_projected["reference"], indices
            )
            current_pdp = pdp_values[label][indices]
            reference_pdp = pdp_values["reference"][indices]
            metrics = {
                "pdp_mmd_rbf": float(
                    rbf_mmd(current_pdp, reference_pdp, [0.5, 1.0, 2.0, 4.0])
                ),
                "pdp_sliced_wasserstein": float(
                    sliced_wasserstein(
                        current_pdp,
                        reference_pdp,
                        projections=128,
                        seed=args.seed + int(map_id),
                    )
                ),
                "raw_full_csi_projected_mmd_rbf": raw_mmd,
                "raw_full_csi_projected_sliced_wasserstein": raw_swd,
                "phase_invariant_full_csi_projected_mmd_rbf": invariant_mmd,
                "phase_invariant_full_csi_projected_sliced_wasserstein": invariant_swd,
                **_spatial_metrics(
                    device_csi[label][indices], device_csi["reference"][indices]
                ),
            }
            models[label] = metrics
        per_map[f"map_{int(map_id)}"] = {
            "map_id": int(map_id),
            "samples": int(indices.numel()),
            "status": "ok",
            "models": models,
        }
        print(
            json.dumps(
                {
                    "event": "map_complete",
                    "map": int(map_id),
                    "map_index": map_number,
                    "maps_total": int(torch.unique(map_ids).numel()),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    maps = [item for item in per_map.values() if item.get("status") == "ok"]
    if not maps:
        raise RuntimeError("no Map meets --minimum-per-map")
    baseline_labels = [label for label in model_csi if label != args.candidate_label]
    metrics = tuple(next(iter(maps))["models"][args.candidate_label])
    summaries: dict[str, Any] = {}
    gates: dict[str, Any] = {}
    for baseline_number, baseline in enumerate(baseline_labels):
        name = f"{args.candidate_label}_vs_{baseline}"
        summaries[name] = {}
        for metric_number, metric in enumerate(metrics):
            paired = paired_map_summary(
                [item["models"][args.candidate_label][metric] for item in maps],
                [item["models"][baseline][metric] for item in maps],
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed + baseline_number * 1000 + metric_number,
            )
            summaries[name][metric] = _annotate_noninferiority(
                paired, args.relative_noninferiority_margin
            )
        primary = {metric: summaries[name][metric] for metric in PRIMARY_METRICS}
        failed = [
            metric
            for metric, values in primary.items()
            if not values["passes_noninferiority_gate"]
        ]
        gates[name] = {
            "status": "pass" if not failed else "fail",
            "failed_primary_metrics": failed,
            "primary_metrics": list(PRIMARY_METRICS),
        }

    report = {
        "format_version": 1,
        "diagnostic": "aod_residual_csi_nonregression",
        "candidate_label": args.candidate_label,
        "source_artifacts": source_paths,
        "samples": int(map_ids.numel()),
        "maps_evaluated": len(maps),
        "minimum_per_map": args.minimum_per_map,
        "metric_direction": "lower_is_better",
        "relative_noninferiority_margin": args.relative_noninferiority_margin,
        "configuration": {
            "projection_dimension": args.projection_dimension,
            "projection_seeds": list(seeds),
            "phase_probe_count": PHASE_PROBE_COUNT,
            "phase_probe_seed": PHASE_PROBE_SEED,
            "bootstrap_samples": args.bootstrap_samples,
        },
        "normalization_audit": {
            "pdp": pdp_audit,
            "raw_full_csi": raw_basis.audit(),
            "phase_invariant_full_csi": invariant_basis.audit(),
        },
        "nonregression_gates": gates,
        "summaries": summaries,
        "per_map": per_map,
        "interpretation": (
            "Positive paired improvement favors Link-global. Superior means the two-sided "
            "bootstrap interval is entirely above zero. Noninferior means its lower bound stays "
            "above the predeclared negative margin. Inconclusive or regressed metrics fail the gate."
        ),
        "cautions": [
            (
                "Map bootstrap measures spatial test consistency, not independent "
                "training-seed stability."
            ),
            (
                "Raw Full-CSI is global-phase sensitive; phase-invariant Full-CSI "
                "is reported separately."
            ),
            (
                "Tx spatial correlation is global-phase invariant but remains sensitive "
                "to AoD-driven array response."
            ),
        ],
    }
    atomic_json_dump(report, args.output)
    print(
        json.dumps(
            {
                "event": "aod_csi_nonregression_complete",
                "output": str(Path(args.output).resolve()),
                "gates": gates,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
