#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
    CORE_DISTANCE_PATHS,
    PATH_KEYS,
    _align_donor_paths,
    _canonicalize_path_order,
    _condition_consistency,
    _core_improvements,
    _distance_metrics,
    _get_nested,
    _match_reference_donors,
    _metric_sanity,
    _path_integrity,
    _projection_changes,
    _subset,
    _synthesize,
)
from physics_csi.config import load_yaml
from physics_csi.physics.constraints import PhysicsConstraintLayer
from physics_csi.physics.synthesizer import MIMOOFDMSynthesizer, SystemConfig
from physics_csi.training.trainer import resolve_device
from physics_csi.utils.io import atomic_json_dump


REPAIRS = {
    "power_aod_joint": ("path_powers_linear", "aod_az_deg", "aod_el_deg"),
    "delay_power_joint": ("path_delays_ns", "path_powers_linear"),
    "delay_power_aod_joint": (
        "path_delays_ns",
        "path_powers_linear",
        "aod_az_deg",
        "aod_el_deg",
    ),
}


def _project(
    payload: Mapping[str, torch.Tensor],
    constraint: PhysicsConstraintLayer,
    synthesizer: MIMOOFDMSynthesizer,
    batch_size: int,
) -> tuple[dict[str, Any], Any, dict[str, torch.Tensor]]:
    projected: dict[str, Any] = _canonicalize_path_order(payload)
    before = {key: projected[key].clone() for key in PATH_KEYS}
    result = constraint(
        projected["path_delays_ns"],
        projected["path_powers_linear"].clamp_min(1e-20),
        projected["path_mask"],
        projected["target_conditions"],
        target_condition_validity=projected["target_condition_validity"],
        direct_path_mask=projected["direct_path_mask"],
    )
    projected["path_delays_ns"] = result.delays_ns
    projected["path_powers_linear"] = result.powers_linear
    projected["direct_path_mask"] = result.direct_path_mask
    projected["actual_conditions"] = result.actual_conditions
    projected["actual_condition_validity"] = result.condition_validity
    projected["csi"] = _synthesize(projected, synthesizer, batch_size)
    return projected, result, before


@torch.inference_mode()
def _repair_suite(
    generated: Mapping[str, torch.Tensor],
    donor: Mapping[str, torch.Tensor],
    *,
    constraint: PhysicsConstraintLayer,
    synthesizer: MIMOOFDMSynthesizer,
    batch_size: int,
) -> dict[str, Any]:
    donor_aligned, alignment_cost = _align_donor_paths(generated, donor)
    control, control_result, control_before = _project(
        generated, constraint, synthesizer, batch_size
    )
    control_metrics = _distance_metrics(control, donor)
    repairs: dict[str, Any] = {}
    for name, keys in REPAIRS.items():
        replaced: dict[str, Any] = dict(generated)
        for key in keys:
            replaced[key] = donor_aligned[key].clone()
        raw_integrity = _path_integrity(replaced)
        repaired, result, before = _project(
            replaced, constraint, synthesizer, batch_size
        )
        metrics = _distance_metrics(repaired, donor)
        projection = _projection_changes(
            before, repaired, replaced_parameters=keys
        )
        repairs[name] = {
            "parameters": list(keys),
            "metrics": metrics,
            "metric_sanity": _metric_sanity(metrics),
            "gain_control_minus_repaired": _core_improvements(
                control_metrics, metrics
            ),
            "constraint_feasible_rate": float(result.feasible.float().mean()),
            "condition_consistency": _condition_consistency(repaired),
            "projection_effect": projection,
            "delay_order_audit": {
                "after_replacement_before_canonicalization": raw_integrity[
                    "delay_order_violation_samples"
                ],
                "before_projection_after_canonicalization": _path_integrity(before)[
                    "delay_order_violation_samples"
                ],
                "after_projection": _path_integrity(repaired)[
                    "delay_order_violation_samples"
                ],
            },
        }
    return {
        "unrepaired_metrics": _distance_metrics(generated, donor),
        "projection_only_control": {
            "metrics": control_metrics,
            "metric_sanity": _metric_sanity(control_metrics),
            "constraint_feasible_rate": float(
                control_result.feasible.float().mean()
            ),
            "condition_consistency": _condition_consistency(control),
            "projection_effect": _projection_changes(
                control_before, control, replaced_parameters=()
            ),
        },
        "repairs": repairs,
        "path_alignment_cost_mean": float(alignment_cost.mean()),
        "path_alignment_cost_p90": float(torch.quantile(alignment_cost, 0.9)),
    }


def _differential(
    scene_suite: Mapping[str, Any], baseline_suite: Mapping[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for repair in REPAIRS:
        scene_repair = scene_suite["repairs"][repair]
        baseline_repair = baseline_suite["repairs"][repair]
        metrics: dict[str, Any] = {}
        for path in CORE_DISTANCE_PATHS:
            scene_gain = scene_repair["gain_control_minus_repaired"][path]
            baseline_gain = baseline_repair["gain_control_minus_repaired"][path]
            scene_after = _get_nested(scene_repair["metrics"], path)
            baseline_after = _get_nested(baseline_repair["metrics"], path)
            scene_before = _get_nested(
                scene_suite["projection_only_control"]["metrics"], path
            )
            baseline_before = _get_nested(
                baseline_suite["projection_only_control"]["metrics"], path
            )
            metrics[path] = {
                "b3_scene_id_before": scene_before,
                "b3_before": baseline_before,
                "b3_scene_id_after": scene_after,
                "b3_after": baseline_after,
                "b3_scene_id_gain": scene_gain,
                "b3_gain": baseline_gain,
                "differential_gain_scene_minus_b3": scene_gain - baseline_gain,
                "scene_minus_b3_error_before": scene_before - baseline_before,
                "scene_minus_b3_error_after": scene_after - baseline_after,
            }
        result[repair] = {
            "metrics": metrics,
            "scene_attribution_guard": scene_repair["projection_effect"][
                "single_group_attribution_guard"
            ],
            "b3_attribution_guard": baseline_repair["projection_effect"][
                "single_group_attribution_guard"
            ],
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply identical real-parameter repairs to B3 and B3-S-ID on common links."
    )
    parser.add_argument("--generated", required=True)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--maximum-samples", type=int, default=2048)
    parser.add_argument("--maximum-donor-distance", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = resolve_device(args.device)
    source = torch.load(args.generated, map_location="cpu", weights_only=False)
    for key in ("reference", "correct_map", "physics_only_baseline"):
        if source.get(key) is None:
            raise ValueError(f"generated artifact must contain {key}")
    sample_count = source["reference"]["csi"].shape[0]
    indices = torch.arange(sample_count)
    generator = torch.Generator().manual_seed(args.seed)
    if sample_count > args.maximum_samples:
        indices = torch.randperm(sample_count, generator=generator)[
            : args.maximum_samples
        ].sort().values
    reference = _subset(source["reference"], indices, device)
    scene = _subset(source["correct_map"], indices, device)
    baseline = _subset(source["physics_only_baseline"], indices, device)

    matched, donors, distances = _match_reference_donors(scene, reference)
    scene_count = scene["path_mask"].sum(-1)
    baseline_count = baseline["path_mask"].sum(-1)
    close_donor = distances <= args.maximum_donor_distance
    same_model_path_count = scene_count[matched] == baseline_count[matched]
    common = (
        close_donor
        & same_model_path_count
    )
    initially_matched = int(matched.numel())
    close_donor_samples = int(close_donor.sum())
    same_path_count_samples = int(same_model_path_count.sum())
    matched = matched[common]
    donors = donors[common]
    distances = distances[common]
    if matched.numel() < 64:
        raise ValueError("fewer than 64 common, close-donor counterfactual links")
    scene_common = _subset(scene, matched, device)
    baseline_common = _subset(baseline, matched, device)
    donor_common = _subset(reference, donors, device)
    target_difference = (
        scene_common["target_conditions"] - baseline_common["target_conditions"]
    ).abs()
    if float(target_difference.max()) > 1e-5:
        raise ValueError("B3 and B3-S-ID target conditions differ on the common subset")

    data = load_yaml(args.data_config)["data"]
    synthesizer = MIMOOFDMSynthesizer(
        SystemConfig.from_mapping(data["system"])
    ).to(device).eval()
    constraint = PhysicsConstraintLayer(
        enforce_total_power=True,
        enforce_delay_spread=True,
        enforce_k_factor=True,
        enforce_los_direct_path=True,
        epsilon=1e-20,
        invalid_k_fill_db=float(
            data.get("k_factor", {}).get("invalid_fill_db", 0.0)
        ),
    ).to(device).eval()

    scene_suite = _repair_suite(
        scene_common,
        donor_common,
        constraint=constraint,
        synthesizer=synthesizer,
        batch_size=args.batch_size,
    )
    baseline_suite = _repair_suite(
        baseline_common,
        donor_common,
        constraint=constraint,
        synthesizer=synthesizer,
        batch_size=args.batch_size,
    )
    unique_donors, reuse = torch.unique(donors, return_counts=True)
    report = {
        "format_version": 1,
        "diagnostic": "symmetric_b3_vs_b3_scene_id_counterfactual",
        "source": str(Path(args.generated).resolve()),
        "selected_samples": int(indices.numel()),
        "common_counterfactual_samples": int(matched.numel()),
        "common_sample_rate": float(matched.numel() / indices.numel()),
        "selection": {
            "same_generated_path_count_required": True,
            "maximum_donor_distance": args.maximum_donor_distance,
            "initial_scene_to_real_matches": initially_matched,
            "samples_passing_donor_distance": close_donor_samples,
            "samples_with_equal_b3_and_b3_scene_id_path_count": same_path_count_samples,
            "samples_passing_both_filters": int(matched.numel()),
            "condition_distance_mean": float(distances.mean()),
            "condition_distance_p90": float(torch.quantile(distances, 0.9)),
            "condition_distance_maximum": float(distances.max()),
            "unique_donor_rate": float(unique_donors.numel() / donors.numel()),
            "maximum_donor_reuse": int(reuse.max()),
            "target_condition_maximum_difference_between_models": float(
                target_difference.max()
            ),
        },
        "b3_scene_id": scene_suite,
        "b3": baseline_suite,
        "differential_recovery": _differential(scene_suite, baseline_suite),
        "interpretation": (
            "Positive differential_gain_scene_minus_b3 means B3-S-ID benefited more from "
            "the identical repair on the controlled common subset. It is evidence of a "
            "scene-specific recoverable gap, not proof that Map ID causally created it."
        ),
    }
    atomic_json_dump(report, args.output)
    print(
        json.dumps(
            {
                "event": "symmetric_counterfactual_complete",
                "output": str(Path(args.output).resolve()),
                "common_samples": int(matched.numel()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
