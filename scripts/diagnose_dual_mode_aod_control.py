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

from physics_csi.evaluation.conditions import condition_accuracy
from physics_csi.evaluation.diffusion_comparison import pairwise_diversity
from physics_csi.data.physical_labels import compute_physical_conditions
from physics_csi.physics.aod_control import (
    AoDControlProjector,
    circular_difference_degrees,
)
from physics_csi.physics.constraints import check_condition_feasibility
from physics_csi.training.trainer import resolve_device
from physics_csi.utils.io import atomic_json_dump
from scripts.generate_dual_mode_aod_csi import (
    _load_models,
    _map_id,
    _support_audit,
    _synthesize,
)


def _condition(value: str) -> tuple[float, float, float, int, int]:
    fields = [item.strip() for item in value.split(",")]
    if len(fields) != 5:
        raise argparse.ArgumentTypeError("condition must be POWER,DS,K,LOS,L")
    try:
        power, spread, k = (float(item) for item in fields[:3])
        los, path_count = (int(item) for item in fields[3:])
    except ValueError as error:
        raise argparse.ArgumentTypeError("condition contains a non-numeric field") from error
    if los not in {0, 1} or path_count < 1:
        raise argparse.ArgumentTypeError("LOS must be 0/1 and L must be positive")
    return power, spread, k, los, path_count


def _float_list(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from error
    if not result or not all(math.isfinite(item) for item in result):
        raise argparse.ArgumentTypeError("list must contain finite numbers")
    return result


def _int_list(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not result:
        raise argparse.ArgumentTypeError("list cannot be empty")
    return result


def _safe_audit(values: torch.Tensor) -> dict[str, Any]:
    values = values.detach().float().flatten()
    if not values.numel():
        return {"count": 0}
    return {
        "count": int(values.numel()),
        "mean": float(values.mean()),
        "p50": float(torch.quantile(values, 0.50)),
        "p95": float(torch.quantile(values, 0.95)),
        "p99": float(torch.quantile(values, 0.99)),
        "maximum": float(values.max()),
    }


def _relative_interval_error(
    before: torch.Tensor, after: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    before_difference = circular_difference_degrees(
        before[:, :, None], before[:, None, :]
    )
    after_difference = circular_difference_degrees(
        after[:, :, None], after[:, None, :]
    )
    pair_mask = mask[:, :, None] & mask[:, None, :]
    return circular_difference_degrees(
        after_difference, before_difference
    ).abs()[pair_mask]


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for mode in ("mean", "strongest"):
        selected = [record for record in records if record["mode"] == mode]
        valid_errors = torch.tensor(
            [
                record["valid_control_error_maximum_deg"]
                for record in selected
                if math.isfinite(record["valid_control_error_maximum_deg"])
            ]
        )
        result[mode] = {
            "runs": len(selected),
            "runs_with_valid_control_samples": int(valid_errors.numel()),
            "maximum_valid_control_error_deg": (
                float(valid_errors.max()) if valid_errors.numel() else None
            ),
            "mean_valid_control_error_maximum_deg": (
                float(valid_errors.mean()) if valid_errors.numel() else None
            ),
            "minimum_path_count_accuracy": min(
                record["path_count_accuracy"] for record in selected
            ),
            "minimum_finite_csi_rate": min(
                record["finite_csi_rate"] for record in selected
            ),
            "maximum_relative_aod_interval_error_deg": max(
                record["relative_aod_interval_error_deg"]["maximum"]
                for record in selected
            ),
            "maximum_preserved_parameter_error": max(
                max(record["preservation_maximum_absolute_error"].values())
                for record in selected
            ),
            "diversity_ratio_range": [
                min(record["csi_diversity_ratio_controlled_to_natural"] for record in selected),
                max(record["csi_diversity_ratio_controlled_to_natural"] for record in selected),
            ],
            "mean_control_valid_rate": sum(
                record["control_valid_rate"] for record in selected
            )
            / len(selected),
            "mean_strongest_near_tie_rate": sum(
                record["strongest_near_tie_rate"] for record in selected
            )
            / len(selected),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Numerical-control diagnostic for Mean/Strongest AoD projection."
    )
    parser.add_argument("--b3-checkpoint", required=True)
    parser.add_argument("--dp-checkpoint", required=True)
    parser.add_argument("--link-checkpoint", required=True)
    parser.add_argument("--support-cache", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--map-id", type=_map_id, default=10)
    parser.add_argument("--samples-per-condition", type=int, default=128)
    parser.add_argument("--generation-seeds", type=_int_list, default=_int_list("2026,2027,2028"))
    parser.add_argument(
        "--target-angles",
        type=_float_list,
        default=_float_list("-180,-135,-90,-45,0,45,90,135"),
    )
    parser.add_argument("--condition", action="append", type=_condition, default=None)
    parser.add_argument("--minimum-mean-concentration", type=float, default=0.10)
    parser.add_argument("--strongest-near-tie-margin", type=float, default=0.02)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    conditions_spec = args.condition or [
        (-95.0, 30.0, 0.0, 0, 2),
        (-90.0, 30.0, 5.0, 1, 2),
        (-100.0, 100.0, 10.0, 1, 3),
        (-85.0, 50.0, 0.0, 0, 4),
        (-90.0, 100.0, 5.0, 1, 5),
    ]
    if args.samples_per_condition < 2:
        parser.error("samples-per-condition must be at least two")

    device = resolve_device(args.device)
    b3, dp, link, b3_checkpoint = _load_models(
        Path(args.b3_checkpoint),
        Path(args.dp_checkpoint),
        Path(args.link_checkpoint),
        device,
    )
    projector = AoDControlProjector(
        minimum_mean_concentration=args.minimum_mean_concentration,
        strongest_near_tie_margin=args.strongest_near_tie_margin,
    )
    synthesizer = b3.synthesizer
    sampling = b3_checkpoint["config"]["sampling"]
    records: list[dict[str, Any]] = []
    condition_support: list[dict[str, Any]] = []
    with torch.inference_mode():
        for condition_number, (power, spread, k, los, path_count) in enumerate(
            conditions_spec
        ):
            if path_count > b3.max_paths:
                raise ValueError(f"condition requests L={path_count} above max_paths")
            k_valid = bool(los and path_count >= 2)
            condition_dict = {
                "total_power_db": power,
                "delay_spread_ns": spread,
                "k_factor_db": k,
                "los_status": los,
            }
            check_condition_feasibility(
                condition_dict, path_count=path_count, k_factor_valid=k_valid
            )
            conditions = torch.tensor(
                [power, spread, k, los], device=device, dtype=torch.float32
            ).expand(args.samples_per_condition, -1)
            validity = torch.ones_like(conditions, dtype=torch.bool)
            validity[:, 2] = k_valid
            support = _support_audit(
                args.support_cache,
                map_id=args.map_id,
                conditions=conditions.detach().cpu(),
                validity=validity.detach().cpu(),
                path_count=path_count,
            )
            condition_support.append(
                {"condition_index": condition_number, "condition": condition_dict, "L": path_count, "support": support}
            )
            external_map = torch.full(
                (args.samples_per_condition,),
                args.map_id,
                device=device,
                dtype=torch.long,
            )
            map_index = link.map_indices(external_map)
            for generation_seed in args.generation_seeds:
                generator = torch.Generator(device=device).manual_seed(generation_seed)
                base = b3.generate(
                    conditions,
                    args.samples_per_condition,
                    condition_validity=validity,
                    sampler=sampling["sampler"],
                    sampling_steps=int(sampling["sampling_steps"]),
                    eta=float(sampling.get("eta", 0.0)),
                    guidance_scale=float(sampling["guidance_scale"]),
                    forced_path_count=path_count,
                    generator=generator,
                )
                dp_output = dp(base, conditions, validity, map_index=map_index)
                dp_payload = dp.apply_to_payload(base, dp_output)
                link_output = link(dp_payload, conditions, validity, map_index=map_index)
                natural = link.apply_to_payload(dp_payload, link_output)
                natural["csi"] = _synthesize(natural, synthesizer)
                for mode in ("mean", "strongest"):
                    for target in args.target_angles:
                        output = projector(natural, target, mode=mode)
                        controlled = projector.apply_to_payload(natural, output)
                        controlled["csi"] = _synthesize(controlled, synthesizer)
                        actual, actual_validity = compute_physical_conditions(
                            controlled["path_powers_linear"],
                            controlled["path_delays_ns"],
                            controlled["path_mask"],
                            controlled["direct_path_mask"],
                        )
                        accuracy = condition_accuracy(
                            conditions,
                            actual,
                            validity & actual_validity,
                        )
                        active_count = controlled["path_mask"].sum(-1)
                        finite = (
                            torch.isfinite(controlled["csi"].real)
                            & torch.isfinite(controlled["csi"].imag)
                        ).flatten(1).all(-1)
                        relative_error = _relative_interval_error(
                            natural["aod_az_deg"],
                            controlled["aod_az_deg"],
                            controlled["path_mask"],
                        )
                        preserve_keys = (
                            "path_delays_ns",
                            "path_powers_linear",
                            "path_phases_rad",
                            "aoa_az_deg",
                            "aoa_el_deg",
                            "aod_el_deg",
                        )
                        preservation = {
                            key: float(
                                (controlled[key] - natural[key]).abs().max()
                            )
                            for key in preserve_keys
                        }
                        natural_diversity = pairwise_diversity(natural["csi"])
                        controlled_diversity = pairwise_diversity(controlled["csi"])
                        valid_error = output.circular_error_deg[output.control_valid]
                        records.append(
                            {
                                "condition_index": condition_number,
                                "condition": condition_dict,
                                "path_count": path_count,
                                "generation_seed": generation_seed,
                                "mode": mode,
                                "target_az_deg": target,
                                "samples": args.samples_per_condition,
                                "control_valid_rate": float(output.control_valid.float().mean()),
                                "valid_control_error_deg": _safe_audit(valid_error),
                                "valid_control_error_maximum_deg": (
                                    float(valid_error.max()) if valid_error.numel() else float("nan")
                                ),
                                "all_control_error_deg": _safe_audit(output.circular_error_deg),
                                "rotation_absolute_deg": _safe_audit(output.rotation_deg.abs()),
                                "mean_resultant_concentration": _safe_audit(
                                    output.mean_resultant_concentration
                                ),
                                "strongest_near_tie_rate": float(
                                    output.strongest_near_tie.float().mean()
                                ),
                                "path_count_accuracy": float(
                                    (active_count == path_count).float().mean()
                                ),
                                "finite_csi_rate": float(finite.float().mean()),
                                "condition_accuracy": accuracy,
                                "relative_aod_interval_error_deg": _safe_audit(relative_error),
                                "preservation_maximum_absolute_error": preservation,
                                "natural_csi_pairwise_diversity": natural_diversity,
                                "controlled_csi_pairwise_diversity": controlled_diversity,
                                "csi_diversity_ratio_controlled_to_natural": (
                                    controlled_diversity / natural_diversity
                                    if natural_diversity > 1e-20
                                    else float("nan")
                                ),
                            }
                        )
                print(
                    json.dumps(
                        {
                            "event": "dual_mode_aod_condition_seed_complete",
                            "condition_index": condition_number,
                            "generation_seed": generation_seed,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    report = {
        "format_version": 1,
        "experiment": "dual_mode_aod_numerical_control",
        "map_id": args.map_id,
        "generation_seeds": list(args.generation_seeds),
        "target_angles_deg": list(args.target_angles),
        "samples_per_condition": args.samples_per_condition,
        "condition_support": condition_support,
        "aggregate": _aggregate(records),
        "runs": records,
        "success_rule": (
            "For valid samples, circular control error must be numerical, L and physical "
            "conditions must remain exact, Delay/Power/Phase/elevation and relative azimuth "
            "intervals must be unchanged, CSI must stay finite, and diversity must not collapse."
        ),
        "caution": (
            "This experiment proves numerical controllability only. Map-conditional fidelity "
            "requires separate angle-matched RayVerse reference sets for Mean and Strongest modes."
        ),
    }
    atomic_json_dump(report, args.output)
    print(
        json.dumps(
            {
                "event": "dual_mode_aod_control_diagnostic_complete",
                "output": str(Path(args.output).resolve()),
                "aggregate": report["aggregate"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
