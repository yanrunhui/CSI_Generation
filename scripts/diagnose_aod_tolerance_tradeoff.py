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

from physics_csi.physics.aod_control import (
    AoDControlProjector,
    circular_difference_degrees,
)
from physics_csi.physics.synthesizer import MIMOOFDMSynthesizer
from physics_csi.training.trainer import resolve_device
from physics_csi.utils.io import atomic_json_dump
from scripts.evaluate_dual_mode_aod_fidelity import (
    _angle_audit,
    _greedy_condition_match,
    _metrics,
    _real_real_floor,
    _resolve_system,
    _subset,
    _synthesize,
    _validate_artifact,
)


def _float_list(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from error
    if not result or any(not math.isfinite(item) for item in result):
        raise argparse.ArgumentTypeError("the list must contain finite values")
    return result


def _int_list(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not result:
        raise argparse.ArgumentTypeError("the list cannot be empty")
    return result


def _mode_list(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result or any(item not in {"mean", "strongest"} for item in result):
        raise argparse.ArgumentTypeError("modes must contain mean and/or strongest")
    return result


def _pool_success_statistics(
    eligible: torch.Tensor,
    *,
    requested_samples: int,
    multipliers: tuple[int, ...],
    trials: int,
    seed: int,
) -> dict[str, Any]:
    eligible = eligible.bool().cpu()
    population = int(eligible.numel())
    generator = torch.Generator().manual_seed(seed)
    result: dict[str, Any] = {}
    for multiplier in multipliers:
        requested_pool = requested_samples * multiplier
        if requested_pool > population:
            result[str(multiplier)] = {
                "status": "insufficient_unique_artifact_candidates",
                "requested_pool_samples": requested_pool,
                "available_unique_candidates": population,
            }
            continue
        accepted: list[int] = []
        for _ in range(trials):
            indices = torch.randperm(population, generator=generator)[:requested_pool]
            accepted.append(int(eligible[indices].sum()))
        values = torch.tensor(accepted, dtype=torch.float32)
        result[str(multiplier)] = {
            "status": "ok",
            "requested_pool_samples": requested_pool,
            "trials": trials,
            "mean_accepted_samples": float(values.mean()),
            "p05_accepted_samples": float(torch.quantile(values, 0.05)),
            "minimum_accepted_samples": int(values.min()),
            "probability_of_filling_request": float(
                (values >= requested_samples).float().mean()
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AoD tolerance versus candidate-pool size and conditional fidelity."
    )
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--link-checkpoint", default=None)
    parser.add_argument("--cache", default=None)
    parser.add_argument("--map-ids", default="10")
    parser.add_argument("--path-counts", type=_int_list, default=_int_list("5"))
    parser.add_argument(
        "--modes", type=_mode_list, default=_mode_list("mean,strongest")
    )
    parser.add_argument(
        "--target-angles", type=_float_list, default=_float_list("-150,-90,-30,30,90,150")
    )
    parser.add_argument(
        "--tolerances-deg", type=_float_list, default=_float_list("0,5,10,15")
    )
    parser.add_argument(
        "--candidate-multipliers", type=_int_list, default=_int_list("5,10,20")
    )
    parser.add_argument("--requested-samples", type=int, default=8)
    parser.add_argument("--pool-trials", type=int, default=200)
    parser.add_argument("--minimum-fidelity-samples", type=int, default=8)
    parser.add_argument("--maximum-fidelity-samples", type=int, default=64)
    parser.add_argument("--minimum-mean-concentration", type=float, default=0.10)
    parser.add_argument("--strongest-near-tie-margin", type=float, default=0.02)
    parser.add_argument("--swd-projections", type=int, default=64)
    parser.add_argument("--projection-dimension", type=int, default=64)
    parser.add_argument("--projection-seeds", type=_int_list, default=_int_list("0,1,2"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.requested_samples < 1 or args.pool_trials < 1:
        parser.error("requested-samples and pool-trials must be positive")
    if args.minimum_fidelity_samples < 4:
        parser.error("minimum-fidelity-samples must be at least four")
    if args.maximum_fidelity_samples < args.minimum_fidelity_samples:
        parser.error("maximum-fidelity-samples cannot be smaller than the minimum")
    if any(value < 0 or value > 180 for value in args.tolerances_deg):
        parser.error("tolerances must lie in [0, 180]")
    if any(value < 1 for value in args.candidate_multipliers):
        parser.error("candidate multipliers must be positive")

    device = resolve_device(args.device)
    artifact = torch.load(args.artifact, map_location="cpu", weights_only=False)
    _validate_artifact(artifact)
    cache_path = next(
        (
            candidate
            for candidate in (artifact.get("cache"), args.cache)
            if candidate and Path(candidate).is_file()
        ),
        None,
    )
    b3_cache_seed = None
    if cache_path:
        cache_metadata = torch.load(cache_path, map_location="cpu", weights_only=False)
        b3_cache_seed = cache_metadata.get("sampling", {}).get("seed")
    system, system_source = _resolve_system(
        artifact,
        link_checkpoint_path=args.link_checkpoint,
        cache_path=args.cache,
    )
    synthesizer = MIMOOFDMSynthesizer(system).to(device).eval()
    projector = AoDControlProjector(
        minimum_mean_concentration=args.minimum_mean_concentration,
        strongest_near_tie_margin=args.strongest_near_tie_margin,
    )
    reference_cpu = artifact["reference"]
    natural_cpu = artifact["correct_map"]
    available_maps = sorted(int(value) for value in torch.unique(reference_cpu["map_id"]))
    map_ids = (
        available_maps
        if args.map_ids.strip().lower() == "all"
        else list(_int_list(args.map_ids))
    )
    missing = sorted(set(map_ids) - set(available_maps))
    if missing:
        raise ValueError(f"artifact has no requested Maps: {missing}")
    path_count = reference_cpu["path_mask"].bool().sum(-1)
    los = reference_cpu["conditions"][:, 3] >= 0.5
    k_valid = reference_cpu["condition_validity"][:, 2].bool()
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for map_id in map_ids:
            for requested_l in args.path_counts:
                for los_value in (False, True):
                    for k_valid_value in (False, True):
                        stratum = torch.nonzero(
                            (reference_cpu["map_id"] == map_id)
                            & (path_count == requested_l)
                            & (los == los_value)
                            & (k_valid == k_valid_value),
                            as_tuple=False,
                        ).flatten()
                        if stratum.numel() < args.minimum_fidelity_samples:
                            continue
                        real = _subset(reference_cpu, stratum, device)
                        natural = _subset(natural_cpu, stratum, device)
                        natural["conditions"] = real["conditions"]
                        natural["condition_validity"] = real["condition_validity"]
                        natural["map_id"] = real["map_id"]
                        for mode_number, mode in enumerate(args.modes):
                            real_angle, real_valid, _ = _angle_audit(real, projector, mode)
                            natural_angle, natural_valid, _ = _angle_audit(
                                natural, projector, mode
                            )
                            for target_number, target in enumerate(args.target_angles):
                                target_tensor = torch.tensor(target, device=device)
                                real_error = circular_difference_degrees(
                                    real_angle, target_tensor
                                ).abs()
                                natural_error = circular_difference_degrees(
                                    natural_angle, target_tensor
                                ).abs()
                                for tolerance_number, tolerance in enumerate(
                                    args.tolerances_deg
                                ):
                                    local_seed = (
                                        args.seed
                                        + map_id * 100003
                                        + requested_l * 1009
                                        + int(los_value) * 101
                                        + mode_number * 31
                                        + target_number * 7
                                        + tolerance_number
                                    )
                                    real_selected = torch.nonzero(
                                        real_valid & (real_error <= tolerance),
                                        as_tuple=False,
                                    ).flatten()
                                    natural_eligible = natural_valid & (
                                        natural_error <= tolerance
                                    )
                                    natural_selected = torch.nonzero(
                                        natural_eligible, as_tuple=False
                                    ).flatten()
                                    row: dict[str, Any] = {
                                        "map_id": map_id,
                                        "L": requested_l,
                                        "los_status": int(los_value),
                                        "k_factor_valid": k_valid_value,
                                        "mode": mode,
                                        "target_az_deg": target,
                                        "tolerance_deg": tolerance,
                                        "stratum_samples": int(stratum.numel()),
                                        "real_support": int(real_selected.numel()),
                                        "rayverse_support_rate": float(
                                            real_selected.numel()
                                            / max(stratum.numel(), 1)
                                        ),
                                        "natural_accepted_samples": int(
                                            natural_selected.numel()
                                        ),
                                        "natural_acceptance_rate": float(
                                            natural_eligible.float().mean()
                                        ),
                                        "candidate_pool_statistics": _pool_success_statistics(
                                            natural_eligible,
                                            requested_samples=args.requested_samples,
                                            multipliers=args.candidate_multipliers,
                                            trials=args.pool_trials,
                                            seed=local_seed,
                                        ),
                                    }
                                    if (
                                        real_selected.numel()
                                        >= args.minimum_fidelity_samples
                                        and natural_selected.numel()
                                        >= args.minimum_fidelity_samples
                                    ):
                                        real_candidates = {
                                            key: value[real_selected]
                                            for key, value in real.items()
                                        }
                                        natural_candidates = {
                                            key: value[natural_selected]
                                            for key, value in natural.items()
                                        }
                                        real_match, natural_match, match_audit = (
                                            _greedy_condition_match(
                                                real_candidates,
                                                natural_candidates,
                                                args.maximum_fidelity_samples,
                                            )
                                        )
                                        matched_real = {
                                            key: value[real_match.to(device)]
                                            for key, value in real_candidates.items()
                                        }
                                        realism_first = {
                                            key: value[natural_match.to(device)]
                                            for key, value in natural_candidates.items()
                                        }
                                        exact_output = projector(
                                            realism_first, target, mode=mode
                                        )
                                        exact_control = projector.apply_to_payload(
                                            realism_first, exact_output
                                        )
                                        exact_control["csi"] = _synthesize(
                                            exact_control, synthesizer
                                        )
                                        row["status"] = "fidelity_evaluated"
                                        row["condition_match"] = match_audit
                                        row["matched_samples"] = int(real_match.numel())
                                        row["selected_natural_error_deg"] = {
                                            "mean": float(
                                                exact_output.rotation_deg.abs().mean()
                                            ),
                                            "p95": float(
                                                torch.quantile(
                                                    exact_output.rotation_deg.abs(), 0.95
                                                )
                                            ),
                                            "maximum": float(
                                                exact_output.rotation_deg.abs().max()
                                            ),
                                        }
                                        row["realism_first_vs_real"] = _metrics(
                                            realism_first,
                                            matched_real,
                                            projections=args.swd_projections,
                                            projection_dimension=args.projection_dimension,
                                            projection_seeds=args.projection_seeds,
                                            seed=local_seed,
                                        )
                                        row["exact_control_vs_real"] = _metrics(
                                            exact_control,
                                            matched_real,
                                            projections=args.swd_projections,
                                            projection_dimension=args.projection_dimension,
                                            projection_seeds=args.projection_seeds,
                                            seed=local_seed,
                                        )
                                        row["real_vs_real"] = _real_real_floor(
                                            matched_real,
                                            swd_projections=args.swd_projections,
                                            projection_dimension=args.projection_dimension,
                                            projection_seeds=args.projection_seeds,
                                            seed=local_seed + 1000,
                                        )
                                    else:
                                        row["status"] = "insufficient_fidelity_support"
                                    rows.append(row)
                                    print(
                                        json.dumps(
                                            {
                                                "event": "aod_tolerance_row_complete",
                                                "map_id": map_id,
                                                "L": requested_l,
                                                "mode": mode,
                                                "target": target,
                                                "tolerance": tolerance,
                                            },
                                            sort_keys=True,
                                        ),
                                        flush=True,
                                    )
    report = {
        "format_version": 1,
        "experiment": "aod_tolerance_candidate_multiplier_tradeoff",
        "artifact": str(Path(args.artifact).resolve()),
        "system_configuration_source": system_source,
        "maps_requested": map_ids,
        "path_counts_requested": list(args.path_counts),
        "modes_requested": list(args.modes),
        "artifact_evaluation_seed": artifact.get("seed"),
        "b3_cache_generation_seed": b3_cache_seed,
        "target_angles_deg": list(args.target_angles),
        "tolerances_deg": list(args.tolerances_deg),
        "candidate_multipliers": list(args.candidate_multipliers),
        "requested_samples_per_pool": args.requested_samples,
        "pool_trials": args.pool_trials,
        "rows": rows,
        "interpretation": [
            "realism_first accepts natural links inside the stated tolerance and applies no angle rotation.",
            "exact_control uses the same accepted links and rotates them to zero final angle error, isolating the projection effect.",
            "Candidate-pool success probabilities use unique samples within each Monte Carlo pool; no candidate is duplicated inside a trial.",
            "Pool probabilities are marginal estimates over the reported Map/L/LoS/K-validity stratum; use the generation interface for repeated candidates at one exact continuous physical-condition tuple.",
            "A tolerance with inadequate angle-matched RayVerse support cannot establish conditional fidelity.",
        ],
    }
    atomic_json_dump(report, args.output)
    print(
        json.dumps(
            {
                "event": "aod_tolerance_tradeoff_complete",
                "output": str(Path(args.output).resolve()),
                "rows": len(rows),
                "fidelity_rows": sum(
                    item["status"] == "fidelity_evaluated" for item in rows
                ),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
