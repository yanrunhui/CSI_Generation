#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_csi.data.scene_id import make_scene_id_datasets
from physics_csi.evaluation.scene_support import SceneConditionSupportChecker
from physics_csi.models.diffusion.multipath_diffusion import MultipathDiffusionModel
from physics_csi.physics.synthesizer import SystemConfig
from physics_csi.training.trainer import resolve_device


def _cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    return value


def _map_id(value: str) -> int:
    normalized = value.strip().lower()
    if normalized.startswith("map_"):
        normalized = normalized[4:]
    try:
        return int(normalized)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Map ID must look like 10 or map_10") from error


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CSI with B3-S-ID.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--map-id", type=_map_id, required=True)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--total-power-db", type=float, required=True)
    parser.add_argument("--delay-spread-ns", type=float, required=True)
    parser.add_argument("--k-factor-db", type=float, default=0.0)
    parser.add_argument("--los-status", type=int, choices=[0, 1], required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sampler", choices=["ddpm", "ddim"], default=None)
    parser.add_argument("--sampling-steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--path-count", type=int, default=None)
    parser.add_argument("--path-count-temperature", type=float, default=None)
    parser.add_argument(
        "--skip-support-check",
        action="store_true",
        help="Skip the empirical warning only; unknown Map IDs still raise an error.",
    )
    args = parser.parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("model_kind") != "multipath_diffusion_scene_id":
        raise ValueError("checkpoint is not a B3-S-ID model")
    config = checkpoint["config"]
    plan = config.get("scene_id_experiment")
    if plan is None:
        raise ValueError("checkpoint does not contain the scene split/mapping metadata")
    model = MultipathDiffusionModel(
        config["multipath_diffusion"],
        SystemConfig.from_mapping(config["data"]["system"]),
        checkpoint["statistics"],
    ).to(device)
    model.load_state_dict(checkpoint.get("ema_state_dict", checkpoint["model_state_dict"]))
    model.eval()
    # Resolve once before expensive sampling and produce an explicit unseen-map error.
    model.map_indices(args.map_id, 1, device)
    conditions = {
        "total_power_db": args.total_power_db,
        "delay_spread_ns": args.delay_spread_ns,
        "k_factor_db": args.k_factor_db,
        "los_status": args.los_status,
    }
    support: dict[str, Any] | None = None
    if not args.skip_support_check:
        datasets = make_scene_id_datasets(config["data"]["prepared_root"], plan)
        checker = SceneConditionSupportChecker.from_dataset(
            datasets["train"], checkpoint["statistics"]
        )
        support = checker.check(args.map_id, conditions)
        print(json.dumps({"scene_condition_support": support}, indent=2), flush=True)
    sampling = config["sampling"]
    generator = torch.Generator(device=device).manual_seed(args.seed)
    started = time.perf_counter()
    generated = model.generate(
        conditions,
        args.num_samples,
        map_id=args.map_id,
        sampler=args.sampler or sampling["sampler"],
        sampling_steps=(
            args.sampling_steps
            if args.sampling_steps is not None
            else int(sampling["sampling_steps"])
        ),
        eta=float(sampling.get("eta", 0.0)),
        guidance_scale=(
            args.guidance_scale
            if args.guidance_scale is not None
            else float(sampling["guidance_scale"])
        ),
        path_count_temperature=(
            args.path_count_temperature
            if args.path_count_temperature is not None
            else float(sampling.get("path_count_temperature", 1.0))
        ),
        forced_path_count=args.path_count,
        generator=generator,
    )
    elapsed = time.perf_counter() - started
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_kind": "multipath_diffusion_scene_id",
            "generated": _cpu(generated),
            "conditions": conditions,
            "map_id": args.map_id,
            "map_name": f"map_{args.map_id}",
            "scene_condition_support": support,
            "seed": args.seed,
            "generation_metadata": {
                "seconds": elapsed,
                "samples_per_second": args.num_samples / max(elapsed, 1e-12),
                "sampler": args.sampler or sampling["sampler"],
                "sampling_steps": args.sampling_steps or int(sampling["sampling_steps"]),
                "guidance_scale": (
                    args.guidance_scale
                    if args.guidance_scale is not None
                    else float(sampling["guidance_scale"])
                ),
                "forced_path_count": args.path_count,
            },
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "scene_plan_fingerprint": plan["fingerprint"],
        },
        output,
    )


if __name__ == "__main__":
    main()
