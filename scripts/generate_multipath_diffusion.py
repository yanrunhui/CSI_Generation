#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_csi.models.diffusion.multipath_diffusion import MultipathDiffusionModel
from physics_csi.physics.synthesizer import SystemConfig
from physics_csi.training.trainer import resolve_device


def _cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.cpu()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CSI through multipath diffusion.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
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
    parser.add_argument(
        "--path-count",
        type=int,
        default=None,
        help="Force L for a continuous-parameter ablation instead of sampling p(L|c).",
    )
    parser.add_argument(
        "--path-count-temperature",
        type=float,
        default=None,
        help="Override the calibrated categorical sampling temperature.",
    )
    args = parser.parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("model_kind") != "multipath_diffusion":
        raise ValueError("checkpoint is not a multipath diffusion model")
    config = checkpoint["config"]
    model = MultipathDiffusionModel(
        config["multipath_diffusion"],
        SystemConfig.from_mapping(config["data"]["system"]),
        checkpoint["statistics"],
    ).to(device)
    model.load_state_dict(checkpoint.get("ema_state_dict", checkpoint["model_state_dict"]))
    model.eval()
    sampling = config["sampling"]
    generator = torch.Generator(device=device).manual_seed(args.seed)
    conditions = {
        "total_power_db": args.total_power_db,
        "delay_spread_ns": args.delay_spread_ns,
        "k_factor_db": args.k_factor_db,
        "los_status": args.los_status,
    }
    started = time.perf_counter()
    generated = model.generate(
        conditions,
        args.num_samples,
        sampler=args.sampler or sampling["sampler"],
        sampling_steps=args.sampling_steps if args.sampling_steps is not None else int(sampling["sampling_steps"]),
        eta=float(sampling.get("eta", 0.0)),
        guidance_scale=args.guidance_scale if args.guidance_scale is not None else float(sampling["guidance_scale"]),
        path_count_temperature=(
            args.path_count_temperature
            if args.path_count_temperature is not None
            else float(sampling.get("path_count_temperature", 1.0))
        ),
        forced_path_count=args.path_count,
        generator=generator,
    )
    generation_seconds = time.perf_counter() - started
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 2,
            "model_kind": "multipath_diffusion",
            "generated": _cpu(generated),
            "conditions": conditions,
            "seed": args.seed,
            "generation_metadata": {
                "seconds": generation_seconds,
                "samples_per_second": args.num_samples / max(generation_seconds, 1e-12),
                "sampler": args.sampler or sampling["sampler"],
                "sampling_steps": args.sampling_steps if args.sampling_steps is not None else int(sampling["sampling_steps"]),
                "guidance_scale": args.guidance_scale if args.guidance_scale is not None else float(sampling["guidance_scale"]),
                "forced_path_count": args.path_count,
                "path_count_temperature": (
                    args.path_count_temperature
                    if args.path_count_temperature is not None
                    else float(sampling.get("path_count_temperature", 1.0))
                ),
            },
            "checkpoint": str(Path(args.checkpoint).resolve()),
        },
        output,
    )


if __name__ == "__main__":
    main()

