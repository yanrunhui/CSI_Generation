#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_csi.models.conditional_vae import PhysicsConditionedVAE
from physics_csi.physics.synthesizer import SystemConfig
from physics_csi.training.trainer import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate controllable MIMO-OFDM CSI.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--total-power-db", type=float, required=True)
    parser.add_argument("--delay-spread-ns", type=float, required=True)
    parser.add_argument("--k-factor-db", type=float, default=0.0)
    parser.add_argument("--los-status", type=int, choices=[0, 1], required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    system = SystemConfig.from_mapping(config["data"]["system"])
    model = PhysicsConditionedVAE(config["model"], system).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    generator = torch.Generator(device=device).manual_seed(args.seed)
    conditions = {
        "total_power_db": args.total_power_db,
        "delay_spread_ns": args.delay_spread_ns,
        "k_factor_db": args.k_factor_db,
        "los_status": args.los_status,
    }
    generated = model.generate(
        conditions, args.num_samples, generator=generator
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "generated": {key: value.cpu() for key, value in generated.items()},
            "system": system.as_dict(),
            "conditions": conditions,
            "seed": args.seed,
            "checkpoint": str(Path(args.checkpoint).resolve()),
        },
        output,
    )


if __name__ == "__main__":
    main()

