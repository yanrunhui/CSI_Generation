#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_csi.data.scene_id import load_seen_map_spatial_plan, make_scene_id_datasets
from physics_csi.models.diffusion.multipath_diffusion import MultipathDiffusionModel
from physics_csi.physics.synthesizer import SystemConfig
from physics_csi.training.trainer import resolve_device
from physics_csi.utils.seed import seed_everything


REFERENCE_KEYS = (
    "path_mask",
    "path_delays_ns",
    "path_powers_linear",
    "path_phases_rad",
    "aoa_az_deg",
    "aoa_el_deg",
    "aod_az_deg",
    "aod_el_deg",
    "direct_path_mask",
    "conditions",
    "condition_validity",
    "map_id",
)
BASE_KEYS = (
    "path_mask",
    "path_delays_ns",
    "path_powers_linear",
    "path_phases_rad",
    "aoa_az_deg",
    "aoa_el_deg",
    "aod_az_deg",
    "aod_el_deg",
    "direct_path_mask",
    "target_conditions",
    "target_condition_validity",
    "actual_conditions",
    "actual_condition_validity",
    "constraint_feasible",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _append(chunks: dict[str, list[torch.Tensor]], payload: Mapping[str, Any], keys: tuple[str, ...]) -> None:
    for key in keys:
        value = payload[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{key} must be a tensor")
        chunks.setdefault(key, []).append(value.detach().cpu())


def _finish(chunks: Mapping[str, list[torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.cat(values, dim=0) for key, values in chunks.items()}


def _subset_batch(
    payload: Mapping[str, Any], keep: torch.Tensor, batch_size: int
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, torch.Tensor) and value.shape[:1] == (batch_size,):
            result[key] = value[keep]
        elif isinstance(value, Mapping):
            result[key] = _subset_batch(value, keep, batch_size)
        else:
            result[key] = value
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache independent frozen-B3 samples for unpaired DP-residual training."
    )
    parser.add_argument("--b3-checkpoint", required=True)
    parser.add_argument("--scene-plan", required=True)
    parser.add_argument("--prepared-root", default="artifacts/prepared_d2los")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sampler", choices=["ddpm", "ddim"], default=None)
    parser.add_argument("--sampling-steps", type=int, default=None)
    parser.add_argument("--maximum-samples-per-split", type=int, default=None)
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch size must be positive and workers cannot be negative")

    seed_everything(args.seed)
    device = resolve_device(args.device)
    checkpoint_path = Path(args.b3_checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    system = SystemConfig.from_mapping(config["data"]["system"])
    model = MultipathDiffusionModel(
        config["multipath_diffusion"], system, checkpoint["statistics"]
    ).to(device)
    if model.scene_aware_enabled:
        raise ValueError("The cache backbone must be the Map-independent B3 checkpoint")
    model.load_state_dict(checkpoint.get("ema_state_dict", checkpoint["model_state_dict"]))
    model.requires_grad_(False).eval()
    sampling = config["sampling"]
    sampler = args.sampler or sampling["sampler"]
    sampling_steps = args.sampling_steps or int(sampling["sampling_steps"])

    plan = load_seen_map_spatial_plan(args.scene_plan)
    datasets = make_scene_id_datasets(args.prepared_root, plan)
    cached_splits: dict[str, Any] = {}
    exclusion_audit: dict[str, Any] = {}
    generator = torch.Generator(device=device).manual_seed(args.seed)
    for split_number, split in enumerate(("train", "validation", "test")):
        dataset = datasets[split]
        if args.maximum_samples_per_split is not None:
            maximum = min(len(dataset), args.maximum_samples_per_split)
            dataset = torch.utils.data.Subset(dataset, range(maximum))
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
        )
        reference_chunks: dict[str, list[torch.Tensor]] = {}
        base_chunks: dict[str, list[torch.Tensor]] = {}
        completed = 0
        retained = 0
        excluded_single_path_positive_spread = 0
        excluded_single_path_valid_k = 0
        started = time.monotonic()
        for batch_number, batch in enumerate(loader, start=1):
            raw_batch_size = int(batch["conditions"].shape[0])
            path_count_cpu = batch["path_count"].long()
            single_path = path_count_cpu < 2
            positive_spread = batch["conditions"][:, 1].float() > 0.0
            valid_k = batch["condition_validity"][:, 2].bool()
            infeasible = single_path & (positive_spread | valid_k)
            excluded_single_path_positive_spread += int(
                (single_path & positive_spread).sum()
            )
            excluded_single_path_valid_k += int((single_path & valid_k).sum())
            keep = ~infeasible
            completed += raw_batch_size
            if not bool(keep.any()):
                continue
            if bool(infeasible.any()):
                batch = _subset_batch(batch, keep, raw_batch_size)
            conditions = batch["conditions"].to(device, non_blocking=True)
            validity = batch["condition_validity"].to(device, non_blocking=True)
            path_count = batch["path_count"].to(device, non_blocking=True)
            generated = model.generate(
                conditions,
                int(conditions.shape[0]),
                condition_validity=validity,
                sampler=sampler,
                sampling_steps=sampling_steps,
                eta=float(sampling.get("eta", 0.0)),
                guidance_scale=float(sampling["guidance_scale"]),
                path_count_temperature=float(sampling.get("path_count_temperature", 1.0)),
                forced_path_count=path_count,
                generator=generator,
            )
            _append(reference_chunks, batch, REFERENCE_KEYS)
            _append(base_chunks, generated, BASE_KEYS)
            retained += int(conditions.shape[0])
            if batch_number % 20 == 0 or completed == len(dataset):
                print(
                    json.dumps(
                        {
                            "event": "b3_cache_progress",
                            "split": split,
                            "complete": completed,
                            "retained": retained,
                            "excluded_physically_infeasible": completed - retained,
                            "total": len(dataset),
                            "samples_per_second": retained / max(time.monotonic() - started, 1e-9),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        reference = _finish(reference_chunks)
        base = _finish(base_chunks)
        if not torch.equal(reference["path_mask"].sum(-1), base["path_mask"].sum(-1)):
            raise RuntimeError("forced path-count contract was not preserved")
        if not torch.equal(reference["conditions"], base["target_conditions"]):
            raise RuntimeError("B3 cache conditions no longer match the reference anchors")
        cached_splits[split] = {"reference": reference, "base": base}
        exclusion_audit[split] = {
            "input_samples": len(dataset),
            "retained_samples": retained,
            "excluded_samples": len(dataset) - retained,
            "single_path_with_positive_delay_spread": excluded_single_path_positive_spread,
            "single_path_with_valid_k_factor": excluded_single_path_valid_k,
            "policy": (
                "Exclude rather than change path count or target conditions; categories may "
                "overlap, so their counts need not sum to excluded_samples."
            ),
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "kind": "b3_dp_residual_cache",
        "scene_plan_fingerprint": plan["fingerprint"],
        "map_id_to_index": plan["map_id_to_index"],
        "b3_checkpoint": str(checkpoint_path.resolve()),
        "b3_checkpoint_sha256": _sha256(checkpoint_path),
        "b3_model_kind": checkpoint.get("model_kind"),
        "b3_statistics": checkpoint["statistics"],
        "b3_multipath_config": config["multipath_diffusion"],
        "system": system.as_dict(),
        "sampling": {
            "sampler": sampler,
            "sampling_steps": sampling_steps,
            "eta": float(sampling.get("eta", 0.0)),
            "guidance_scale": float(sampling["guidance_scale"]),
            "seed": args.seed,
            "forced_to_reference_path_count": True,
        },
        "physical_feasibility_exclusions": exclusion_audit,
        "unpaired_contract": (
            "B3 noise is independent of RayVerse; conditions and path count are anchors only. "
            "Training losses compare neighborhood distributions and never sample pairs."
        ),
        "splits": cached_splits,
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    print(
        json.dumps(
            {
                "event": "b3_dp_residual_cache_complete",
                "output": str(output.resolve()),
                "samples": {
                    split: int(values["reference"]["map_id"].numel())
                    for split, values in cached_splits.items()
                },
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
