from __future__ import annotations

import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from physics_csi.data.physical_labels import compute_physical_conditions
from physics_csi.data.rayverse import (
    RayVerseLink,
    discover_propbin_files,
    ids_from_path,
    path_gain_from_loss,
    phase_from_distance,
    read_propbin,
)
from physics_csi.data.splits import MapSplit, make_map_split, spatial_split_name
from physics_csi.physics.synthesizer import MIMOOFDMSynthesizer, SystemConfig
from physics_csi.utils.io import atomic_json_dump


def prepare_dataset(config: dict[str, Any], *, overwrite: bool = False) -> dict[str, Any]:
    data_config = config["data"]
    output_root = Path(data_config["prepared_root"])
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Prepared root already exists: {output_root}. Pass overwrite=True explicitly."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    files = discover_propbin_files(
        data_config["raw_root"],
        max_sources_per_map=data_config.get("max_sources_per_map"),
    )
    if not files:
        raise FileNotFoundError(f"No RayVerse propbin files below {data_config['raw_root']}")
    files = _ordered_files(files, data_config["sampling"], int(data_config["seed"]))
    split_config = data_config["split"]
    map_ids = sorted({ids_from_path(path)[0] for path in files})
    map_split: MapSplit | None = None
    if split_config["mode"] == "cross_map":
        map_split = make_map_split(
            map_ids,
            float(split_config["train_ratio"]),
            float(split_config["validation_ratio"]),
            float(split_config["test_ratio"]),
            int(data_config["seed"]),
        )
    elif split_config["mode"] == "single_map":
        selected_map = int(split_config["single_map_id"])
        files = [path for path in files if ids_from_path(path)[0] == selected_map]
        if not files:
            raise ValueError(f"single_map_id={selected_map} contains no source files")
    else:
        raise ValueError(f"Unsupported split mode {split_config['mode']!r}")

    system = SystemConfig.from_mapping(data_config["system"])
    synthesizer = MIMOOFDMSynthesizer(system).cpu().eval()
    shard_size = int(data_config["shard_size"])
    max_samples = data_config.get("max_samples")
    max_samples = None if max_samples is None else int(max_samples)
    rng = np.random.default_rng(int(data_config["seed"]))
    buffers: dict[str, list[dict[str, Any]]] = defaultdict(list)
    entries: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    path_count_histogram: dict[int, int] = defaultdict(int)
    original_path_count_histogram: dict[int, int] = defaultdict(int)
    truncated_path_samples = 0
    observed_carriers: set[float] = set()
    total = 0

    for path in files:
        propbin = read_propbin(path, fallback_carrier_hz=system.carrier_hz)
        observed_carriers.add(propbin.carrier_hz)
        if not np.isclose(propbin.carrier_hz, system.carrier_hz):
            raise ValueError(
                f"Carrier mismatch: {path} implies {propbin.carrier_hz}, "
                f"configuration requests {system.carrier_hz}"
            )
        for link in propbin.iter_links(
            max_receivers=data_config.get("max_receivers_per_source"), rng=rng
        ):
            record = link_to_record(
                link,
                max_paths=int(data_config["max_paths"]),
                overflow_policy=str(data_config["overflow_policy"]),
                invalid_k_fill_db=float(data_config["k_factor"]["invalid_fill_db"]),
            )
            split = _split_for_link(link, split_config, map_split, int(data_config["seed"]))
            buffers[split].append(record)
            path_count_histogram[int(record["path_count"])] += 1
            original_path_count_histogram[int(record["original_path_count"])] += 1
            truncated_path_samples += int(record["path_count_truncated"])
            total += 1
            if len(buffers[split]) >= shard_size:
                _flush_shard(output_root, split, buffers[split], entries[split], synthesizer)
                counts[split] += len(buffers[split])
                buffers[split].clear()
            if max_samples is not None and total >= max_samples:
                break
        if max_samples is not None and total >= max_samples:
            break

    for split in ("train", "validation", "test"):
        if buffers[split]:
            _flush_shard(output_root, split, buffers[split], entries[split], synthesizer)
            counts[split] += len(buffers[split])
            buffers[split].clear()

    manifest = {
        "format_version": 1,
        "source": "RayVerse PROP v2",
        "raw_root": str(Path(data_config["raw_root"]).resolve()),
        "system": system.as_dict(),
        "max_paths": int(data_config["max_paths"]),
        "power_definition": data_config["path_power"],
        "phase_definition": data_config["phase"],
        "k_factor_definition": data_config["k_factor"],
        "split_mode": split_config["mode"],
        "map_split": map_split.as_dict() if map_split else None,
        "observed_carriers_hz": sorted(observed_carriers),
        "path_count_histogram": {
            str(key): value for key, value in sorted(path_count_histogram.items())
        },
        "original_path_count_histogram": {
            str(key): value for key, value in sorted(original_path_count_histogram.items())
        },
        "truncated_path_samples": truncated_path_samples,
        "path_count_label_semantics": (
            "retained_path_count; max_paths class includes overflow samples"
            if truncated_path_samples
            else "exact_path_count"
        ),
        "total_samples": total,
        "splits": {
            split: {"count": counts[split], "shards": entries[split]}
            for split in ("train", "validation", "test")
        },
    }
    atomic_json_dump(manifest, output_root / "manifest.json")
    return manifest


def link_to_record(
    link: RayVerseLink,
    *,
    max_paths: int,
    overflow_policy: str,
    invalid_k_fill_db: float,
) -> dict[str, Any]:
    indices = np.arange(link.path_count)
    if link.path_count > max_paths:
        if overflow_policy == "error":
            raise ValueError(
                f"{link.map_id}/{link.source_id}/{link.receiver_id} has "
                f"{link.path_count} paths, exceeding max_paths={max_paths}"
            )
        if overflow_policy != "strongest":
            raise ValueError(f"Unsupported overflow policy {overflow_policy!r}")
        indices = np.argsort(link.path_loss_db, kind="stable")[:max_paths]
        indices = indices[np.argsort(link.delay_ns[indices], kind="stable")]

    count = min(link.path_count, max_paths)
    mask = torch.zeros(max_paths, dtype=torch.bool)
    mask[:count] = True

    def pad(values: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        result = torch.zeros(max_paths, dtype=dtype)
        selected = torch.as_tensor(values[indices][:count].copy(), dtype=dtype)
        result[:count] = selected
        return result

    powers = pad(path_gain_from_loss(link.path_loss_db))
    delays = pad(link.delay_ns)
    interaction_count = pad(link.interaction_count, torch.int64)
    path_type = pad(link.path_type, torch.int64)
    direct = mask & (interaction_count == 0) & (path_type != -1)
    phases = pad(phase_from_distance(link.distance_m, link.carrier_hz))
    conditions, condition_validity = compute_physical_conditions(
        powers.unsqueeze(0),
        delays.unsqueeze(0),
        mask.unsqueeze(0),
        direct.unsqueeze(0),
        invalid_k_fill_db=invalid_k_fill_db,
    )
    return {
        "sample_id": f"map_{link.map_id}-source_{link.source_id}-rx_{link.receiver_id}",
        "map_id": torch.tensor(link.map_id, dtype=torch.int64),
        "source_id": torch.tensor(link.source_id, dtype=torch.int64),
        "receiver_id": torch.tensor(link.receiver_id, dtype=torch.int64),
        "tx_xyz_m": torch.from_numpy(link.tx_xyz_m.astype(np.float32)),
        "rx_xyz_m": torch.from_numpy(link.rx_xyz_m.astype(np.float32)),
        "path_count": torch.tensor(count, dtype=torch.int64),
        "original_path_count": torch.tensor(link.path_count, dtype=torch.int64),
        "path_count_truncated": torch.tensor(link.path_count > max_paths, dtype=torch.bool),
        "path_mask": mask,
        "direct_path_mask": direct,
        "path_delays_ns": delays,
        "path_powers_linear": powers,
        "path_phases_rad": phases,
        "aoa_az_deg": pad(link.aoa_az_deg),
        "aoa_el_deg": pad(link.aoa_el_deg),
        "aod_az_deg": pad(link.aod_az_deg),
        "aod_el_deg": pad(link.aod_el_deg),
        "interaction_count": interaction_count,
        "path_type": path_type,
        "conditions": conditions.squeeze(0),
        "condition_validity": condition_validity.squeeze(0),
        "rayverse_total_intensity_dbm": torch.tensor(
            link.total_intensity_dbm, dtype=torch.float32
        ),
        "rayverse_total_loss_db": torch.tensor(link.total_loss_db, dtype=torch.float32),
    }


def _flush_shard(
    output_root: Path,
    split: str,
    records: list[dict[str, Any]],
    entries: list[dict[str, Any]],
    synthesizer: MIMOOFDMSynthesizer,
) -> None:
    samples = _stack_records(records)
    with torch.no_grad():
        samples["csi"] = synthesizer(
            samples["path_delays_ns"],
            samples["path_powers_linear"],
            samples["path_phases_rad"],
            samples["aoa_az_deg"],
            samples["aoa_el_deg"],
            samples["aod_az_deg"],
            samples["aod_el_deg"],
            samples["path_mask"],
        ).cpu()
    split_dir = output_root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    relative = Path(split) / f"shard_{len(entries):05d}.pt"
    torch.save({"samples": samples}, output_root / relative)
    entries.append({"file": str(relative), "count": len(records)})


def _stack_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in records[0]:
        values = [record[key] for record in records]
        result[key] = torch.stack(values) if isinstance(values[0], torch.Tensor) else values
    return result


def _split_for_link(
    link: RayVerseLink,
    split_config: dict[str, Any],
    map_split: MapSplit | None,
    seed: int,
) -> str:
    if map_split is not None:
        return map_split.split_for(link.map_id)
    return spatial_split_name(
        float(link.rx_xyz_m[0]),
        float(link.rx_xyz_m[1]),
        tile_size_m=float(split_config["spatial_tile_size_m"]),
        train_ratio=float(split_config["train_ratio"]),
        validation_ratio=float(split_config["validation_ratio"]),
        seed=seed,
    )


def _ordered_files(files: Iterable[Path], mode: str, seed: int) -> list[Path]:
    ordered = list(files)
    if mode == "sequential":
        return ordered
    rng = random.Random(seed)
    if mode == "uniform":
        rng.shuffle(ordered)
        return ordered
    if mode != "map_uniform":
        raise ValueError(f"Unsupported sampling mode {mode!r}")
    by_map: dict[int, list[Path]] = defaultdict(list)
    for path in ordered:
        by_map[ids_from_path(path)[0]].append(path)
    map_ids = list(by_map)
    rng.shuffle(map_ids)
    for group in by_map.values():
        rng.shuffle(group)
    interleaved: list[Path] = []
    while any(by_map.values()):
        for map_id in map_ids:
            if by_map[map_id]:
                interleaved.append(by_map[map_id].pop())
    return interleaved

