from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from physics_csi.data.dataset import PreparedShardDataset


class CombinedPreparedDataset(Dataset):
    """Read every persisted sample regardless of its original cross-map split."""

    split_names = ("train", "validation", "test")

    def __init__(self, prepared_root: str | Path) -> None:
        self.datasets = [
            PreparedShardDataset(prepared_root, split) for split in self.split_names
        ]
        self.ends: list[int] = []
        total = 0
        for dataset in self.datasets:
            total += len(dataset)
            self.ends.append(total)

    def __len__(self) -> int:
        return self.ends[-1] if self.ends else 0

    def locate(self, index: int) -> tuple[str, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        for split_index, end in enumerate(self.ends):
            if index < end:
                start = 0 if split_index == 0 else self.ends[split_index - 1]
                return self.split_names[split_index], index - start
        raise IndexError(index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        split, local_index = self.locate(index)
        return self.datasets[self.split_names.index(split)][local_index]


class SceneIdDataset(Dataset):
    """A deterministic split view that attaches the learned categorical map index."""

    def __init__(
        self,
        dataset: Dataset,
        indices: Sequence[int],
        map_id_to_index: Mapping[str | int, int],
    ) -> None:
        self.dataset = dataset
        self.indices = tuple(int(index) for index in indices)
        self.map_id_to_index = {
            int(map_id): int(index) for map_id, index in map_id_to_index.items()
        }

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.dataset[self.indices[index]]
        map_id = int(record["map_id"])
        if map_id not in self.map_id_to_index:
            raise ValueError(f"map_{map_id} is absent from the experiment mapping")
        result = dict(record)
        result["map_index"] = torch.tensor(
            self.map_id_to_index[map_id], dtype=torch.int64
        )
        return result


def _allocate_groups(
    groups: list[tuple[int, int]],
    *,
    seed: int,
    validation_ratio: float,
    test_ratio: float,
) -> dict[tuple[int, int], str]:
    if len(groups) < 3:
        raise ValueError("each map needs at least three receiver tiles")
    shuffled = sorted(groups)
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, round(len(shuffled) * validation_ratio))
    test_count = max(1, round(len(shuffled) * test_ratio))
    if validation_count + test_count >= len(shuffled):
        validation_count = 1
        test_count = 1
    assigned: dict[tuple[int, int], str] = {}
    for group in shuffled[:validation_count]:
        assigned[group] = "validation"
    for group in shuffled[validation_count : validation_count + test_count]:
        assigned[group] = "test"
    for group in shuffled[validation_count + test_count :]:
        assigned[group] = "train"
    return assigned


def create_seen_map_spatial_plan(
    prepared_root: str | Path,
    *,
    name: str,
    seed: int,
    tile_size_m: float,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    guard_band_tiles: int = 0,
) -> dict[str, Any]:
    """Re-split all prepared records into spatially disjoint regions within every map."""
    ratios = (float(train_ratio), float(validation_ratio), float(test_ratio))
    if tile_size_m <= 0:
        raise ValueError("tile_size_m must be positive")
    if any(value <= 0 for value in ratios) or not math.isclose(sum(ratios), 1.0):
        raise ValueError("positive train/validation/test ratios must sum to one")
    if guard_band_tiles < 0:
        raise ValueError("guard_band_tiles cannot be negative")

    root = Path(prepared_root)
    dataset = CombinedPreparedDataset(root)
    group_indices: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    map_by_index: list[int] = []
    source_by_index: list[int] = []
    path_count_by_index: list[int] = []
    for global_index in range(len(dataset)):
        record = dataset[global_index]
        map_id = int(record["map_id"])
        rx = record["rx_xyz_m"]
        tile_x = math.floor(float(rx[0]) / tile_size_m)
        tile_y = math.floor(float(rx[1]) / tile_size_m)
        group = (map_id, tile_x, tile_y)
        group_indices[group].append(global_index)
        map_by_index.append(map_id)
        source_by_index.append(int(record["source_id"]))
        path_count_by_index.append(int(record["path_count"]))
    maps = sorted({group[0] for group in group_indices})
    if not maps:
        raise ValueError("prepared dataset contains no Map IDs")

    group_assignment: dict[tuple[int, int, int], str] = {}
    dropped_guard_groups: set[tuple[int, int, int]] = set()
    for map_id in maps:
        tiles = [(x, y) for current_map, x, y in group_indices if current_map == map_id]
        assignment = _allocate_groups(
            tiles,
            seed=int(seed) ^ (map_id * 0x9E3779B1),
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
        )
        held_out = {
            tile for tile, split in assignment.items() if split in {"validation", "test"}
        }
        if guard_band_tiles:
            for tile, split in list(assignment.items()):
                if split != "train":
                    continue
                if any(
                    max(abs(tile[0] - other[0]), abs(tile[1] - other[1]))
                    <= guard_band_tiles
                    for other in held_out
                ):
                    dropped_guard_groups.add((map_id, tile[0], tile[1]))
                    del assignment[tile]
        for tile, split in assignment.items():
            group_assignment[(map_id, tile[0], tile[1])] = split

    split_indices: dict[str, list[int]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    support: dict[str, Any] = {}
    for map_id in maps:
        map_support: dict[str, Any] = {}
        for split in split_indices:
            selected_groups = [
                group
                for group, assigned in group_assignment.items()
                if group[0] == map_id and assigned == split
            ]
            selected_indices = [
                index for group in selected_groups for index in group_indices[group]
            ]
            split_indices[split].extend(selected_indices)
            path_histogram: Counter[int] = Counter(
                path_count_by_index[index] for index in selected_indices
            )
            map_support[split] = {
                "samples": len(selected_indices),
                "receiver_tiles": len(selected_groups),
                "path_count_histogram": {
                    str(key): value for key, value in sorted(path_histogram.items())
                },
            }
        if any(map_support[split]["samples"] == 0 for split in split_indices):
            raise ValueError(f"map_{map_id} is missing from at least one spatial split")
        support[str(map_id)] = map_support

    for values in split_indices.values():
        values.sort()
    all_selected = [index for values in split_indices.values() for index in values]
    if len(all_selected) != len(set(all_selected)):
        raise RuntimeError("seen-map spatial split contains overlapping samples")
    training_maps = sorted(
        {map_by_index[index] for index in split_indices["train"]}
    )
    if training_maps != maps:
        raise RuntimeError("all encoded Map IDs must occur in the training view")
    mapping = {str(map_id): index for index, map_id in enumerate(training_maps)}
    source_sets: dict[str, dict[int, set[int]]] = {
        split: defaultdict(set) for split in split_indices
    }
    for split, indices in split_indices.items():
        for index in indices:
            source_sets[split][map_by_index[index]].add(source_by_index[index])
    source_overlap = {
        str(map_id): {
            "train_validation": len(
                source_sets["train"][map_id] & source_sets["validation"][map_id]
            ),
            "train_test": len(
                source_sets["train"][map_id] & source_sets["test"][map_id]
            ),
            "note": (
                "Tx/source identity is not a model input; identical receiver tiles are kept "
                "together, while sources may span spatial regions."
            ),
        }
        for map_id in maps
    }
    payload: dict[str, Any] = {
        "format_version": 1,
        "kind": "seen_map_spatial_holdout",
        "name": name,
        "prepared_root": str(root),
        "prepared_manifest_sha256": hashlib.sha256(
            (root / "manifest.json").read_bytes()
        ).hexdigest(),
        "seed": int(seed),
        "tile_size_m": float(tile_size_m),
        "ratios": {
            "train": float(train_ratio),
            "validation": float(validation_ratio),
            "test": float(test_ratio),
        },
        "guard_band_tiles": int(guard_band_tiles),
        "guard_band_samples_dropped": sum(
            len(group_indices[group]) for group in dropped_guard_groups
        ),
        "map_id_to_index": mapping,
        "indices": split_indices,
        "support": support,
        "source_overlap_audit": source_overlap,
        "total_prepared_samples": len(dataset),
        "selected_samples": len(all_selected),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["fingerprint"] = hashlib.sha256(canonical.encode()).hexdigest()
    return payload


def load_seen_map_spatial_plan(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("format_version") != 1 or payload.get("kind") != "seen_map_spatial_holdout":
        raise ValueError("unsupported scene-ID experiment plan")
    stored_fingerprint = payload.get("fingerprint")
    unsigned = {key: value for key, value in payload.items() if key != "fingerprint"}
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":"))
    expected_fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
    if stored_fingerprint != expected_fingerprint:
        raise ValueError("scene-ID experiment plan fingerprint is invalid")
    return payload


def make_scene_id_datasets(
    prepared_root: str | Path,
    plan: Mapping[str, Any],
) -> dict[str, Dataset]:
    manifest_digest = hashlib.sha256(
        (Path(prepared_root) / "manifest.json").read_bytes()
    ).hexdigest()
    if manifest_digest != plan.get("prepared_manifest_sha256"):
        raise ValueError(
            "prepared dataset manifest does not match the scene-ID experiment plan"
        )
    source = CombinedPreparedDataset(prepared_root)
    return {
        split: SceneIdDataset(source, plan["indices"][split], plan["map_id_to_index"])
        for split in ("train", "validation", "test")
    }


def make_scene_id_dataloaders(
    datasets: Mapping[str, Dataset],
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool = True,
) -> dict[str, DataLoader]:
    return {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        )
        for split, dataset in datasets.items()
    }
