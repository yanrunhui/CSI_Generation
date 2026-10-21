from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class MapSplit:
    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]

    def split_for(self, map_id: int) -> str:
        if map_id in self.train:
            return "train"
        if map_id in self.validation:
            return "validation"
        if map_id in self.test:
            return "test"
        raise KeyError(f"Map {map_id} is not in the split manifest")

    def as_dict(self) -> dict[str, list[int]]:
        return {
            "train": list(self.train),
            "validation": list(self.validation),
            "test": list(self.test),
        }


def make_map_split(
    map_ids: Iterable[int],
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> MapSplit:
    ids = sorted(set(int(value) for value in map_ids))
    if len(ids) < 3:
        raise ValueError("At least three non-empty maps are required for a cross-map split")
    if abs(train_ratio + validation_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to one")
    random.Random(seed).shuffle(ids)
    n_validation = max(1, round(len(ids) * validation_ratio))
    n_test = max(1, round(len(ids) * test_ratio))
    n_train = len(ids) - n_validation - n_test
    if n_train < 1:
        raise ValueError("Split ratios leave no training maps")
    return MapSplit(
        train=tuple(sorted(ids[:n_train])),
        validation=tuple(sorted(ids[n_train : n_train + n_validation])),
        test=tuple(sorted(ids[n_train + n_validation :])),
    )


def spatial_split_name(
    x_m: float,
    y_m: float,
    *,
    tile_size_m: float,
    train_ratio: float,
    validation_ratio: float,
    seed: int,
) -> str:
    if tile_size_m <= 0:
        raise ValueError("tile_size_m must be positive")
    tile_x = int(x_m // tile_size_m)
    tile_y = int(y_m // tile_size_m)
    digest = hashlib.sha256(f"{seed}:{tile_x}:{tile_y}".encode()).digest()
    value = int.from_bytes(digest[:8], "little") / float(2**64)
    if value < train_ratio:
        return "train"
    if value < train_ratio + validation_ratio:
        return "validation"
    return "test"

