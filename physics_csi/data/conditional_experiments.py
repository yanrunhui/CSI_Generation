from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader, Dataset

from physics_csi.data.dataset import PreparedShardDataset


CONDITION_INDEX = {
    "total_power_db": 0,
    "delay_spread_ns": 1,
    "k_factor_db": 2,
    "los_status": 3,
}


class IndexedPreparedDataset(Dataset):
    """Read-only view over persisted shards using reproducible global indices."""

    def __init__(self, dataset: Dataset, indices: Sequence[int] | None) -> None:
        self.dataset = dataset
        self.indices = None if indices is None else tuple(int(index) for index in indices)

    def __len__(self) -> int:
        return len(self.dataset) if self.indices is None else len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        source_index = index if self.indices is None else self.indices[index]
        return self.dataset[source_index]


def _inside_region(record: Mapping[str, Any], region: Mapping[str, Any]) -> bool:
    field = str(region["field"])
    if field not in CONDITION_INDEX:
        raise ValueError(f"unsupported condition field {field!r}")
    field_index = CONDITION_INDEX[field]
    if not bool(record["condition_validity"][field_index]):
        return False
    value = float(record["conditions"][field_index])
    lower = float(region["lower"])
    upper = float(region["upper"])
    if not lower <= value < upper:
        return False
    los_status = region.get("los_status")
    if los_status is not None and int(record["conditions"][3] >= 0.5) != int(los_status):
        return False
    path_counts = region.get("path_counts")
    if path_counts and int(record["path_count"]) not in {int(item) for item in path_counts}:
        return False
    return True


def _partition(
    dataset: PreparedShardDataset, region: Mapping[str, Any]
) -> tuple[list[int], list[int], dict[str, int]]:
    inside: list[int] = []
    outside: list[int] = []
    histogram: Counter[int] = Counter()
    for index in range(len(dataset)):
        record = dataset[index]
        if _inside_region(record, region):
            inside.append(index)
            histogram[int(record["path_count"])] += 1
        else:
            outside.append(index)
    return inside, outside, {str(key): value for key, value in sorted(histogram.items())}


def create_condition_experiment_plan(
    prepared_root: str | Path,
    *,
    name: str,
    mode: str,
    region: Mapping[str, Any],
    seed: int,
    sparse_retain_count: int | None = None,
) -> dict[str, Any]:
    """Create matched ID/sparse/OOD training views and an untouched test region."""
    if mode not in {"id", "sparse", "ood"}:
        raise ValueError("mode must be id, sparse or ood")
    root = Path(prepared_root)
    train = PreparedShardDataset(root, "train")
    validation = PreparedShardDataset(root, "validation")
    test = PreparedShardDataset(root, "test")
    train_inside, train_outside, train_histogram = _partition(train, region)
    validation_inside, validation_outside, validation_histogram = _partition(
        validation, region
    )
    test_inside, _, test_histogram = _partition(test, region)
    if not train_inside:
        raise ValueError("condition region has no training samples")
    if not test_inside:
        raise ValueError("condition region has no independent test samples")

    retained: list[int]
    if mode == "id":
        retained = train_inside
        train_indices: list[int] | None = None
        # Use the same outside-region validation view for ID, sparse and OOD
        # so checkpoint selection is controlled across the three protocols.
        validation_indices: list[int] | None = validation_outside
    else:
        order = train_inside.copy()
        random.Random(seed).shuffle(order)
        retain_count = 0 if mode == "ood" else int(sparse_retain_count or 0)
        if retain_count < 0:
            raise ValueError("sparse retain count cannot be negative")
        retained = sorted(order[: min(retain_count, len(order))])
        train_indices = sorted(train_outside + retained)
        # Excluding the target region from validation prevents checkpoint
        # selection from leaking the deliberately sparse/held-out condition.
        validation_indices = validation_outside
    if not validation_indices:
        raise ValueError("region removal leaves no validation samples")

    payload: dict[str, Any] = {
        "format_version": 1,
        "name": name,
        "mode": mode,
        "prepared_root": str(root),
        "seed": int(seed),
        "region": dict(region),
        "sparse_retain_count": len(retained),
        "indices": {
            "train": train_indices,
            "validation": validation_indices,
            "test_reference": test_inside,
        },
        "support": {
            "train_total": len(train),
            "train_region": len(train_inside),
            "train_region_retained": len(retained),
            "validation_total": len(validation),
            "validation_region": len(validation_inside),
            "test_total": len(test),
            "test_region": len(test_inside),
            "train_region_path_count_histogram": train_histogram,
            "validation_region_path_count_histogram": validation_histogram,
            "test_region_path_count_histogram": test_histogram,
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["fingerprint"] = hashlib.sha256(canonical.encode()).hexdigest()
    return payload


def load_condition_experiment_plan(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("format_version") != 1:
        raise ValueError("unsupported condition experiment plan format")
    return payload


def make_experiment_datasets(
    prepared_root: str | Path,
    plan: Mapping[str, Any],
) -> dict[str, Dataset]:
    indices = plan["indices"]
    return {
        "train": IndexedPreparedDataset(
            PreparedShardDataset(prepared_root, "train"), indices["train"]
        ),
        "validation": IndexedPreparedDataset(
            PreparedShardDataset(prepared_root, "validation"), indices["validation"]
        ),
        "test": IndexedPreparedDataset(
            PreparedShardDataset(prepared_root, "test"), indices["test_reference"]
        ),
    }


def make_experiment_dataloaders(
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
