from __future__ import annotations

import bisect
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset


class PreparedShardDataset(Dataset):
    def __init__(self, prepared_root: str | Path, split: str) -> None:
        self.root = Path(prepared_root)
        with (self.root / "manifest.json").open("r", encoding="utf-8") as stream:
            self.manifest = json.load(stream)
        if split not in self.manifest["splits"]:
            raise KeyError(f"Unknown split {split!r}")
        self.entries = self.manifest["splits"][split]["shards"]
        self.ends: list[int] = []
        total = 0
        for entry in self.entries:
            total += int(entry["count"])
            self.ends.append(total)
        self._cached_index: int | None = None
        self._cached_samples: dict[str, Any] | None = None

    def __len__(self) -> int:
        return self.ends[-1] if self.ends else 0

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index = bisect.bisect_right(self.ends, index)
        start = 0 if shard_index == 0 else self.ends[shard_index - 1]
        if self._cached_index != shard_index:
            payload = torch.load(
                self.root / self.entries[shard_index]["file"],
                map_location="cpu",
                weights_only=False,
            )
            self._cached_samples = payload["samples"]
            self._cached_index = shard_index
        assert self._cached_samples is not None
        local_index = index - start
        return {key: _index_value(value, local_index) for key, value in self._cached_samples.items()}


def _index_value(value: Any, index: int) -> Any:
    if isinstance(value, torch.Tensor):
        return value[index]
    if isinstance(value, list):
        return value[index]
    if isinstance(value, dict):
        return {key: _index_value(item, index) for key, item in value.items()}
    raise TypeError(f"Unsupported sharded value type: {type(value)!r}")


def make_dataloaders(
    prepared_root: str | Path,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool = True,
) -> dict[str, DataLoader]:
    loaders: dict[str, DataLoader] = {}
    for split in ("train", "validation", "test"):
        dataset = PreparedShardDataset(prepared_root, split)
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        )
    return loaders

