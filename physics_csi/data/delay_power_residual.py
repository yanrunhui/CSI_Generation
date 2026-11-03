from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterator, Mapping
from typing import Any

import torch
from torch.utils.data import Dataset, Sampler


def _index_nested(value: Any, index: int) -> Any:
    if isinstance(value, torch.Tensor):
        return value[index]
    if isinstance(value, Mapping):
        return {key: _index_nested(item, index) for key, item in value.items()}
    return value


class DelayPowerResidualCacheDataset(Dataset):
    """In-memory view of one split from a frozen-B3 residual cache."""

    def __init__(self, cache: Mapping[str, Any], split: str) -> None:
        if cache.get("format_version") != 1 or cache.get("kind") != "b3_dp_residual_cache":
            raise ValueError("unsupported B3 DP-residual cache")
        if split not in cache.get("splits", {}):
            raise KeyError(f"cache has no {split!r} split")
        self.payload = cache["splits"][split]
        self.reference = self.payload["reference"]
        self.base = self.payload["base"]
        self.conditions = self.reference["conditions"].float()
        self.validity = self.reference["condition_validity"].bool()
        self.map_ids = self.reference["map_id"].long()
        self.path_counts = self.reference["path_mask"].sum(-1).long()
        self.sample_count = int(self.conditions.shape[0])
        for name, values in (("reference", self.reference), ("base", self.base)):
            for key, value in values.items():
                if isinstance(value, torch.Tensor) and value.shape[:1] != (self.sample_count,):
                    raise ValueError(f"{name}.{key} has an inconsistent sample dimension")
        if not torch.equal(self.path_counts, self.base["path_mask"].sum(-1).long()):
            raise ValueError("cache must preserve the real path count in every B3 sample")

    def __len__(self) -> int:
        return self.sample_count

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "reference": _index_nested(self.reference, index),
            "base": _index_nested(self.base, index),
        }


class ConditionNeighborhoodBatchSampler(Sampler[list[int]]):
    """Map/path-count strata with nearest physical-condition neighborhoods.

    Each yielded batch contains one Map, LoS state, K-validity state and path
    count.  Nearest neighbors are selected using standardized Power, Delay
    Spread and (when valid) K-factor, so distribution losses never compare a
    dense mixture of unrelated user conditions.
    """

    def __init__(
        self,
        dataset: DelayPowerResidualCacheDataset,
        *,
        batch_size: int,
        batches_per_epoch: int,
        minimum_batch_size: int,
        condition_std: list[float] | torch.Tensor,
        seed: int,
        shuffle: bool,
    ) -> None:
        if batch_size < 2 or minimum_batch_size < 2:
            raise ValueError("distribution batches require at least two samples")
        if minimum_batch_size > batch_size:
            raise ValueError("minimum_batch_size cannot exceed batch_size")
        if batches_per_epoch < 1:
            raise ValueError("batches_per_epoch must be positive")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.batches_per_epoch = int(batches_per_epoch)
        self.minimum_batch_size = int(minimum_batch_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0
        scale = torch.as_tensor(condition_std, dtype=torch.float32)
        if scale.shape != (4,):
            raise ValueError("condition_std must contain four values")
        features = dataset.conditions[:, :3].clone()
        features[:, 2] = torch.where(
            dataset.validity[:, 2], features[:, 2], torch.zeros_like(features[:, 2])
        )
        self.features = features / scale[:3].clamp_min(1e-6)
        strata: dict[tuple[int, bool, bool, int], list[int]] = defaultdict(list)
        for index in range(len(dataset)):
            key = (
                int(dataset.map_ids[index]),
                bool(dataset.conditions[index, 3] >= 0.5),
                bool(dataset.validity[index, 2]),
                int(dataset.path_counts[index]),
            )
            strata[key].append(index)
        self.strata = {
            key: torch.tensor(indices, dtype=torch.long)
            for key, indices in strata.items()
            if len(indices) >= self.minimum_batch_size
        }
        if not self.strata:
            raise ValueError("no Map/condition/path-count stratum can form a distribution batch")
        self.index_to_stratum: dict[int, tuple[int, bool, bool, int]] = {}
        for key, indices in self.strata.items():
            for index in indices.tolist():
                self.index_to_stratum[index] = key
        self.eligible = torch.tensor(sorted(self.index_to_stratum), dtype=torch.long)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        if self.shuffle:
            repeats = math.ceil(self.batches_per_epoch / max(self.eligible.numel(), 1))
            anchors = torch.cat(
                [self.eligible[torch.randperm(self.eligible.numel(), generator=generator)] for _ in range(repeats)]
            )[: self.batches_per_epoch]
        else:
            positions = torch.linspace(
                0,
                max(self.eligible.numel() - 1, 0),
                self.batches_per_epoch,
            ).round().long()
            anchors = self.eligible[positions]
        for anchor_tensor in anchors:
            anchor = int(anchor_tensor)
            candidates = self.strata[self.index_to_stratum[anchor]]
            distance = (self.features[candidates] - self.features[anchor]).square().sum(-1)
            count = min(self.batch_size, int(candidates.numel()))
            nearest = torch.topk(distance, count, largest=False, sorted=True).indices
            selected = candidates[nearest]
            if self.shuffle and selected.numel() > 2:
                selected = selected[torch.randperm(selected.numel(), generator=generator)]
            yield selected.tolist()


def cache_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    def stack(values: list[Any]) -> Any:
        first = values[0]
        if isinstance(first, torch.Tensor):
            return torch.stack(values)
        if isinstance(first, Mapping):
            return {key: stack([value[key] for value in values]) for key in first}
        return values

    return stack(batch)
