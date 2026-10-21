from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset


class SceneConditionSupportChecker:
    """Measure empirical training support for a requested Map ID and condition."""

    fields = ("total_power_db", "delay_spread_ns", "k_factor_db", "los_status")

    def __init__(
        self,
        map_ids: torch.Tensor,
        conditions: torch.Tensor,
        validity: torch.Tensor,
        condition_std: torch.Tensor,
        *,
        supported_radius: float = 1.0,
        sparse_radius: float = 2.0,
        supported_neighbors: int = 32,
        sparse_neighbors: int = 8,
    ) -> None:
        self.map_ids = map_ids.long().cpu()
        self.conditions = conditions.float().cpu()
        self.validity = validity.bool().cpu()
        self.condition_std = condition_std.float().cpu().clamp_min(1e-6)
        self.supported_radius = float(supported_radius)
        self.sparse_radius = float(sparse_radius)
        self.supported_neighbors = int(supported_neighbors)
        self.sparse_neighbors = int(sparse_neighbors)

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        statistics: Mapping[str, Any],
        *,
        batch_size: int = 1024,
        num_workers: int = 0,
    ) -> "SceneConditionSupportChecker":
        map_chunks: list[torch.Tensor] = []
        condition_chunks: list[torch.Tensor] = []
        validity_chunks: list[torch.Tensor] = []
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        for batch in loader:
            map_chunks.append(batch["map_id"].long())
            condition_chunks.append(batch["conditions"].float())
            validity_chunks.append(batch["condition_validity"].bool())
        if not map_chunks:
            raise ValueError("training dataset is empty")
        map_ids = torch.cat(map_chunks)
        conditions = torch.cat(condition_chunks)
        validity = torch.cat(validity_chunks)
        condition_std = torch.as_tensor(statistics["condition_std"]).float()
        supported_radius, sparse_radius = cls._calibrate_radii(
            map_ids,
            conditions,
            validity,
            condition_std,
            supported_neighbors=32,
            sparse_neighbors=8,
        )
        return cls(
            map_ids,
            conditions,
            validity,
            condition_std,
            supported_radius=supported_radius,
            sparse_radius=sparse_radius,
        )

    @staticmethod
    def _calibrate_radii(
        map_ids: torch.Tensor,
        conditions: torch.Tensor,
        validity: torch.Tensor,
        condition_std: torch.Tensor,
        *,
        supported_neighbors: int,
        sparse_neighbors: int,
        maximum_group_samples: int = 256,
    ) -> tuple[float, float]:
        """Calibrate radii from within-map, exact-LoS training neighborhoods."""
        normalized = conditions.float() / condition_std.float().clamp_min(1e-6)
        normalized[:, 2] = torch.where(
            validity[:, 2].bool(), normalized[:, 2], torch.zeros_like(normalized[:, 2])
        )
        normalized = normalized[:, :3]
        supported_distances: list[torch.Tensor] = []
        sparse_distances: list[torch.Tensor] = []
        for map_id in torch.unique(map_ids.long()):
            in_map = map_ids.long() == map_id
            for los in (False, True):
                selected = torch.nonzero(
                    in_map & ((conditions[:, 3] >= 0.5) == los), as_tuple=False
                ).flatten()
                if selected.numel() <= sparse_neighbors:
                    continue
                if selected.numel() > maximum_group_samples:
                    positions = torch.linspace(
                        0, selected.numel() - 1, maximum_group_samples
                    ).round().long()
                    selected = selected[positions]
                distance = torch.cdist(normalized[selected], normalized[selected])
                distance.fill_diagonal_(torch.inf)
                ordered = distance.sort(dim=1).values
                sparse_index = min(sparse_neighbors, ordered.shape[1] - 1) - 1
                sparse_distances.append(ordered[:, sparse_index])
                if selected.numel() > supported_neighbors:
                    supported_distances.append(ordered[:, supported_neighbors - 1])
        if not sparse_distances:
            return 1.0, 2.0
        sparse = torch.cat(sparse_distances)
        supported_radius = (
            float(torch.quantile(torch.cat(supported_distances), 0.5).clamp_min(1e-3))
            if supported_distances
            else float(torch.quantile(sparse, 0.95).clamp_min(1e-3))
        )
        # The sparse region deliberately extends beyond the typical dense
        # neighborhood, while requiring fewer neighbors.
        sparse_radius = max(
            supported_radius * 1.5,
            float(torch.quantile(sparse, 0.95).clamp_min(1e-3)),
        )
        return supported_radius, sparse_radius

    def check(
        self,
        map_id: int,
        conditions: Mapping[str, float] | torch.Tensor,
        validity: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if isinstance(conditions, Mapping):
            values = torch.tensor(
                [float(conditions[field]) for field in self.fields], dtype=torch.float32
            )
            valid = torch.ones(4, dtype=torch.bool)
            valid[2] = values[3] >= 0.5
        else:
            values = torch.as_tensor(conditions, dtype=torch.float32).reshape(-1)
            if values.shape != (4,):
                raise ValueError("conditions must contain four values")
            valid = (
                torch.ones(4, dtype=torch.bool)
                if validity is None
                else torch.as_tensor(validity, dtype=torch.bool).reshape(-1)
            )
            if valid.shape != (4,):
                raise ValueError("condition validity must contain four values")
        in_map = self.map_ids == int(map_id)
        if not bool(in_map.any()):
            raise ValueError(f"map_{map_id} has no training samples")
        candidates = self.conditions[in_map]
        candidate_validity = self.validity[in_map]
        same_los = (candidates[:, 3] >= 0.5) == (values[3] >= 0.5)
        comparable = candidate_validity & valid.unsqueeze(0)
        comparable[:, 3] = False
        distance = torch.where(
            comparable,
            (candidates - values.unsqueeze(0)) / self.condition_std,
            torch.zeros_like(candidates),
        ).square().sum(-1).sqrt()
        distance = distance[same_los]
        inside_supported = int((distance <= self.supported_radius).sum())
        inside_sparse = int((distance <= self.sparse_radius).sum())
        if inside_supported >= self.supported_neighbors:
            level = "Supported"
        elif inside_sparse >= self.sparse_neighbors:
            level = "Sparse"
        else:
            level = "Out-of-Support"
        return {
            "map_id": int(map_id),
            "map_name": f"map_{int(map_id)}",
            "map_training_samples": int(in_map.sum()),
            "los_matched_samples": int(same_los.sum()),
            "nearest_standardized_condition_distance": (
                float(distance.min()) if distance.numel() else None
            ),
            "neighbors_within_supported_radius": inside_supported,
            "neighbors_within_sparse_radius": inside_sparse,
            "support_level": level,
            "thresholds": {
                "supported": {
                    "radius": self.supported_radius,
                    "minimum_neighbors": self.supported_neighbors,
                },
                "sparse": {
                    "radius": self.sparse_radius,
                    "minimum_neighbors": self.sparse_neighbors,
                },
                "distance": (
                    "Euclidean distance over valid standardized physical conditions; "
                    "LoS/NLoS must match exactly"
                ),
                "calibration": (
                    "Radii are calibrated from within-map, exact-LoS training k-nearest-"
                    "neighbor distances (median 32-neighbor dense radius; sparse radius "
                    "extends beyond it and uses the 8-neighbor distribution)."
                ),
            },
        }
