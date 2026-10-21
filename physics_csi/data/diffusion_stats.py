from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from physics_csi.data.dataset import PreparedShardDataset
from physics_csi.models.diffusion.angle_delay_transform import AngleDelayTransform
from physics_csi.physics.synthesizer import SystemConfig
from physics_csi.utils.io import atomic_json_dump


class _Moments:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.total_square = 0.0

    def update(self, values: torch.Tensor) -> None:
        values = values.detach().double().reshape(-1)
        finite = values[torch.isfinite(values)]
        self.count += int(finite.numel())
        self.total += float(finite.sum())
        self.total_square += float(finite.square().sum())

    def result(self) -> tuple[float, float]:
        if self.count == 0:
            return 0.0, 1.0
        mean = self.total / self.count
        variance = max(self.total_square / self.count - mean * mean, 1e-12)
        return mean, variance**0.5


@torch.no_grad()
def compute_diffusion_statistics(
    prepared_root: str | Path,
    system: SystemConfig,
    *,
    batch_size: int = 512,
    num_workers: int = 0,
    dataset: Dataset | None = None,
    source_identity: str = "train",
) -> dict[str, Any]:
    """Compute normalization constants exclusively from the training split."""
    dataset = dataset or PreparedShardDataset(prepared_root, "train")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    condition_moments = [_Moments() for _ in range(4)]
    first_moments = _Moments()
    relative_moments = _Moments()
    relative_shape_moments = _Moments()
    power_moments = _Moments()
    angle_delay_moments = _Moments()
    transform = AngleDelayTransform(system, scale=1.0)
    path_count_histogram = torch.zeros(0, dtype=torch.int64)
    for batch in loader:
        mask = batch["path_mask"].bool()
        validity = batch["condition_validity"].bool()
        for index, moments in enumerate(condition_moments):
            moments.update(batch["conditions"][:, index][validity[:, index]])
        delays = batch["path_delays_ns"].float()
        first = torch.where(mask, delays, torch.full_like(delays, torch.inf)).amin(-1)
        relative = (delays - first.unsqueeze(-1)).clamp_min(0.0)
        powers = batch["path_powers_linear"].float() * mask
        fractions = powers / powers.sum(-1, keepdim=True).clamp_min(1e-20)
        weighted_mean = (fractions * relative).sum(-1, keepdim=True)
        weighted_spread = (
            fractions * (relative - weighted_mean).square()
        ).sum(-1, keepdim=True).clamp_min(0.0).sqrt()
        relative_shape = torch.where(
            weighted_spread > 1e-8,
            relative / weighted_spread.clamp_min(1e-8),
            relative,
        )
        first_moments.update(torch.log1p(first))
        relative_moments.update(torch.log1p(relative[mask]))
        relative_shape_moments.update(torch.log1p(relative_shape[mask]))
        power_moments.update(10.0 * torch.log10(fractions[mask].clamp_min(1e-20)))
        angle_delay_moments.update(transform(batch["csi"]).reshape(-1))
        counts = batch["path_count"].long()
        required_size = int(counts.max()) + 1
        if path_count_histogram.numel() < required_size:
            path_count_histogram = torch.nn.functional.pad(
                path_count_histogram, (0, required_size - path_count_histogram.numel())
            )
        path_count_histogram += torch.bincount(
            counts, minlength=path_count_histogram.numel()
        )
    condition = [moments.result() for moments in condition_moments]
    first = first_moments.result()
    relative = relative_moments.result()
    relative_shape = relative_shape_moments.result()
    power = power_moments.result()
    _, angle_delay_std = angle_delay_moments.result()
    return {
        "format_version": 2,
        "source_split": "train",
        "source_identity": source_identity,
        "sample_count": len(dataset),
        "condition_mean": [item[0] for item in condition],
        "condition_std": [item[1] for item in condition],
        "multipath": {
            "first_delay_log_ns_mean": first[0],
            "first_delay_log_ns_std": first[1],
            "relative_delay_log_ns_mean": relative[0],
            "relative_delay_log_ns_std": relative[1],
            "relative_delay_shape_log_mean": relative_shape[0],
            "relative_delay_shape_log_std": relative_shape[1],
            "log_power_fraction_db_mean": power[0],
            "log_power_fraction_db_std": power[1],
        },
        "path_count_histogram": path_count_histogram.tolist(),
        "angle_delay_scale": angle_delay_std,
    }


def load_or_compute_diffusion_statistics(
    path: str | Path,
    prepared_root: str | Path,
    system: SystemConfig,
    *,
    batch_size: int = 512,
    num_workers: int = 0,
    dataset: Dataset | None = None,
    source_identity: str = "train",
) -> dict[str, Any]:
    path = Path(path)
    if path.exists():
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if payload.get("source_split") != "train":
            raise ValueError("diffusion statistics must come from the training split")
        cached_identity = payload.get("source_identity", "train")
        if cached_identity != source_identity:
            raise ValueError(
                f"statistics cache belongs to {cached_identity!r}, expected {source_identity!r}"
            )
        if dataset is not None and int(payload.get("sample_count", -1)) != len(dataset):
            raise ValueError("statistics cache sample count does not match experiment train view")
        return payload
    payload = compute_diffusion_statistics(
        prepared_root,
        system,
        batch_size=batch_size,
        num_workers=num_workers,
        dataset=dataset,
        source_identity=source_identity,
    )
    atomic_json_dump(payload, path)
    return payload
