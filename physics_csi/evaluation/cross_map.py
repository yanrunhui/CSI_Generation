from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from physics_csi.training.losses import complex_nmse
from physics_csi.training.trainer import move_to_device


@torch.no_grad()
def evaluate_cross_map_reconstruction(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
) -> dict[int, dict[str, float]]:
    model.eval()
    totals: dict[int, float] = defaultdict(float)
    counts: dict[int, int] = defaultdict(int)
    for batch in loader:
        batch = move_to_device(batch, device)
        output = model(batch)
        for map_id in batch["map_id"].unique():
            select = batch["map_id"] == map_id
            value = complex_nmse(output["csi"][select], batch["csi"][select])
            count = int(select.sum())
            totals[int(map_id)] += float(value) * count
            counts[int(map_id)] += count
    return {
        map_id: {"csi_nmse": totals[map_id] / counts[map_id], "samples": counts[map_id]}
        for map_id in sorted(counts)
    }

