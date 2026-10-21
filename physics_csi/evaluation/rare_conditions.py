from __future__ import annotations

import torch


def condition_density_classes(
    train_conditions: torch.Tensor,
    query_conditions: torch.Tensor,
    *,
    scales: torch.Tensor | None = None,
    neighbors: int = 20,
    quantiles: tuple[float, float] = (0.2, 0.8),
    max_density_reference: int = 10000,
    chunk_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Classify queries as common(0), medium(1), or rare(2) by kNN radius."""
    scales = scales if scales is not None else torch.tensor(
        [20.0, 100.0, 10.0, 1.0], device=train_conditions.device
    )
    train = train_conditions.float() / scales
    query = query_conditions.float() / scales
    if train.shape[0] > max_density_reference:
        indices = torch.linspace(
            0, train.shape[0] - 1, max_density_reference, device=train.device
        ).long()
        train = train[indices]
    k = min(neighbors, train.shape[0])
    radii = _knn_radius(query, train, k, chunk_size)
    train_radii = _knn_radius(train, train, min(k + 1, train.shape[0]), chunk_size)
    lower, upper = torch.quantile(
        train_radii, torch.tensor(quantiles, device=train_radii.device)
    )
    labels = torch.ones_like(radii, dtype=torch.int64)
    labels[radii <= lower] = 0
    labels[radii >= upper] = 2
    return labels, radii


def _knn_radius(
    query: torch.Tensor,
    reference: torch.Tensor,
    neighbors: int,
    chunk_size: int,
) -> torch.Tensor:
    chunks = []
    for start in range(0, query.shape[0], chunk_size):
        distance = torch.cdist(query[start : start + chunk_size], reference)
        chunks.append(distance.topk(neighbors, largest=False).values[:, -1])
    return torch.cat(chunks)

