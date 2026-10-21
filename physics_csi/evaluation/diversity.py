from __future__ import annotations

import torch

from physics_csi.evaluation.distribution import multipath_features


def diversity_metrics(
    generated: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor] | None = None,
) -> dict[str, float]:
    features = multipath_features(generated)
    pairwise = torch.pdist(features)
    metrics = {
        "pairwise_feature_distance_mean": float(pairwise.mean()) if pairwise.numel() else 0.0,
        "pairwise_feature_distance_std": float(pairwise.std()) if pairwise.numel() > 1 else 0.0,
        "feature_variance_mean": float(features.var(dim=0, unbiased=False).mean()),
    }
    if reference is not None:
        reference_features = multipath_features(reference)
        distances = torch.cdist(features, reference_features)
        metrics["nearest_reference_distance_mean"] = float(distances.amin(dim=1).mean())
        reference_self = torch.cdist(reference_features, reference_features)
        reference_self.fill_diagonal_(torch.inf)
        radius = reference_self.amin(dim=1).median()
        metrics["reference_mode_coverage"] = float(
            (distances.amin(dim=0) <= radius).float().mean()
        )
    return metrics

