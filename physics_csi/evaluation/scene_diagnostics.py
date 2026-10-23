from __future__ import annotations

import math
from typing import Any

import torch

from physics_csi.evaluation.distribution import rbf_mmd, sliced_wasserstein
from physics_csi.physics.synthesizer import MIMOOFDMSynthesizer


def _normalized_path_power(payload: dict[str, torch.Tensor]) -> torch.Tensor:
    mask = payload["path_mask"].bool()
    power = payload["path_powers_linear"].float() * mask
    return power / power.sum(-1, keepdim=True).clamp_min(1e-20)


def _direction_vectors(
    azimuth_deg: torch.Tensor, elevation_deg: torch.Tensor
) -> torch.Tensor:
    azimuth = torch.deg2rad(azimuth_deg.float())
    elevation = torch.deg2rad(elevation_deg.float())
    horizontal = elevation.cos()
    return torch.stack(
        [
            horizontal * azimuth.cos(),
            horizontal * azimuth.sin(),
            elevation.sin(),
        ],
        dim=-1,
    )


def _standardized_sliced_wasserstein(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    projections: int,
) -> torch.Tensor:
    mean = reference.mean(0, keepdim=True)
    std = reference.std(0, unbiased=False, keepdim=True).clamp_min(1e-6)
    return sliced_wasserstein(
        (candidate - mean) / std,
        (reference - mean) / std,
        projections=projections,
    )


def aod_direction_samples(payload: dict[str, torch.Tensor]) -> torch.Tensor:
    """Return active-path AoD unit vectors, preserving azimuth circularity."""
    mask = payload["path_mask"].bool()
    directions = _direction_vectors(payload["aod_az_deg"], payload["aod_el_deg"])
    return directions[mask]


def _angular_cluster_features(payload: dict[str, torch.Tensor]) -> torch.Tensor:
    """Power-weighted per-link angular concentration and RMS angular spread."""
    mask = payload["path_mask"].bool()
    weight = _normalized_path_power(payload)
    features: list[torch.Tensor] = []
    for azimuth_key, elevation_key in (
        ("aod_az_deg", "aod_el_deg"),
        ("aoa_az_deg", "aoa_el_deg"),
    ):
        direction = _direction_vectors(payload[azimuth_key], payload[elevation_key])
        mean_vector = (weight.unsqueeze(-1) * direction).sum(1)
        concentration = mean_vector.norm(dim=-1).clamp(0.0, 1.0)
        mean_direction = mean_vector / concentration.unsqueeze(-1).clamp_min(1e-8)
        angular_offset = torch.acos(
            (direction * mean_direction.unsqueeze(1)).sum(-1).clamp(-1.0, 1.0)
        )
        spread_deg = torch.rad2deg(
            (weight * angular_offset.square() * mask).sum(-1).clamp_min(0.0).sqrt()
        )
        features.extend([concentration, spread_deg])
    return torch.stack(features, dim=-1)


def _safe_correlation(values: torch.Tensor) -> torch.Tensor:
    if values.shape[0] < 2:
        return torch.zeros(
            values.shape[-1],
            values.shape[-1],
            dtype=values.dtype,
            device=values.device,
        )
    standardized = (values - values.mean(0, keepdim=True)) / values.std(
        0, unbiased=False, keepdim=True
    ).clamp_min(1e-6)
    return torch.nan_to_num(torch.corrcoef(standardized.T))


def power_angle_correlation(payload: dict[str, torch.Tensor]) -> torch.Tensor:
    """Correlation of log power fraction with circular AoD/AoA directions."""
    mask = payload["path_mask"].bool()
    power_db = 10.0 * torch.log10(_normalized_path_power(payload).clamp_min(1e-20))
    aod = _direction_vectors(payload["aod_az_deg"], payload["aod_el_deg"])
    aoa = _direction_vectors(payload["aoa_az_deg"], payload["aoa_el_deg"])
    features = torch.cat([power_db.unsqueeze(-1), aod, aoa], dim=-1)
    return _safe_correlation(features[mask])


def spatial_correlation_matrix(csi: torch.Tensor) -> torch.Tensor:
    """Full normalized transmit-side spatial correlation matrix."""
    if csi.ndim != 4 or not csi.is_complex():
        raise ValueError("CSI must be complex [batch, rx, tx, subcarrier]")
    samples = csi.permute(0, 1, 3, 2).reshape(-1, csi.shape[2])
    samples = samples / samples.abs().square().mean(-1, keepdim=True).sqrt().clamp_min(
        1e-12
    )
    correlation = samples.conj().T @ samples / max(samples.shape[0], 1)
    diagonal_mean = correlation.diagonal().real.mean().clamp_min(1e-12)
    return correlation / diagonal_mean


def _spatial_distances(
    candidate: torch.Tensor, reference: torch.Tensor
) -> dict[str, float]:
    candidate_matrix = spatial_correlation_matrix(candidate)
    reference_matrix = spatial_correlation_matrix(reference)
    difference = candidate_matrix - reference_matrix
    size = reference_matrix.shape[0]
    off_diagonal = ~torch.eye(size, dtype=torch.bool, device=reference_matrix.device)
    phase_valid = off_diagonal & (reference_matrix.abs() > 1e-3)
    if phase_valid.any():
        phase_error = torch.angle(
            candidate_matrix[phase_valid] * reference_matrix[phase_valid].conj()
        ).abs().mean()
    else:
        phase_error = torch.zeros((), device=reference_matrix.device)
    return {
        "spatial_correlation_relative_frobenius": float(
            torch.linalg.matrix_norm(difference)
            / torch.linalg.matrix_norm(reference_matrix).clamp_min(1e-12)
        ),
        "spatial_correlation_magnitude_mae": float(
            (candidate_matrix.abs() - reference_matrix.abs()).abs().mean()
        ),
        "spatial_correlation_phase_mae_rad": float(phase_error),
    }


def _wrap_degrees(values: torch.Tensor) -> torch.Tensor:
    return torch.remainder(values + 180.0, 360.0) - 180.0


def _frame_aligned_payload(
    payload: dict[str, torch.Tensor],
    *,
    frame_yaw_deg: float,
    synthesizer: MIMOOFDMSynthesizer,
) -> dict[str, torch.Tensor]:
    """Rotate global azimuths into a canonical map frame and resynthesize CSI.

    This is a coordinate-sensitivity diagnostic.  It does not claim that the
    physical array was rotated: the prepared data contain positions, but no
    per-transmitter antenna yaw/pitch/roll metadata.
    """
    aligned = dict(payload)
    yaw = torch.as_tensor(
        frame_yaw_deg,
        dtype=payload["aod_az_deg"].dtype,
        device=payload["aod_az_deg"].device,
    )
    aligned["aod_az_deg"] = _wrap_degrees(payload["aod_az_deg"] - yaw)
    aligned["aoa_az_deg"] = _wrap_degrees(payload["aoa_az_deg"] - yaw)
    aligned["csi"] = synthesizer(
        aligned["path_delays_ns"],
        aligned["path_powers_linear"],
        aligned["path_phases_rad"],
        aligned["aoa_az_deg"],
        aligned["aoa_el_deg"],
        aligned["aod_az_deg"],
        aligned["aod_el_deg"],
        aligned["path_mask"],
    )
    return aligned


@torch.inference_mode()
def rotation_aligned_scene_structure_distances(
    candidate: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
    *,
    candidate_frame_yaw_deg: float,
    reference_frame_yaw_deg: float,
    synthesizer: MIMOOFDMSynthesizer,
    projections: int = 64,
) -> dict[str, float | None]:
    """Scene distances after canonicalizing each map's horizontal frame.

    The frame yaw is estimated outside this function from receiver-coordinate
    PCA.  Both AoD and AoA azimuths are rotated and CSI is resynthesized, so the
    aligned spatial-correlation metrics remain internally consistent with the
    project's channel convention.
    """
    candidate_aligned = _frame_aligned_payload(
        candidate,
        frame_yaw_deg=candidate_frame_yaw_deg,
        synthesizer=synthesizer,
    )
    reference_aligned = _frame_aligned_payload(
        reference,
        frame_yaw_deg=reference_frame_yaw_deg,
        synthesizer=synthesizer,
    )
    candidate_aod = aod_direction_samples(candidate_aligned)
    reference_aod = aod_direction_samples(reference_aligned)
    candidate_csi = _normalized_csi_features(candidate_aligned["csi"])
    reference_csi = _normalized_csi_features(reference_aligned["csi"])
    spatial = _spatial_distances(
        candidate_aligned["csi"], reference_aligned["csi"]
    )
    result = {
        "rotation_aligned_aod_direction_sliced_wasserstein": float(
            sliced_wasserstein(
                candidate_aod,
                reference_aod,
                projections=projections,
            )
        ),
        "rotation_aligned_full_csi_sliced_wasserstein": float(
            sliced_wasserstein(
                candidate_csi,
                reference_csi,
                projections=projections,
            )
        ),
        "rotation_aligned_full_csi_mmd_rbf": float(
            rbf_mmd(candidate_csi, reference_csi, [0.25, 0.5, 1.0, 2.0])
        ),
        **{f"rotation_aligned_{key}": value for key, value in spatial.items()},
    }
    return {
        key: value if value is None or math.isfinite(value) else None
        for key, value in result.items()
    }


def _normalized_csi_features(csi: torch.Tensor) -> torch.Tensor:
    flat = csi.flatten(1)
    flat = flat / flat.abs().square().sum(-1, keepdim=True).sqrt().clamp_min(1e-12)
    return torch.cat([flat.real, flat.imag], dim=-1).float()


@torch.inference_mode()
def scene_structure_distances(
    candidate: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
    *,
    projections: int = 64,
) -> dict[str, float]:
    """Distances for scene-specific propagation structure; every metric is lower-is-better."""
    candidate_aod = aod_direction_samples(candidate)
    reference_aod = aod_direction_samples(reference)
    candidate_cluster = _angular_cluster_features(candidate)
    reference_cluster = _angular_cluster_features(reference)
    candidate_correlation = power_angle_correlation(candidate)
    reference_correlation = power_angle_correlation(reference)
    candidate_csi = _normalized_csi_features(candidate["csi"])
    reference_csi = _normalized_csi_features(reference["csi"])

    result = {
        "aod_direction_sliced_wasserstein": float(
            sliced_wasserstein(
                candidate_aod,
                reference_aod,
                projections=projections,
            )
        ),
        "angular_clustering_sliced_wasserstein": float(
            _standardized_sliced_wasserstein(
                candidate_cluster,
                reference_cluster,
                projections=projections,
            )
        ),
        "power_angle_correlation_mae": float(
            (candidate_correlation[0, 1:] - reference_correlation[0, 1:])
            .abs()
            .mean()
        ),
        "full_csi_sliced_wasserstein": float(
            sliced_wasserstein(
                candidate_csi,
                reference_csi,
                projections=projections,
            )
        ),
        "full_csi_mmd_rbf": float(
            rbf_mmd(candidate_csi, reference_csi, [0.25, 0.5, 1.0, 2.0])
        ),
        **_spatial_distances(candidate["csi"], reference["csi"]),
    }
    return {
        key: value if math.isfinite(value) else None
        for key, value in result.items()
    }


def paired_map_summary(
    correct: list[float],
    comparison: list[float],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Macro-average paired Map analysis; positive improvement favors correct Map ID."""
    correct_tensor = torch.tensor(correct, dtype=torch.float64)
    comparison_tensor = torch.tensor(comparison, dtype=torch.float64)
    if correct_tensor.numel() == 0 or correct_tensor.shape != comparison_tensor.shape:
        raise ValueError("paired map metrics must be non-empty and have equal length")
    improvement = comparison_tensor - correct_tensor
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(
        improvement.numel(),
        (bootstrap_samples, improvement.numel()),
        generator=generator,
    )
    bootstrap_mean = improvement[indices].mean(-1)
    interval = torch.quantile(
        bootstrap_mean,
        torch.tensor([0.025, 0.975], dtype=bootstrap_mean.dtype),
    )
    tolerance = 1e-12
    wins = int((improvement > tolerance).sum())
    losses = int((improvement < -tolerance).sum())
    ties = int(improvement.numel() - wins - losses)
    non_ties = wins + losses
    if non_ties:
        tail_count = sum(
            math.comb(non_ties, index)
            for index in range(min(wins, losses) + 1)
        )
        sign_test_p = min(1.0, 2.0 * tail_count / (2**non_ties))
    else:
        sign_test_p = 1.0
    lower, upper = (float(interval[0]), float(interval[1]))
    if lower > 0:
        evidence = "positive"
    elif upper < 0:
        evidence = "negative"
    else:
        evidence = "inconclusive"
    comparison_mean = float(comparison_tensor.mean())
    return {
        "maps": int(improvement.numel()),
        "correct_map_mean_distance": float(correct_tensor.mean()),
        "comparison_mean_distance": comparison_mean,
        "mean_improvement_comparison_minus_correct": float(improvement.mean()),
        "relative_mean_improvement": (
            float(improvement.mean()) / comparison_mean
            if abs(comparison_mean) > 1e-20
            else None
        ),
        "median_improvement_comparison_minus_correct": float(improvement.median()),
        "bootstrap_95_percent_ci": [lower, upper],
        "correct_map_wins": wins,
        "correct_map_losses": losses,
        "ties": ties,
        "win_rate_excluding_ties": wins / non_ties if non_ties else None,
        "two_sided_sign_test_p": sign_test_p,
        "evidence": evidence,
        "note": "Positive improvement means the correct Map ID is closer to held-out real data.",
    }
