from __future__ import annotations

import math
from typing import Any

import torch

from physics_csi.evaluation.conditions import condition_accuracy
from physics_csi.evaluation.distribution import distribution_metrics, rbf_mmd, sliced_wasserstein
from physics_csi.evaluation.diversity import diversity_metrics


def csi_summary_features(csi: torch.Tensor) -> torch.Tensor:
    """Low-dimensional channel statistics suitable for two-sample metrics."""
    if csi.ndim != 4 or not csi.is_complex():
        raise ValueError("CSI must be a complex tensor [batch, rx, tx, subcarrier]")
    energy = csi.abs().square()
    total_db = 10.0 * torch.log10(energy.mean(dim=(1, 2, 3)).clamp_min(1e-20))
    delay = torch.fft.ifft(csi, dim=-1, norm="ortho").abs().square().mean(dim=(1, 2))
    delay = delay / delay.sum(-1, keepdim=True).clamp_min(1e-20)
    bins = torch.arange(delay.shape[-1], device=delay.device, dtype=delay.dtype)
    delay_mean = (delay * bins).sum(-1)
    delay_spread = (delay * (bins - delay_mean[:, None]).square()).sum(-1).sqrt()
    frequency = csi.flatten(1, 2)
    frequency = frequency / frequency.abs().square().mean(-1, keepdim=True).sqrt().clamp_min(1e-12)
    frequency_corr = torch.stack(
        [
            (frequency[..., :-lag] * frequency[..., lag:].conj()).mean(dim=(1, 2)).abs()
            for lag in (1, 2, 4, 8)
            if lag < frequency.shape[-1]
        ],
        dim=-1,
    )
    tx = csi.permute(0, 1, 3, 2).flatten(1, 2)
    tx = tx / tx.abs().square().mean(-1, keepdim=True).sqrt().clamp_min(1e-12)
    spatial_corr = torch.stack(
        [
            (tx[..., :-lag] * tx[..., lag:].conj()).mean(dim=(1, 2)).abs()
            for lag in (1, 2, 4, 8)
            if lag < tx.shape[-1]
        ],
        dim=-1,
    )
    return torch.cat(
        [total_db[:, None], delay_mean[:, None], delay_spread[:, None], frequency_corr, spatial_corr],
        dim=-1,
    ).float()


def pairwise_diversity(csi: torch.Tensor, max_pairs: int = 2048) -> float:
    count = min(csi.shape[0] // 2, max_pairs)
    if count == 0:
        return float("nan")
    first = csi[:count].flatten(1)
    second = csi[count : count * 2].flatten(1)
    distance = (first - second).abs().square().sum(-1)
    normalizer = 0.5 * (
        first.abs().square().sum(-1) + second.abs().square().sum(-1)
    ).clamp_min(1e-20)
    return float((distance / normalizer).mean())


def compare_csi_distributions(
    generated_csi: torch.Tensor,
    reference_csi: torch.Tensor,
    *,
    mmd_bandwidths: list[float] | None = None,
) -> dict[str, float]:
    generated = csi_summary_features(generated_csi)
    reference = csi_summary_features(reference_csi)
    mean = reference.mean(0, keepdim=True)
    std = reference.std(0, unbiased=False, keepdim=True).clamp_min(1e-6)
    generated = (generated - mean) / std
    reference = (reference - mean) / std
    generated_pdp = torch.fft.ifft(
        generated_csi, dim=-1, norm="ortho"
    ).abs().square().mean(dim=(1, 2))
    reference_pdp = torch.fft.ifft(
        reference_csi, dim=-1, norm="ortho"
    ).abs().square().mean(dim=(1, 2))
    generated_pdp = generated_pdp / generated_pdp.sum(-1, keepdim=True).clamp_min(1e-20)
    reference_pdp = reference_pdp / reference_pdp.sum(-1, keepdim=True).clamp_min(1e-20)
    pdp_mean = reference_pdp.mean(0, keepdim=True)
    pdp_std = reference_pdp.std(0, unbiased=False, keepdim=True).clamp_min(1e-6)
    generated_pdp = (generated_pdp - pdp_mean) / pdp_std
    reference_pdp = (reference_pdp - pdp_mean) / pdp_std
    return {
        "csi_summary_mmd_rbf": float(
            rbf_mmd(generated, reference, mmd_bandwidths or [0.5, 1.0, 2.0, 4.0])
        ),
        "csi_summary_sliced_wasserstein": float(sliced_wasserstein(generated, reference)),
        "pdp_sliced_wasserstein": float(
            sliced_wasserstein(generated_pdp.float(), reference_pdp.float())
        ),
        "generated_pairwise_diversity": pairwise_diversity(generated_csi),
        "reference_pairwise_diversity": pairwise_diversity(reference_csi),
    }


def evaluate_generated_payload(
    model_kind: str,
    generated: dict[str, Any],
    reference: dict[str, torch.Tensor],
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "model_kind": model_kind,
        "samples": int(generated["csi"].shape[0]),
        "distribution": compare_csi_distributions(generated["csi"], reference["csi"]),
    }
    if model_kind in {
        "multipath_diffusion",
        "multipath_diffusion_scene_id",
        "multipath_diffusion_seen_map_baseline",
    }:
        report["condition_accuracy"] = condition_accuracy(
            generated["target_conditions"],
            generated["actual_conditions"],
            generated["target_condition_validity"] & generated["actual_condition_validity"],
        )
        if "constraint_feasible" in generated:
            report["constraint_feasible_rate"] = float(
                generated["constraint_feasible"].float().mean()
            )
        generated_with_conditions = dict(generated)
        generated_with_conditions["conditions"] = generated["actual_conditions"]
        report["multipath_distribution"] = distribution_metrics(
            generated_with_conditions,
            reference,
            [0.5, 1.0, 2.0, 4.0],
        )
        report["multipath_diversity"] = diversity_metrics(
            generated_with_conditions, reference
        )
        if all(
            key in generated
            for key in (
                "constraint_delay_change_ns",
                "constraint_power_fraction_l1",
                "sampled_path_count",
                "delay_feasibility_repaired",
            )
        ):
            report["constraint_changes"] = {
                "delay_change_ns_mean": float(generated["constraint_delay_change_ns"].mean()),
                "power_fraction_l1_mean": float(
                    generated["constraint_power_fraction_l1"].mean()
                ),
                "path_count_feasibility_adjustment_rate": float(
                    (generated["path_count"] != generated["sampled_path_count"]).float().mean()
                ),
                "delay_feasibility_repair_rate": float(
                    generated["delay_feasibility_repaired"].float().mean()
                ),
            }
        if "delay_shape_repaired" in generated and "constraint_changes" in report:
            report["constraint_changes"]["delay_shape_repair_rate"] = float(
                generated["delay_shape_repaired"].float().mean()
            )
        if "parameterization_power_fraction_l1" in generated and "constraint_changes" in report:
            report["constraint_changes"]["parameterization_power_fraction_l1_mean"] = float(
                generated["parameterization_power_fraction_l1"].mean()
            )
    elif model_kind == "angle_delay_diffusion":
        report["condition_accuracy"] = {
            "status": "not_path_identifiable",
            "note": (
                "K-factor and LoS are path-level labels and are not reported as exact CSI-domain "
                "condition recovery. Use a fixed external estimator for an estimator-dependent comparison."
            ),
            "csi_energy_proxy_db_mean": float(
                10.0 * torch.log10(generated["csi_energy_proxy_after"].clamp_min(1e-20)).mean()
            ),
        }
    else:
        raise ValueError(f"unsupported model kind: {model_kind}")
    for section in report.values():
        if isinstance(section, dict):
            for key, value in section.items():
                if isinstance(value, float) and not math.isfinite(value):
                    section[key] = None
    return report

