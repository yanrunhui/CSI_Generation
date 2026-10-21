from __future__ import annotations

import torch


def masked_sum(values: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return (values * mask.to(values.dtype)).sum(dim=dim)


def rms_delay_spread_ns(
    delays_ns: torch.Tensor,
    powers_linear: torch.Tensor,
    path_mask: torch.Tensor,
    eps: float = 1e-20,
) -> torch.Tensor:
    weights = powers_linear * path_mask.to(powers_linear.dtype)
    total = weights.sum(dim=-1).clamp_min(eps)
    mean = (weights * delays_ns).sum(dim=-1) / total
    variance = (weights * (delays_ns - mean.unsqueeze(-1)).square()).sum(dim=-1) / total
    # Preserve an exact zero for zero-spread channels, but evaluate sqrt on a
    # positive floor so the backward derivative is finite.  torch.where then
    # removes the floored branch from the zero-variance result.
    safe_spread = variance.clamp_min(eps).sqrt()
    return torch.where(variance > eps, safe_spread, torch.zeros_like(safe_spread))


def compute_physical_conditions(
    powers_linear: torch.Tensor,
    delays_ns: torch.Tensor,
    path_mask: torch.Tensor,
    direct_path_mask: torch.Tensor,
    *,
    eps: float = 1e-20,
    invalid_k_fill_db: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return conditions and validity flags in canonical four-field order.

    Order: total_power_db, delay_spread_ns, k_factor_db, los_status.
    """
    mask = path_mask.bool()
    direct = direct_path_mask.bool() & mask
    total_linear = masked_sum(powers_linear, mask).clamp_min(eps)
    total_power_db = 10.0 * torch.log10(total_linear)
    delay_spread = rms_delay_spread_ns(delays_ns, powers_linear, mask, eps=eps)

    direct_power = masked_sum(powers_linear, direct)
    remaining_power = masked_sum(powers_linear, mask & ~direct)
    los_status = direct.any(dim=-1).to(powers_linear.dtype)
    k_valid = (direct_power > eps) & (remaining_power > eps)
    raw_k_db = 10.0 * torch.log10((direct_power / remaining_power.clamp_min(eps)).clamp_min(eps))
    k_factor_db = torch.where(
        k_valid,
        raw_k_db,
        torch.full_like(raw_k_db, invalid_k_fill_db),
    )
    conditions = torch.stack(
        [total_power_db, delay_spread, k_factor_db, los_status], dim=-1
    )
    validity = torch.ones_like(conditions, dtype=torch.bool)
    validity[..., 2] = k_valid
    validity[..., 3] = mask.any(dim=-1)
    return conditions, validity


def condition_dict(conditions: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "total_power_db": conditions[..., 0],
        "delay_spread_ns": conditions[..., 1],
        "k_factor_db": conditions[..., 2],
        "los_status": conditions[..., 3],
    }

