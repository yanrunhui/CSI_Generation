from __future__ import annotations

from typing import Mapping

import torch


CONDITION_NAMES = (
    "total_power_db",
    "delay_spread_ns",
    "k_factor_db",
    "los_status",
)


def condition_accuracy(
    target: torch.Tensor,
    actual: torch.Tensor,
    validity: torch.Tensor,
) -> dict[str, float]:
    if target.shape != actual.shape or target.shape != validity.shape:
        raise ValueError("target, actual and validity must have identical shapes")
    metrics: dict[str, float] = {}
    for index, name in enumerate(CONDITION_NAMES[:3]):
        valid = validity[:, index].bool()
        metrics[f"{name}_mae"] = (
            float((target[valid, index] - actual[valid, index]).abs().mean())
            if valid.any()
            else float("nan")
        )
    valid_los = validity[:, 3].bool()
    metrics["los_consistency"] = (
        float(
            ((target[valid_los, 3] >= 0.5) == (actual[valid_los, 3] >= 0.5))
            .float()
            .mean()
        )
        if valid_los.any()
        else float("nan")
    )
    return metrics


def condition_tolerance_rates(
    target: torch.Tensor,
    actual: torch.Tensor,
    validity: torch.Tensor,
    tolerances: Mapping[str, float],
) -> dict[str, float]:
    rates: dict[str, float] = {}
    for index, name in enumerate(CONDITION_NAMES[:3]):
        valid = validity[:, index].bool()
        rates[f"{name}_within_tolerance"] = (
            float(
                ((target[valid, index] - actual[valid, index]).abs() <= tolerances[name])
                .float()
                .mean()
            )
            if valid.any()
            else float("nan")
        )
    return rates

