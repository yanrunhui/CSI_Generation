from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn

from physics_csi.physics.constraints import ConstraintResult, PhysicsConstraintLayer


@dataclass
class DelayPowerResidualOutput:
    """Projected output of the sample-dependent scene transport."""

    constrained: ConstraintResult
    canonical_order: torch.Tensor
    delay_log_gap_residual: torch.Tensor
    power_logit_residual: torch.Tensor
    pre_projection_delays_ns: torch.Tensor
    pre_projection_power_fractions: torch.Tensor


def _gather_paths(value: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    if value.ndim < 2 or value.shape[:2] != order.shape:
        return value
    gather = order
    for _ in range(value.ndim - 2):
        gather = gather.unsqueeze(-1)
    return value.gather(1, gather.expand_as(value))


class SceneConditionedDelayPowerResidual(nn.Module):
    """Map-conditioned transport of a frozen B3 Delay--Power sample.

    The transport is deliberately post-generative: it changes the distribution
    of each B3 sample instead of predicting a per-Map mean.  Positive ordered
    delay gaps are multiplied in log space, power fractions are changed in
    logit space, and the differentiable physics layer is part of ``forward``.
    The two output heads are zero-initialized, so the network starts at the B3
    identity before the final (normally idempotent) physical projection.
    """

    def __init__(
        self,
        *,
        max_paths: int,
        map_id_to_index: Mapping[str | int, int],
        condition_mean: list[float] | torch.Tensor,
        condition_std: list[float] | torch.Tensor,
        constraints: Mapping[str, Any],
        hidden_dim: int = 128,
        map_embedding_dim: int = 32,
        condition_hidden_dim: int = 64,
        layers: int = 2,
        attention_heads: int = 4,
        dropout: float = 0.05,
        maximum_delay_log_shift: float = 1.0,
        maximum_power_logit_shift: float = 2.0,
    ) -> None:
        super().__init__()
        self.max_paths = int(max_paths)
        if self.max_paths < 2:
            raise ValueError("Delay--Power residual requires max_paths >= 2")
        self.map_id_to_index = {
            int(map_id): int(index) for map_id, index in map_id_to_index.items()
        }
        indices = sorted(self.map_id_to_index.values())
        if indices != list(range(len(indices))) or not indices:
            raise ValueError("Map indices must be non-empty, unique and contiguous")
        if hidden_dim % attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        self.maximum_delay_log_shift = float(maximum_delay_log_shift)
        self.maximum_power_logit_shift = float(maximum_power_logit_shift)
        if self.maximum_delay_log_shift <= 0 or self.maximum_power_logit_shift <= 0:
            raise ValueError("Residual bounds must be positive")

        mean = torch.as_tensor(condition_mean, dtype=torch.float32)
        std = torch.as_tensor(condition_std, dtype=torch.float32)
        if mean.shape != (4,) or std.shape != (4,):
            raise ValueError("Condition statistics must have four canonical fields")
        self.register_buffer("condition_mean", mean)
        self.register_buffer("condition_std", std.clamp_min(1e-6))

        self.map_embedding = nn.Embedding(len(indices), map_embedding_dim)
        self.condition_encoder = nn.Sequential(
            nn.Linear(5, condition_hidden_dim),
            nn.SiLU(),
            nn.Linear(condition_hidden_dim, condition_hidden_dim),
            nn.SiLU(),
        )
        context_dim = condition_hidden_dim + map_embedding_dim
        # token = log gap, log power fraction, rank, normalized path count,
        # direct flag, active flag.
        self.token_input = nn.Linear(6 + context_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=attention_heads,
            dim_feedforward=hidden_dim * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.path_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=layers, enable_nested_tensor=False
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.delay_head = nn.Linear(hidden_dim, 1)
        self.power_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.delay_head.weight)
        nn.init.zeros_(self.delay_head.bias)
        nn.init.zeros_(self.power_head.weight)
        nn.init.zeros_(self.power_head.bias)
        self.constraints = PhysicsConstraintLayer(**dict(constraints))

    def map_indices(self, map_id: torch.Tensor | int) -> torch.Tensor:
        external = torch.as_tensor(map_id, dtype=torch.long).reshape(-1)
        values: list[int] = []
        unknown: set[int] = set()
        for item in external.detach().cpu().tolist():
            key = int(item)
            if key not in self.map_id_to_index:
                unknown.add(key)
            else:
                values.append(self.map_id_to_index[key])
        if unknown:
            names = ", ".join(f"map_{value}" for value in sorted(unknown))
            raise ValueError(f"DP residual has no learned embedding for {names}")
        return torch.tensor(values, device=external.device, dtype=torch.long)

    def external_map_ids(self, map_index: torch.Tensor) -> torch.Tensor:
        inverse = {index: map_id for map_id, index in self.map_id_to_index.items()}
        values = [inverse[int(item)] for item in map_index.detach().cpu().tolist()]
        return torch.tensor(values, device=map_index.device, dtype=torch.long)

    def _canonicalize(
        self,
        path_delays_ns: torch.Tensor,
        path_powers_linear: torch.Tensor,
        path_mask: torch.Tensor,
        direct_path_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = path_mask.bool()
        if mask.ndim != 2 or mask.shape[1] != self.max_paths:
            raise ValueError("path_mask must have shape [batch, max_paths]")
        if path_delays_ns.shape != mask.shape or path_powers_linear.shape != mask.shape:
            raise ValueError("Delay, power and mask tensors must share a shape")
        order = torch.argsort(
            torch.where(
                mask,
                path_delays_ns.float(),
                torch.full_like(path_delays_ns.float(), torch.inf),
            ),
            dim=-1,
            stable=True,
        )
        sorted_mask = mask.gather(1, order)
        sorted_delay = path_delays_ns.float().gather(1, order)
        sorted_power = path_powers_linear.float().gather(1, order)
        direct = (
            direct_path_mask.bool()
            if direct_path_mask is not None
            else torch.zeros_like(mask)
        ).gather(1, order)
        return sorted_delay, sorted_power, sorted_mask, direct, order

    def forward(
        self,
        base: Mapping[str, torch.Tensor],
        target_conditions: torch.Tensor,
        target_condition_validity: torch.Tensor,
        *,
        map_id: torch.Tensor | int | None = None,
        map_index: torch.Tensor | None = None,
    ) -> DelayPowerResidualOutput:
        conditions = target_conditions.float()
        validity = target_condition_validity.bool()
        batch_size = conditions.shape[0]
        if conditions.shape != (batch_size, 4) or validity.shape != conditions.shape:
            raise ValueError("Conditions and validity must have shape [batch, 4]")
        if map_index is None:
            if map_id is None:
                raise ValueError("map_id or map_index is required")
            map_index = self.map_indices(map_id)
        map_index = map_index.to(device=conditions.device, dtype=torch.long).reshape(-1)
        if map_index.shape != (batch_size,):
            raise ValueError("map_index must have shape [batch]")
        if bool(((map_index < 0) | (map_index >= len(self.map_id_to_index))).any()):
            raise ValueError("map_index is outside the learned embedding table")

        delay, power, mask, direct, order = self._canonicalize(
            base["path_delays_ns"],
            base["path_powers_linear"],
            base["path_mask"],
            base.get("direct_path_mask"),
        )
        if not bool(mask.any(-1).all()):
            raise ValueError("Every residual sample must contain at least one path")
        first_delay = torch.where(
            mask, delay, torch.full_like(delay, torch.inf)
        ).amin(-1)
        previous = torch.cat([first_delay.unsqueeze(-1), delay[:, :-1]], dim=-1)
        gap_mask = mask.clone()
        gap_mask[:, 0] = False
        gaps = torch.where(gap_mask, (delay - previous).clamp_min(0.0), torch.zeros_like(delay))
        total_power = (power * mask).sum(-1, keepdim=True).clamp_min(1e-20)
        fractions = power.clamp_min(1e-20) * mask / total_power

        normalized_condition = (conditions - self.condition_mean) / self.condition_std
        # Invalid K is a missing value rather than a numerical zero.  Add its
        # validity bit so NLoS and a genuine 0 dB K-factor stay distinguishable.
        normalized_condition = normalized_condition.clone()
        normalized_condition[:, 2] = torch.where(
            validity[:, 2], normalized_condition[:, 2], torch.zeros_like(normalized_condition[:, 2])
        )
        condition_input = torch.cat(
            [normalized_condition, validity[:, 2:3].float()], dim=-1
        )
        context = torch.cat(
            [self.condition_encoder(condition_input), self.map_embedding(map_index)], dim=-1
        )
        rank = torch.arange(
            self.max_paths, device=delay.device, dtype=delay.dtype
        ).unsqueeze(0).expand(batch_size, -1)
        rank = rank / max(self.max_paths - 1, 1)
        path_count = mask.sum(-1, keepdim=True).to(delay.dtype) / self.max_paths
        token = torch.stack(
            [
                torch.log1p(gaps),
                torch.log(fractions.clamp_min(1e-20)),
                rank,
                path_count.expand(-1, self.max_paths),
                direct.float(),
                mask.float(),
            ],
            dim=-1,
        )
        encoded = self.token_input(
            torch.cat([token, context.unsqueeze(1).expand(-1, self.max_paths, -1)], dim=-1)
        )
        encoded = self.path_encoder(encoded, src_key_padding_mask=~mask)
        encoded = self.output_norm(encoded)

        delay_residual = self.maximum_delay_log_shift * torch.tanh(
            self.delay_head(encoded).squeeze(-1)
        )
        power_residual = self.maximum_power_logit_shift * torch.tanh(
            self.power_head(encoded).squeeze(-1)
        )
        delay_residual = delay_residual * gap_mask
        power_residual = power_residual * mask

        corrected_gaps = gaps * torch.exp(delay_residual)
        corrected_delay = first_delay.unsqueeze(-1) + corrected_gaps.cumsum(-1)
        corrected_delay = corrected_delay * mask
        corrected_logits = torch.log(fractions.clamp_min(1e-20)) + power_residual
        corrected_logits = corrected_logits.masked_fill(~mask, -torch.inf)
        corrected_fraction = torch.softmax(corrected_logits, dim=-1) * mask

        constrained = self.constraints(
            corrected_delay,
            corrected_fraction,
            mask,
            conditions,
            target_condition_validity=validity,
            direct_path_mask=direct,
        )
        return DelayPowerResidualOutput(
            constrained=constrained,
            canonical_order=order,
            delay_log_gap_residual=delay_residual,
            power_logit_residual=power_residual,
            pre_projection_delays_ns=corrected_delay,
            pre_projection_power_fractions=corrected_fraction,
        )

    def apply_to_payload(
        self,
        base: Mapping[str, torch.Tensor],
        output: DelayPowerResidualOutput,
    ) -> dict[str, torch.Tensor]:
        """Reorder all path attributes consistently and replace Delay--Power."""
        result: dict[str, torch.Tensor] = {}
        sample_count = output.canonical_order.shape[0]
        for key, value in base.items():
            if not isinstance(value, torch.Tensor) or value.shape[:1] != (sample_count,):
                continue
            result[key] = _gather_paths(value, output.canonical_order)
        result.update(
            {
                "path_mask": output.constrained.delays_ns.ne(0.0)
                | _gather_paths(base["path_mask"].bool(), output.canonical_order),
                "path_delays_ns": output.constrained.delays_ns,
                "path_powers_linear": output.constrained.powers_linear,
                "direct_path_mask": output.constrained.direct_path_mask,
                "actual_conditions": output.constrained.actual_conditions,
                "actual_condition_validity": output.constrained.condition_validity,
                "constraint_feasible": output.constrained.feasible,
            }
        )
        # The expression above preserves a valid active path at absolute delay
        # zero; use the canonical source mask as the authoritative topology.
        result["path_mask"] = _gather_paths(
            base["path_mask"].bool(), output.canonical_order
        )
        return result
