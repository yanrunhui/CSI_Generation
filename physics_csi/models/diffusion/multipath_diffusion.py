from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from physics_csi.models.blocks import mask_from_path_count
from physics_csi.models.diffusion.condition_encoder import ConditionEncoder
from physics_csi.models.diffusion.multipath_denoiser import MultipathDenoiser
from physics_csi.models.diffusion.multipath_representation import MultipathRepresentation
from physics_csi.models.diffusion.noise_schedule import GaussianDiffusion
from physics_csi.models.diffusion.path_count_model import PathCountPredictor
from physics_csi.models.diffusion.scene_conditioning import SceneConditionFusion
from physics_csi.physics.constraints import PhysicsConstraintLayer, check_condition_feasibility
from physics_csi.physics.parameterization import PhysicsAwareParameterization
from physics_csi.physics.synthesizer import MIMOOFDMSynthesizer, SystemConfig


class MultipathDiffusionModel(nn.Module):
    condition_fields = ("total_power_db", "delay_spread_ns", "k_factor_db", "los_status")

    def __init__(
        self,
        config: Mapping[str, Any],
        system: SystemConfig,
        statistics: Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.config = dict(config)
        self.max_paths = int(config["max_paths"])
        if self.max_paths < 2:
            raise ValueError("multipath diffusion requires max_paths >= 2")
        condition_dim = int(config["condition_embedding_dim"])
        hidden_dim = int(config["hidden_dim"])
        self.condition_dropout = float(config.get("condition_dropout", 0.1))
        self.condition_encoder = ConditionEncoder(
            statistics["condition_mean"], statistics["condition_std"], condition_dim, hidden_dim
        )
        scene_config = dict(config.get("scene_conditioning", {}))
        self.scene_conditioning_enabled = bool(scene_config.get("enabled", False))
        self.map_id_to_index: dict[int, int] = {}
        self.scene_conditioner: SceneConditionFusion | None = None
        if self.scene_conditioning_enabled:
            raw_mapping = scene_config.get("map_id_to_index")
            if not isinstance(raw_mapping, Mapping) or not raw_mapping:
                raise ValueError(
                    "scene-conditioned diffusion requires a non-empty map_id_to_index mapping"
                )
            self.map_id_to_index = {int(key): int(value) for key, value in raw_mapping.items()}
            indices = sorted(self.map_id_to_index.values())
            if indices != list(range(len(indices))) or len(set(indices)) != len(indices):
                raise ValueError("map_id_to_index values must be unique contiguous indices")
            self.scene_conditioner = SceneConditionFusion(
                condition_dim,
                len(self.map_id_to_index),
                int(scene_config.get("embedding_dim", 32)),
                int(scene_config.get("fusion_hidden_dim", hidden_dim)),
                float(scene_config.get("dropout", config.get("dropout", 0.0))),
            )
        self.path_count_model = PathCountPredictor(
            condition_dim, hidden_dim, self.max_paths, float(config.get("dropout", 0.0))
        )
        parameterization_config = dict(config.get("parameterization", {}))
        delay_mode = (
            "rms_normalized_shape"
            if bool(parameterization_config.get("structured_delay", False))
            or parameterization_config.get("mode") == "structured"
            else "legacy_ns"
        )
        self.representation = MultipathRepresentation(
            statistics["multipath"], self.max_paths, delay_mode=delay_mode
        )
        self.denoiser = MultipathDenoiser(
            self.representation.feature_dim,
            condition_dim,
            hidden_dim,
            self.max_paths,
            int(config["layers"]),
            int(config["attention_heads"]),
            float(config.get("dropout", 0.0)),
            int(config["diffusion_steps"]),
        )
        self.diffusion = GaussianDiffusion(
            int(config["diffusion_steps"]), float(config.get("cosine_offset", 0.008))
        )
        self.constraints = PhysicsConstraintLayer(**dict(config["constraints"]))
        self.parameterization = PhysicsAwareParameterization(
            self.constraints, parameterization_config
        )
        self.synthesizer = MIMOOFDMSynthesizer(system)
        count_config = dict(config.get("path_count", {}))
        counts = torch.as_tensor(
            statistics.get("path_count_histogram", [0] + [1] * self.max_paths),
            dtype=torch.float32,
        )
        if counts.numel() == self.max_paths + 1:
            counts = counts[1:]
        if counts.numel() != self.max_paths:
            counts = torch.ones(self.max_paths, dtype=torch.float32)
        power = float(count_config.get("class_weight_power", 0.0))
        weights = counts.sum().clamp_min(1.0) / counts.clamp_min(1.0)
        weights = weights.pow(power)
        maximum_ratio = float(count_config.get("max_weight_ratio", 4.0))
        if maximum_ratio > 0:
            weights = weights.clamp(max=weights.min() * maximum_ratio)
        expected_weight = (weights * counts).sum() / counts.sum().clamp_min(1.0)
        weights = weights / expected_weight.clamp_min(1e-12)
        self.register_buffer("path_count_class_weights", weights, persistent=False)
        self.path_count_prior_correction = bool(
            count_config.get("prior_correction", power > 0.0)
        )
        self.path_count_label_smoothing = float(count_config.get("label_smoothing", 0.0))

    def _path_count_logit_bias(self) -> torch.Tensor | None:
        if not self.path_count_prior_correction:
            return None
        # Weighted CE learns q(L|c) proportional to w_L p(L|c).  Removing
        # log(w_L) at inference recovers the original-distribution posterior
        # before optional temperature calibration.
        return -torch.log(self.path_count_class_weights.clamp_min(1e-12))

    def map_indices(
        self,
        map_id: int | torch.Tensor | None,
        batch_size: int,
        device: torch.device,
        *,
        map_index: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Resolve external RayVerse Map IDs without treating IDs as numeric features."""
        if not self.scene_conditioning_enabled:
            return None
        if map_index is not None:
            resolved = map_index.to(device=device, dtype=torch.long).reshape(-1)
            if resolved.shape[0] == 1:
                resolved = resolved.expand(batch_size)
            if resolved.shape != (batch_size,):
                raise ValueError("map_index must be scalar or have shape [batch]")
            assert self.scene_conditioner is not None
            if bool(((resolved < 0) | (resolved >= self.scene_conditioner.num_scenes)).any()):
                raise ValueError("map_index is outside the learned embedding table")
            if map_id is not None:
                external = torch.as_tensor(map_id, dtype=torch.long).reshape(-1)
                if external.shape[0] == 1:
                    external = external.expand(batch_size)
                if external.shape != (batch_size,):
                    raise ValueError("map_id must be scalar or have shape [batch]")
                expected: list[int] = []
                unknown: set[int] = set()
                for item in external.detach().cpu().tolist():
                    key = int(item)
                    if key not in self.map_id_to_index:
                        unknown.add(key)
                    else:
                        expected.append(self.map_id_to_index[key])
                if unknown:
                    names = ", ".join(f"map_{item}" for item in sorted(unknown))
                    raise ValueError(f"unseen Map ID(s): {names}")
                expected_tensor = torch.tensor(expected, device=device, dtype=torch.long)
                if not torch.equal(resolved, expected_tensor):
                    raise ValueError("map_id and map_index do not follow the checkpoint mapping")
            return resolved
        if map_id is None:
            raise ValueError("map_id is required by a B3-S-ID model")
        external = torch.as_tensor(map_id, dtype=torch.long).reshape(-1)
        if external.shape[0] == 1:
            external = external.expand(batch_size)
        if external.shape != (batch_size,):
            raise ValueError("map_id must be scalar or have shape [batch]")
        values: list[int] = []
        unknown: set[int] = set()
        for item in external.detach().cpu().tolist():
            key = int(item)
            if key not in self.map_id_to_index:
                unknown.add(key)
            else:
                values.append(self.map_id_to_index[key])
        if unknown:
            names = ", ".join(f"map_{item}" for item in sorted(unknown))
            raise ValueError(
                f"unseen Map ID(s): {names}; B3-S-ID has no learned embedding for them"
            )
        return torch.tensor(values, device=device, dtype=torch.long)

    def condition_embedding(
        self,
        conditions: torch.Tensor,
        validity: torch.Tensor | None = None,
        *,
        map_id: int | torch.Tensor | None = None,
        map_index: torch.Tensor | None = None,
        drop_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode the complete condition used by both count and denoising branches."""
        if not self.scene_conditioning_enabled:
            return self.condition_encoder(conditions, validity, drop_mask=drop_mask)
        physical = self.condition_encoder(conditions, validity)
        resolved = self.map_indices(
            map_id, conditions.shape[0], conditions.device, map_index=map_index
        )
        assert resolved is not None and self.scene_conditioner is not None
        return self.scene_conditioner(physical, resolved, drop_mask=drop_mask)

    def unconditional_embedding(
        self, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        if self.scene_conditioning_enabled:
            assert self.scene_conditioner is not None
            return self.scene_conditioner.unconditional(batch_size, device)
        return self.condition_encoder.unconditional(batch_size, device)

    def scene_gradient_norm(self) -> torch.Tensor | None:
        """Return an L2 norm used to detect a disconnected/ignored scene branch."""
        if self.scene_conditioner is None:
            return None
        squares = [
            parameter.grad.detach().float().square().sum()
            for parameter in self.scene_conditioner.parameters()
            if parameter.grad is not None
        ]
        if not squares:
            return torch.zeros((), device=next(self.parameters()).device)
        return torch.stack(squares).sum().sqrt()

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        conditions = batch["conditions"]
        validity = batch.get("condition_validity")
        map_id = batch.get("map_id")
        map_index = batch.get("map_index")
        full_condition = self.condition_embedding(
            conditions, validity, map_id=map_id, map_index=map_index
        )
        count_logits = self.path_count_model(full_condition)
        path_count = batch["path_count"].long().clamp(1, self.max_paths)
        count_loss = F.cross_entropy(
            count_logits,
            path_count - 1,
            weight=self.path_count_class_weights.to(count_logits),
            label_smoothing=self.path_count_label_smoothing,
        )
        corrected_logits = count_logits
        bias = self._path_count_logit_bias()
        if bias is not None:
            corrected_logits = corrected_logits + bias.to(corrected_logits)
        features, feature_mask = self.representation.encode(batch)
        drop_mask = torch.zeros(conditions.shape[0], device=conditions.device, dtype=torch.bool)
        if self.training and self.condition_dropout > 0.0:
            drop_mask = (
                torch.rand(conditions.shape[0], device=conditions.device)
                < self.condition_dropout
            )
        train_condition = self.condition_embedding(
            conditions,
            validity,
            map_id=map_id,
            map_index=map_index,
            drop_mask=drop_mask,
        )

        def predict(noisy: torch.Tensor, timestep: torch.Tensor, unconditional: bool) -> torch.Tensor:
            embedding = (
                self.unconditional_embedding(noisy.shape[0], noisy.device)
                if unconditional
                else train_condition
            )
            return self.denoiser(noisy, timestep, embedding, path_count, batch["path_mask"])

        diffusion_loss = self.diffusion.training_loss(
            features, predict, element_mask=feature_mask
        )
        weight = float(self.config.get("path_count_loss_weight", 1.0))
        return {
            "loss": diffusion_loss + weight * count_loss,
            "diffusion_loss": diffusion_loss,
            "path_count_loss": count_loss,
            "path_count_accuracy": (
                corrected_logits.argmax(-1) == path_count - 1
            ).float().mean(),
            "path_count_accuracy_weighted_logits": (
                count_logits.argmax(-1) == path_count - 1
            ).float().mean(),
        }

    @torch.no_grad()
    def generate(
        self,
        conditions: Mapping[str, float] | torch.Tensor,
        num_samples: int,
        *,
        condition_validity: torch.Tensor | None = None,
        sampler: str = "ddim",
        sampling_steps: int = 50,
        eta: float = 0.0,
        guidance_scale: float = 1.0,
        path_count_temperature: float = 1.0,
        forced_path_count: int | torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        map_id: int | torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        condition_tensor, validity = self._prepare_conditions(
            conditions, num_samples, condition_validity
        )
        resolved_map_index = self.map_indices(
            map_id, num_samples, condition_tensor.device
        )
        embedding = self.condition_embedding(
            condition_tensor,
            validity,
            map_id=map_id,
            map_index=resolved_map_index,
        )
        count_probabilities = self.path_count_model.probabilities(
            embedding,
            temperature=path_count_temperature,
            logit_bias=self._path_count_logit_bias(),
        )
        sampled_count = torch.multinomial(
            count_probabilities, 1, generator=generator
        ).squeeze(-1) + 1
        if forced_path_count is not None:
            forced = torch.as_tensor(forced_path_count, device=condition_tensor.device).long()
            if forced.ndim == 0:
                forced = forced.expand(num_samples)
            if forced.shape != (num_samples,):
                raise ValueError("forced path count must be scalar or have shape [num_samples]")
            if bool(((forced < 1) | (forced > self.max_paths)).any()):
                raise ValueError(f"forced path count must be between 1 and {self.max_paths}")
            sampled_count = forced
        requires_two_paths = (condition_tensor[:, 1] > 0.0) | validity[:, 2]
        if forced_path_count is not None and bool(
            ((sampled_count < 2) & requires_two_paths).any()
        ):
            raise ValueError(
                "forced path count 1 is infeasible for positive delay spread or valid K-factor"
            )
        count = torch.where(requires_two_paths, sampled_count.clamp_min(2), sampled_count)
        path_mask = mask_from_path_count(count, self.max_paths)
        feature_mask = self.representation.feature_mask(path_mask)

        def predict(noisy: torch.Tensor, timestep: torch.Tensor, unconditional: bool) -> torch.Tensor:
            selected = (
                self.unconditional_embedding(num_samples, noisy.device)
                if unconditional
                else embedding
            )
            return self.denoiser(noisy, timestep, selected, count, path_mask)

        features = self.diffusion.sample(
            (num_samples, self.max_paths, self.representation.feature_dim),
            predict,
            device=condition_tensor.device,
            sampler=sampler,
            sampling_steps=sampling_steps,
            eta=eta,
            guidance_scale=guidance_scale,
            clip_denoised=float(self.config.get("clip_denoised", 5.0)),
            element_mask=feature_mask,
            generator=generator,
        )
        raw = self.representation.decode(features, path_mask)
        if "path_delays_ns" in raw:
            active_min = torch.where(
                raw["path_mask"],
                raw["path_delays_ns"],
                torch.full_like(raw["path_delays_ns"], torch.inf),
            ).amin(-1)
            active_max = torch.where(
                raw["path_mask"],
                raw["path_delays_ns"],
                torch.full_like(raw["path_delays_ns"], -torch.inf),
            ).amax(-1)
            repair_delay = requires_two_paths & ((active_max - active_min) <= 1e-6)
            raw["path_delays_ns"][repair_delay, 1] = active_min[repair_delay] + 1.0
        else:
            repair_delay = torch.zeros(num_samples, device=condition_tensor.device, dtype=torch.bool)
        parameterized = self.parameterization(raw, condition_tensor, validity)
        constrained = parameterized.constrained
        csi = self.synthesizer(
            constrained.delays_ns,
            constrained.powers_linear,
            raw["path_phases_rad"],
            raw["aoa_az_deg"],
            raw["aoa_el_deg"],
            raw["aod_az_deg"],
            raw["aod_el_deg"],
            raw["path_mask"],
        )
        if not torch.isfinite(csi.real).all() or not torch.isfinite(csi.imag).all():
            raise FloatingPointError("physics synthesis produced non-finite CSI")
        raw_power_fraction = raw["positive_power_weights"] / raw["positive_power_weights"].sum(
            -1, keepdim=True
        ).clamp_min(1e-20)
        constrained_power_fraction = constrained.powers_linear / constrained.powers_linear.sum(
            -1, keepdim=True
        ).clamp_min(1e-20)
        parameterized_power_fraction = (
            parameterized.parameterized_powers_linear
            / parameterized.parameterized_powers_linear.sum(-1, keepdim=True).clamp_min(1e-20)
        )
        result = {
            "csi": csi,
            "path_count": count,
            "sampled_path_count": sampled_count,
            "path_count_probabilities": count_probabilities,
            "path_count_was_forced": torch.full(
                (num_samples,), forced_path_count is not None, device=count.device, dtype=torch.bool
            ),
            "delay_feasibility_repaired": repair_delay,
            "delay_shape_repaired": parameterized.delay_shape_repaired,
            "path_mask": raw["path_mask"],
            "path_delays_ns": constrained.delays_ns,
            "path_powers_linear": constrained.powers_linear,
            "path_phases_rad": raw["path_phases_rad"],
            "aoa_az_deg": raw["aoa_az_deg"],
            "aoa_el_deg": raw["aoa_el_deg"],
            "aod_az_deg": raw["aod_az_deg"],
            "aod_el_deg": raw["aod_el_deg"],
            "direct_path_mask": constrained.direct_path_mask,
            "target_conditions": condition_tensor,
            "target_condition_validity": validity,
            "actual_conditions": constrained.actual_conditions,
            "actual_condition_validity": constrained.condition_validity,
            "constraint_feasible": constrained.feasible,
            "parameterized_path_delays_ns": parameterized.parameterized_delays_ns,
            "parameterized_path_powers_linear": parameterized.parameterized_powers_linear,
            "delay_shape_rms": parameterized.delay_shape_rms,
            "delay_scale_ns": parameterized.delay_scale_ns,
            "raw_power_weights": raw["positive_power_weights"],
            "constraint_delay_change_ns": (
                constrained.delays_ns - parameterized.parameterized_delays_ns
            ).abs().sum(-1),
            "constraint_power_fraction_l1": (
                constrained_power_fraction - parameterized_power_fraction
            ).abs().sum(-1),
            "parameterization_power_fraction_l1": (
                parameterized_power_fraction - raw_power_fraction
            ).abs().sum(-1),
        }
        if "path_delays_ns" in raw:
            result["raw_path_delays_ns"] = raw["path_delays_ns"]
        if "relative_delay_shape" in raw:
            result["raw_relative_delay_shape"] = raw["relative_delay_shape"]
        if self.scene_conditioning_enabled:
            assert resolved_map_index is not None
            result["map_index"] = resolved_map_index
            result["map_id"] = torch.as_tensor(
                map_id, device=condition_tensor.device, dtype=torch.long
            ).reshape(-1)
            if result["map_id"].shape[0] == 1:
                result["map_id"] = result["map_id"].expand(num_samples)
        return result

    def _prepare_conditions(
        self,
        conditions: Mapping[str, float] | torch.Tensor,
        num_samples: int,
        validity: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        device = next(self.parameters()).device
        if isinstance(conditions, Mapping):
            los = float(conditions["los_status"])
            check_condition_feasibility(conditions, k_factor_valid=los >= 0.5)
            row = torch.tensor([float(conditions[name]) for name in self.condition_fields], device=device)
            values = row.unsqueeze(0).expand(num_samples, -1)
            if validity is None:
                validity = torch.ones_like(values, dtype=torch.bool)
                validity[:, 2] = los >= 0.5
        else:
            values = conditions.to(device=device, dtype=torch.float32)
            if values.ndim == 1:
                values = values.unsqueeze(0)
            if values.shape[0] == 1:
                values = values.expand(num_samples, -1)
            if values.shape != (num_samples, 4):
                raise ValueError("conditions must have shape [num_samples, 4]")
            if validity is None:
                validity = torch.ones_like(values, dtype=torch.bool)
                validity[:, 2] = values[:, 3] >= 0.5
        validity = validity.to(device=device, dtype=torch.bool)
        if validity.ndim == 1:
            validity = validity.unsqueeze(0)
        if validity.shape[0] == 1:
            validity = validity.expand(num_samples, -1)
        if validity.shape != values.shape:
            raise ValueError("condition validity must match conditions")
        return values.float(), validity
