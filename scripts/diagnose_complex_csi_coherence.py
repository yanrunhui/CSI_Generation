#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from diagnose_scene_joint_decomposition import (
    PATH_KEYS,
    _canonicalize_path_order,
    _standardized_mmd,
    _standardized_swd,
    _subset,
    _synthesize,
)
from physics_csi.config import load_yaml
from physics_csi.evaluation.diffusion_comparison import compare_csi_distributions
from physics_csi.evaluation.distribution import rbf_mmd, sliced_wasserstein
from physics_csi.evaluation.research_diagnostics import direction_vectors
from physics_csi.physics.synthesizer import MIMOOFDMSynthesizer, SystemConfig
from physics_csi.training.trainer import resolve_device
from physics_csi.utils.io import atomic_json_dump


def _normalized_power(payload: Mapping[str, torch.Tensor]) -> torch.Tensor:
    mask = payload["path_mask"].bool()
    power = payload["path_powers_linear"].float() * mask
    return power / power.sum(-1, keepdim=True).clamp_min(1e-20)


def _relative_delay(payload: Mapping[str, torch.Tensor]) -> torch.Tensor:
    mask = payload["path_mask"].bool()
    delay = payload["path_delays_ns"].float()
    first = torch.where(mask, delay, torch.full_like(delay, torch.inf)).amin(-1)
    return (delay - first.unsqueeze(-1)).clamp_min(0.0)


def _joint_points(payload: Mapping[str, torch.Tensor], kind: str) -> torch.Tensor:
    mask = payload["path_mask"].bool()
    phase = payload["path_phases_rad"].float()
    phase_features = torch.stack([phase.sin(), phase.cos()], dim=-1)
    aod = direction_vectors(payload["aod_az_deg"], payload["aod_el_deg"])
    aoa = direction_vectors(payload["aoa_az_deg"], payload["aoa_el_deg"])
    delay = _relative_delay(payload).unsqueeze(-1)
    power_db = (
        10.0 * torch.log10(_normalized_power(payload).clamp_min(1e-20))
    ).unsqueeze(-1)
    if kind == "phase_delay":
        values = torch.cat([delay, phase_features], dim=-1)
    elif kind == "phase_aod":
        values = torch.cat([phase_features, aod], dim=-1)
    elif kind == "phase_delay_aod":
        values = torch.cat([delay, power_db, phase_features, aod], dim=-1)
    elif kind == "aoa_aod":
        values = torch.cat([aoa, aod], dim=-1)
    else:
        raise ValueError(kind)
    return values[mask]


def _frequency_complex_features(csi: torch.Tensor) -> torch.Tensor:
    energy = csi.abs().square().mean(dim=(1, 2, 3)).clamp_min(1e-20)
    values: list[torch.Tensor] = []
    for lag in (1, 2, 4, 8, 16):
        if lag >= csi.shape[-1]:
            continue
        correlation = (
            csi[..., :-lag].conj() * csi[..., lag:]
        ).mean(dim=(1, 2, 3)) / energy
        values.extend([correlation.real, correlation.imag])
    return torch.stack(values, dim=-1).float()


def _tx_complex_features(csi: torch.Tensor) -> torch.Tensor:
    tx = csi.permute(0, 1, 3, 2)
    energy = tx.abs().square().mean(dim=(1, 2, 3)).clamp_min(1e-20)
    values: list[torch.Tensor] = []
    for lag in (1, 2, 4, 8, 16):
        if lag >= tx.shape[-1]:
            continue
        correlation = (
            tx[..., :-lag].conj() * tx[..., lag:]
        ).mean(dim=(1, 2, 3)) / energy
        values.extend([correlation.real, correlation.imag])
    return torch.stack(values, dim=-1).float()


def _coherence_metrics(
    candidate: Mapping[str, torch.Tensor], reference: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    joint: dict[str, Any] = {}
    for kind in ("phase_delay", "phase_aod", "phase_delay_aod", "aoa_aod"):
        distance, audit = _standardized_swd(
            _joint_points(candidate, kind), _joint_points(reference, kind)
        )
        joint[f"{kind}_sliced_wasserstein"] = distance
        joint[f"{kind}_normalization"] = audit
    frequency_candidate = _frequency_complex_features(candidate["csi"])
    frequency_reference = _frequency_complex_features(reference["csi"])
    frequency_swd, frequency_audit = _standardized_swd(
        frequency_candidate, frequency_reference
    )
    frequency_mmd, _ = _standardized_mmd(
        frequency_candidate, frequency_reference
    )
    tx_candidate = _tx_complex_features(candidate["csi"])
    tx_reference = _tx_complex_features(reference["csi"])
    tx_swd, tx_audit = _standardized_swd(tx_candidate, tx_reference)
    tx_mmd, _ = _standardized_mmd(tx_candidate, tx_reference)
    return {
        "joint_path_associations": joint,
        "frequency_complex_correlation": {
            "sliced_wasserstein": frequency_swd,
            "mmd_rbf": frequency_mmd,
            "normalization": frequency_audit,
        },
        "tx_complex_correlation": {
            "sliced_wasserstein": tx_swd,
            "mmd_rbf": tx_mmd,
            "normalization": tx_audit,
        },
        "standard_csi_metrics": compare_csi_distributions(
            candidate["csi"], reference["csi"]
        ),
    }


class ProjectionGrid:
    def __init__(
        self,
        reference_csi: torch.Tensor,
        dimensions: tuple[int, ...],
        seeds: tuple[int, ...],
    ) -> None:
        self.dimensions = dimensions
        self.seeds = seeds
        self.device = reference_csi.device
        reference = self._features(reference_csi)
        self.mean = reference.mean(0, keepdim=True)
        raw_scale = reference.std(0, unbiased=False, keepdim=True)
        positive = raw_scale[raw_scale > 0]
        typical = positive.median() if positive.numel() else torch.ones((), device=self.device)
        self.floor = torch.maximum(
            typical * 0.05, torch.tensor(1e-6, device=self.device)
        )
        self.scale = raw_scale.clamp_min(self.floor)
        self.reference = (reference - self.mean) / self.scale
        maximum = max(dimensions)
        self.projections: dict[int, torch.Tensor] = {}
        self.reference_projected: dict[int, torch.Tensor] = {}
        for seed in seeds:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            projection = torch.randint(
                0,
                2,
                (self.reference.shape[1], maximum),
                generator=generator,
                device=self.device,
                dtype=torch.int8,
            ).float()
            projection = (projection * 2.0 - 1.0) / math.sqrt(
                self.reference.shape[1]
            )
            self.projections[seed] = projection
            self.reference_projected[seed] = self.reference @ projection

    @staticmethod
    def _features(csi: torch.Tensor) -> torch.Tensor:
        flat = csi.flatten(1)
        flat = flat / flat.abs().square().mean(-1, keepdim=True).sqrt().clamp_min(1e-20)
        return torch.cat([flat.real, flat.imag], dim=-1).float()

    @torch.inference_mode()
    def evaluate(self, csi: torch.Tensor) -> dict[str, Any]:
        values = (self._features(csi) - self.mean) / self.scale
        per_dimension: dict[str, Any] = {}
        for dimension in self.dimensions:
            runs = []
            for seed in self.seeds:
                candidate = values @ self.projections[seed][:, :dimension]
                reference = self.reference_projected[seed][:, :dimension]
                runs.append(
                    {
                        "seed": seed,
                        "mmd_rbf": float(
                            rbf_mmd(candidate, reference, [2.0, 4.0, 8.0, 16.0])
                        ),
                        "sliced_wasserstein": float(
                            sliced_wasserstein(candidate, reference)
                        ),
                    }
                )
            per_dimension[str(dimension)] = {
                "runs": runs,
                "mmd_mean": sum(run["mmd_rbf"] for run in runs) / len(runs),
                "mmd_min": min(run["mmd_rbf"] for run in runs),
                "mmd_max": max(run["mmd_rbf"] for run in runs),
                "swd_mean": sum(run["sliced_wasserstein"] for run in runs) / len(runs),
                "swd_min": min(run["sliced_wasserstein"] for run in runs),
                "swd_max": max(run["sliced_wasserstein"] for run in runs),
            }
        return {
            "dimensions": per_dimension,
            "normalization_scale_floor": float(self.floor),
            "seeds": list(self.seeds),
        }


def _random_orders(mask: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    scores = torch.rand(mask.shape, generator=generator, device=mask.device)
    scores = torch.where(mask, scores, torch.full_like(scores, 2.0))
    return torch.argsort(scores, dim=-1)


def _permute_keys(
    payload: Mapping[str, Any], keys: tuple[str, ...], order: torch.Tensor
) -> dict[str, Any]:
    result: dict[str, Any] = dict(payload)
    maximum_paths = payload["path_mask"].shape[1]
    for key in keys:
        value = payload[key]
        if value.ndim < 2 or value.shape[1] != maximum_paths:
            raise ValueError(f"{key} is not path-aligned")
        gather = order
        while gather.ndim < value.ndim:
            gather = gather.unsqueeze(-1)
        result[key] = value.gather(1, gather.expand_as(value))
    return result


def _perturbations(
    reference: Mapping[str, Any], *, seed: int
) -> dict[str, dict[str, Any]]:
    generator = torch.Generator(device=reference["path_mask"].device).manual_seed(seed)
    order = _random_orders(reference["path_mask"].bool(), generator)
    all_parameters = tuple(key for key in PATH_KEYS if key != "path_mask")
    phase_angles = (
        "path_phases_rad",
        "aoa_az_deg",
        "aoa_el_deg",
        "aod_az_deg",
        "aod_el_deg",
    )
    delay_power_phase_aoa = (
        "path_delays_ns",
        "path_powers_linear",
        "path_phases_rad",
        "aoa_az_deg",
        "aoa_el_deg",
    )
    tests = {
        "whole_path_permutation_invariance_control": _permute_keys(
            reference, all_parameters, order
        ),
        "break_phase_delay_preserve_phase_angles": _permute_keys(
            reference, phase_angles, order
        ),
        "break_phase_aod_preserve_delay_power_phase": _permute_keys(
            reference, delay_power_phase_aoa, order
        ),
        "break_aoa_aod_by_aoa_permutation": _permute_keys(
            reference, ("aoa_az_deg", "aoa_el_deg"), order
        ),
        "phase_within_link_permutation": _permute_keys(
            reference, ("path_phases_rad",), order
        ),
    }
    randomized = dict(reference)
    mask = reference["path_mask"].bool()
    phase = torch.zeros_like(reference["path_phases_rad"])
    phase[mask] = (
        torch.rand(int(mask.sum()), generator=generator, device=mask.device)
        * (2.0 * math.pi)
        - math.pi
    )
    randomized["path_phases_rad"] = phase
    tests["uniform_phase_randomization_strong_control"] = randomized
    return tests


def _relative_csi_error(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    numerator = (candidate - reference).abs().square().flatten(1).sum(-1)
    denominator = reference.abs().square().flatten(1).sum(-1).clamp_min(1e-20)
    return float((numerator / denominator).mean())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose phase/angle coherence and Full-CSI metric sensitivity."
    )
    parser.add_argument("--generated", required=True)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--maximum-samples", type=int, default=1024)
    parser.add_argument("--projection-dimensions", default="32,64,128")
    parser.add_argument("--projection-seeds", default="0,1,2")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    dimensions = tuple(int(value) for value in args.projection_dimensions.split(","))
    seeds = tuple(int(value) for value in args.projection_seeds.split(","))
    if not dimensions or not seeds or min(dimensions) < 1:
        parser.error("projection dimensions and seeds must be non-empty")

    device = resolve_device(args.device)
    source = torch.load(args.generated, map_location="cpu", weights_only=False)
    sample_count = source["reference"]["csi"].shape[0]
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.arange(sample_count)
    if sample_count > args.maximum_samples:
        indices = torch.randperm(sample_count, generator=generator)[
            : args.maximum_samples
        ].sort().values
    reference = _canonicalize_path_order(
        _subset(source["reference"], indices, device)
    )
    scene = _subset(source["correct_map"], indices, device)
    baseline = _subset(source["physics_only_baseline"], indices, device)
    data = load_yaml(args.data_config)["data"]
    synthesizer = MIMOOFDMSynthesizer(
        SystemConfig.from_mapping(data["system"])
    ).to(device).eval()
    reference_synthesized = dict(reference)
    reference_synthesized["csi"] = _synthesize(
        reference_synthesized, synthesizer, args.batch_size
    )
    projection = ProjectionGrid(reference["csi"], dimensions, seeds)

    model_comparison = {}
    for name, payload in (("b3", baseline), ("b3_scene_id", scene)):
        model_comparison[name] = {
            "coherence": _coherence_metrics(payload, reference),
            "full_csi_projection_grid": projection.evaluate(payload["csi"]),
        }

    tests: dict[str, Any] = {}
    for name, perturbed in _perturbations(reference, seed=args.seed).items():
        candidate = dict(perturbed)
        candidate["csi"] = _synthesize(candidate, synthesizer, args.batch_size)
        tests[name] = {
            "relative_csi_error_vs_resynthesized_real": _relative_csi_error(
                candidate["csi"], reference_synthesized["csi"]
            ),
            "coherence": _coherence_metrics(candidate, reference),
            "full_csi_projection_grid": projection.evaluate(candidate["csi"]),
        }

    report = {
        "format_version": 1,
        "diagnostic": "complex_csi_coherence_and_metric_sensitivity",
        "source": str(Path(args.generated).resolve()),
        "samples": int(indices.numel()),
        "reference_synthesis_validation": {
            "relative_csi_error_stored_vs_resynthesized": _relative_csi_error(
                reference_synthesized["csi"], reference["csi"]
            ),
            "note": (
                "A small value confirms that controlled path perturbations use the same "
                "physical synthesizer convention as the stored reference CSI."
            ),
        },
        "models_vs_real": model_comparison,
        "controlled_real_parameter_tests": tests,
        "projection_grid": {
            "dimensions": list(dimensions),
            "seeds": list(seeds),
            "interpretation": (
                "A useful Full-CSI metric should be nearly invariant to whole-path order, "
                "respond strongly to uniform phase randomization, and give stable rankings "
                "across projection dimensions and seeds."
            ),
        },
        "causal_caution": (
            "Association-breaking perturbations preserve selected marginals but can alter more "
            "than one joint relationship. They diagnose sensitivity and coherence bottlenecks; "
            "they are not independent causal effects."
        ),
    }
    atomic_json_dump(report, args.output)
    print(
        json.dumps(
            {
                "event": "complex_coherence_complete",
                "output": str(Path(args.output).resolve()),
                "samples": int(indices.numel()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
