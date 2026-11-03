from __future__ import annotations

import json
import sys

import torch

from physics_csi.data.physical_labels import compute_physical_conditions
from physics_csi.physics.synthesizer import MIMOOFDMSynthesizer, SystemConfig
from scripts import diagnose_csi_remaining_bottlenecks as diagnostic


def _payload(samples: int = 24) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(17)
    maximum_paths = 3
    mask = torch.ones(samples, maximum_paths, dtype=torch.bool)
    direct = torch.zeros_like(mask)
    direct[:, 0] = True
    delay = torch.stack(
        [
            torch.zeros(samples),
            10.0 + torch.rand(samples, generator=generator) * 5.0,
            30.0 + torch.rand(samples, generator=generator) * 10.0,
        ],
        dim=-1,
    )
    power = torch.rand(samples, maximum_paths, generator=generator) + 0.2
    phase = torch.rand(samples, maximum_paths, generator=generator) * (2.0 * torch.pi)
    aod_azimuth = torch.rand(samples, maximum_paths, generator=generator) * 240.0 - 120.0
    aod_elevation = torch.rand(samples, maximum_paths, generator=generator) * 60.0 - 30.0
    conditions, validity = compute_physical_conditions(power, delay, mask, direct)
    payload = {
        "map_id": torch.arange(2).repeat_interleave(samples // 2),
        "conditions": conditions,
        "condition_validity": validity,
        "target_conditions": conditions,
        "target_condition_validity": validity,
        "actual_conditions": conditions,
        "actual_condition_validity": validity,
        "path_mask": mask,
        "direct_path_mask": direct,
        "path_delays_ns": delay,
        "path_powers_linear": power,
        "path_phases_rad": phase,
        "aoa_az_deg": torch.zeros_like(delay),
        "aoa_el_deg": torch.zeros_like(delay),
        "aod_az_deg": aod_azimuth,
        "aod_el_deg": aod_elevation,
    }
    system = SystemConfig(
        num_subcarriers=64,
        tx_array_shape=(8, 8),
        rx_array_shape=(1, 1),
    )
    payload["csi"] = MIMOOFDMSynthesizer(system)(
        delay,
        power,
        phase,
        payload["aoa_az_deg"],
        payload["aoa_el_deg"],
        aod_azimuth,
        aod_elevation,
        mask,
    )
    return payload


def _generated(reference: dict[str, torch.Tensor], offset: float) -> dict[str, torch.Tensor]:
    payload = {key: value.clone() for key, value in reference.items()}
    payload["path_delays_ns"][:, 1:] += offset
    payload["path_phases_rad"][:, 1:] += offset * 0.1
    payload["aod_az_deg"][:, 1:] += offset
    system = SystemConfig(
        num_subcarriers=64,
        tx_array_shape=(8, 8),
        rx_array_shape=(1, 1),
    )
    payload["csi"] = MIMOOFDMSynthesizer(system)(
        payload["path_delays_ns"],
        payload["path_powers_linear"],
        payload["path_phases_rad"],
        payload["aoa_az_deg"],
        payload["aoa_el_deg"],
        payload["aod_az_deg"],
        payload["aod_el_deg"],
        payload["path_mask"],
    )
    return payload


def test_tx_spacing_groups_cover_every_off_diagonal_pair() -> None:
    groups = diagnostic._tx_spacing_groups(SystemConfig(tx_array_shape=(8, 8)), torch.device("cpu"))
    assert sum(group["pairs"] for group in groups.values()) == 64 * 63 // 2
    assert min(group["distance_wavelengths"] for group in groups.values()) == 0.5


def test_targeted_diagnostic_smoke(tmp_path, monkeypatch) -> None:
    reference = _payload()
    artifact = {
        "model_kind": "b3_plus_dp_plus_aod_residual",
        "aod_architecture": "link_global",
        "reference": reference,
        "physics_only_baseline": _generated(reference, 2.0),
        "correct_map": _generated(reference, 1.0),
    }
    artifact_path = tmp_path / "artifact.pt"
    output_path = tmp_path / "report.json"
    torch.save(artifact, artifact_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "diagnose_csi_remaining_bottlenecks.py",
            "--artifact",
            str(artifact_path),
            "--output",
            str(output_path),
            "--device",
            "cpu",
            "--frequency-lags",
            "1,2,4,8",
            "--minimum-pairs-per-map",
            "2",
            "--maximum-condition-distance",
            "10",
            "--maximum-counterfactual-distance",
            "10",
            "--maximum-counterfactual-samples",
            "8",
            "--bootstrap-samples",
            "100",
            "--batch-size",
            "4",
        ],
    )
    diagnostic.main()
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["maps"] == 2
    assert report["counterfactual_selection_audit"]["selected_pairs"] == 8
    assert set(report["counterfactual"]["models"]) == {"b3", "current"}
    assert "16" not in report["frequency_and_tx_decomposition"]["frequency_per_lag_map_summary"]
    variants = report["counterfactual"]["models"]["current"]["variants"]
    assert "phase_only" in variants
    assert "delay_phase_joint" in variants
    assert "aod_global_orientation_only" in variants
    assert "aod_relative_structure_only" in variants
