#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_csi.config import load_project_config
from physics_csi.data.prepare import link_to_record
from physics_csi.data.rayverse import discover_propbin_files, read_propbin
from physics_csi.physics.synthesizer import MIMOOFDMSynthesizer, SystemConfig


def numpy_reference(record: dict, system: SystemConfig) -> np.ndarray:
    mask = record["path_mask"].numpy().astype(bool)
    delays = record["path_delays_ns"].numpy()[mask] * 1e-9
    powers = record["path_powers_linear"].numpy()[mask]
    phases = record["path_phases_rad"].numpy()[mask]
    aoa_az = np.deg2rad(record["aoa_az_deg"].numpy()[mask])
    aoa_el = np.deg2rad(record["aoa_el_deg"].numpy()[mask])
    aod_az = np.deg2rad(record["aod_az_deg"].numpy()[mask])
    aod_el = np.deg2rad(record["aod_el_deg"].numpy()[mask])

    def positions(shape, spacing, plane):
        row = np.arange(shape[0]) - (shape[0] - 1) / 2
        col = np.arange(shape[1]) - (shape[1] - 1) / 2
        rr, cc = np.meshgrid(row, col, indexing="ij")
        result = np.zeros((shape[0], shape[1], 3), dtype=np.float64)
        axes = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}[plane]
        result[..., axes[0]] = rr * spacing
        result[..., axes[1]] = cc * spacing
        return result.reshape(-1, 3)

    def response(az, el, pos):
        direction = np.stack(
            [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=-1
        )
        value = np.exp(1j * 2 * np.pi * np.einsum("ld,nd->ln", direction, pos))
        if system.normalize_array_response:
            value /= np.sqrt(len(pos))
        return value

    tx = response(
        aod_az,
        aod_el,
        positions(system.tx_array_shape, system.tx_spacing_wavelengths, system.tx_array_plane),
    )
    rx = response(
        aoa_az,
        aoa_el,
        positions(system.rx_array_shape, system.rx_spacing_wavelengths, system.rx_array_plane),
    )
    indices = np.arange(system.num_subcarriers, dtype=np.float64)
    if system.center_subcarriers:
        indices -= (system.num_subcarriers - 1) / 2
    offsets = indices * system.bandwidth_hz / system.num_subcarriers
    frequency = np.exp(-1j * 2 * np.pi * delays[:, None] * offsets[None, :])
    gain = np.sqrt(powers) * np.exp(1j * phases)
    return np.einsum("l,lf,lr,lt->rtf", gain, frequency, rx, tx.conj()).astype(np.complex64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-check PyTorch CSI synthesis with NumPy.")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--file-index", type=int, default=0)
    parser.add_argument("--receiver-index", type=int, default=0)
    args = parser.parse_args()
    config = load_project_config(args.config_dir)
    data_config = config["data"]
    system = SystemConfig.from_mapping(data_config["system"])
    files = discover_propbin_files(data_config["raw_root"])
    propbin = read_propbin(files[args.file_index], fallback_carrier_hz=system.carrier_hz)
    link = next(
        link
        for index, link in enumerate(propbin.iter_links())
        if index == args.receiver_index
    )
    record = link_to_record(
        link,
        max_paths=int(data_config["max_paths"]),
        overflow_policy=str(data_config["overflow_policy"]),
        invalid_k_fill_db=float(data_config["k_factor"]["invalid_fill_db"]),
    )
    synthesizer = MIMOOFDMSynthesizer(system)
    torch_channel = synthesizer(
        record["path_delays_ns"].unsqueeze(0),
        record["path_powers_linear"].unsqueeze(0),
        record["path_phases_rad"].unsqueeze(0),
        record["aoa_az_deg"].unsqueeze(0),
        record["aoa_el_deg"].unsqueeze(0),
        record["aod_az_deg"].unsqueeze(0),
        record["aod_el_deg"].unsqueeze(0),
        record["path_mask"].unsqueeze(0),
    )[0]
    reference = torch.from_numpy(numpy_reference(record, system))
    nmse = (torch_channel - reference).abs().square().sum() / reference.abs().square().sum().clamp_min(1e-20)
    report = {
        "sample_id": record["sample_id"],
        "shape": list(torch_channel.shape),
        "dtype": str(torch_channel.dtype),
        "numpy_crosscheck_nmse": float(nmse),
        "note": "Cross-checks two implementations of the selected synthesis convention; no native CSI is stored in PROP.",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

