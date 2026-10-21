from __future__ import annotations

import gzip
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

SPEED_OF_LIGHT_M_S = 299_792_458.0
PROP_MAGIC = {b"PORP", b"PROP"}
PROP_HEADER_BYTES = 32

RX_DTYPE = np.dtype(
    [
        ("point_id", "<i4"),
        ("x_m", "<f4"),
        ("y_m", "<f4"),
        ("z_m", "<f4"),
        ("total_intensity_dbm", "<f4"),
        ("total_loss_db", "<f4"),
        ("path_count", "<u2"),
        ("_pad", "<u2"),
        ("path_offset", "<u4"),
    ]
)

PATH_DTYPE = np.dtype(
    [
        ("path_loss_db", "<f2"),
        ("distance_m", "<f2"),
        ("delay_ns", "<f2"),
        ("aod_az_rad", "<f2"),
        ("aod_el_rad", "<f2"),
        ("aoa_az_rad", "<f2"),
        ("aoa_el_rad", "<f2"),
        ("interaction_count", "<u2"),
        ("path_type", "<i1"),
        ("_pad1", "<u1"),
        ("_pad2", "<u2"),
        ("interaction_offset", "<u4"),
    ]
)


@dataclass(frozen=True)
class RayVerseLink:
    map_id: int
    source_id: int
    receiver_id: int
    tx_xyz_m: np.ndarray
    rx_xyz_m: np.ndarray
    total_intensity_dbm: float
    total_loss_db: float
    carrier_hz: float
    path_loss_db: np.ndarray
    distance_m: np.ndarray
    delay_ns: np.ndarray
    aod_az_deg: np.ndarray
    aod_el_deg: np.ndarray
    aoa_az_deg: np.ndarray
    aoa_el_deg: np.ndarray
    interaction_count: np.ndarray
    path_type: np.ndarray

    @property
    def path_count(self) -> int:
        return int(self.path_loss_db.size)


@dataclass(frozen=True)
class RayVersePropbin:
    path: Path
    version: int
    source_xyz_m: np.ndarray
    carrier_hz: float
    receivers: np.ndarray
    paths: np.ndarray
    interactions: np.ndarray

    def iter_links(
        self,
        *,
        max_receivers: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> Iterator[RayVerseLink]:
        valid = np.flatnonzero(self.receivers["path_count"] > 0)
        if max_receivers is not None and valid.size > max_receivers:
            rng = rng or np.random.default_rng(0)
            valid = np.sort(rng.choice(valid, size=max_receivers, replace=False))
        map_id, source_id = ids_from_path(self.path)
        for index in valid:
            receiver = self.receivers[int(index)]
            start = int(receiver["path_offset"])
            count = int(receiver["path_count"])
            records = self.paths[start : start + count]
            finite = (
                np.isfinite(records["path_loss_db"])
                & np.isfinite(records["distance_m"])
                & np.isfinite(records["delay_ns"])
            )
            records = records[finite]
            if records.size == 0:
                continue
            order = np.argsort(records["delay_ns"], kind="stable")
            records = records[order]
            yield RayVerseLink(
                map_id=map_id,
                source_id=source_id,
                receiver_id=int(receiver["point_id"]),
                tx_xyz_m=self.source_xyz_m.copy(),
                rx_xyz_m=np.asarray(
                    [receiver["x_m"], receiver["y_m"], receiver["z_m"]],
                    dtype=np.float32,
                ),
                total_intensity_dbm=float(receiver["total_intensity_dbm"]),
                total_loss_db=float(receiver["total_loss_db"]),
                carrier_hz=self.carrier_hz,
                path_loss_db=records["path_loss_db"].astype(np.float32),
                distance_m=records["distance_m"].astype(np.float32),
                delay_ns=records["delay_ns"].astype(np.float32),
                aod_az_deg=np.rad2deg(records["aod_az_rad"].astype(np.float32)),
                aod_el_deg=np.rad2deg(records["aod_el_rad"].astype(np.float32)),
                aoa_az_deg=np.rad2deg(records["aoa_az_rad"].astype(np.float32)),
                aoa_el_deg=np.rad2deg(records["aoa_el_rad"].astype(np.float32)),
                interaction_count=records["interaction_count"].astype(np.int64),
                path_type=records["path_type"].astype(np.int64),
            )


def read_propbin(path: str | Path, fallback_carrier_hz: float = 3.5e9) -> RayVersePropbin:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        data = stream.read()
    if len(data) < PROP_HEADER_BYTES:
        raise ValueError(f"PROP file is too small: {path}")
    if data[:4] not in PROP_MAGIC:
        raise ValueError(f"Unsupported PROP magic {data[:4]!r}: {path}")

    version = int(np.frombuffer(data, dtype="<u4", count=1, offset=4)[0])
    if version != 2:
        raise ValueError(f"Only compact RayVerse PROP v2 is supported, got v{version}: {path}")
    source_xyz = np.frombuffer(data, dtype="<f4", count=3, offset=8).copy()
    receiver_count, path_count, interaction_count = np.frombuffer(
        data, dtype="<u4", count=3, offset=20
    ).astype(np.int64)
    receiver_offset = PROP_HEADER_BYTES
    path_offset = receiver_offset + int(receiver_count) * RX_DTYPE.itemsize
    interaction_offset = path_offset + int(path_count) * PATH_DTYPE.itemsize
    expected_size = interaction_offset + int(interaction_count) * np.dtype("<u2").itemsize
    if len(data) != expected_size:
        raise ValueError(
            f"Unexpected PROP size for {path}: got {len(data)}, expected {expected_size}"
        )
    receivers = np.frombuffer(
        data, dtype=RX_DTYPE, count=int(receiver_count), offset=receiver_offset
    ).copy()
    paths = np.frombuffer(data, dtype=PATH_DTYPE, count=int(path_count), offset=path_offset).copy()
    interactions = np.frombuffer(
        data, dtype="<u2", count=int(interaction_count), offset=interaction_offset
    ).copy()
    return RayVersePropbin(
        path=path,
        version=version,
        source_xyz_m=source_xyz,
        carrier_hz=carrier_hz_from_path(path, fallback_carrier_hz),
        receivers=receivers,
        paths=paths,
        interactions=interactions,
    )


def carrier_hz_from_path(path: Path, fallback: float) -> float:
    for parent in (path.parent, *path.parents):
        match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*GHz", parent.name, re.IGNORECASE)
        if match:
            return float(match.group(1)) * 1e9
    return fallback


def ids_from_path(path: Path) -> tuple[int, int]:
    map_match = next((re.fullmatch(r"map_(\d+)", part) for part in path.parts if part.startswith("map_")), None)
    source_match = re.search(r"source_(\d+)", path.name)
    if map_match is None or source_match is None:
        raise ValueError(f"Cannot infer map/source ids from {path}")
    return int(map_match.group(1)), int(source_match.group(1))


def discover_propbin_files(
    root: str | Path,
    *,
    max_sources_per_map: int | None = None,
) -> list[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    files: list[Path] = []
    map_dirs = sorted(root.glob("map_*"), key=lambda p: _natural_number(p.name))
    for map_dir in map_dirs:
        source_files = sorted(
            map_dir.glob("special_points_propbin_*/source_*.propbin*"),
            key=lambda p: _natural_number(p.name),
        )
        if max_sources_per_map is not None:
            source_files = source_files[:max_sources_per_map]
        files.extend(source_files)
    return files


def phase_from_distance(distance_m: np.ndarray, carrier_hz: float) -> np.ndarray:
    phase = -2.0 * math.pi * carrier_hz * distance_m / SPEED_OF_LIGHT_M_S
    return np.angle(np.exp(1j * phase)).astype(np.float32)


def path_gain_from_loss(path_loss_db: np.ndarray) -> np.ndarray:
    return np.power(10.0, -path_loss_db.astype(np.float64) / 10.0).astype(np.float32)


def _natural_number(value: str) -> int:
    match = re.search(r"(\d+)", value)
    return int(match.group(1)) if match else -1

