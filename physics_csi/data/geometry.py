from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch


GEOMETRY_FORMAT_VERSION = 1


def _quantiles(values: np.ndarray) -> tuple[float, float, float, float]:
    if values.size == 0:
        return (0.0, 0.0, 0.0, 0.0)
    result = np.quantile(values, [0.25, 0.50, 0.75, 0.90])
    return tuple(float(value) for value in result)


def _distribution_features(values: np.ndarray, prefix: str) -> tuple[list[str], list[float]]:
    transformed = np.log1p(np.maximum(values.astype(np.float64), 0.0))
    q25, q50, q75, q90 = _quantiles(transformed)
    names = [
        f"{prefix}_log1p_mean",
        f"{prefix}_log1p_std",
        f"{prefix}_log1p_q25",
        f"{prefix}_log1p_q50",
        f"{prefix}_log1p_q75",
        f"{prefix}_log1p_q90",
    ]
    values_out = [
        float(transformed.mean()) if transformed.size else 0.0,
        float(transformed.std()) if transformed.size else 0.0,
        q25,
        q50,
        q75,
        q90,
    ]
    return names, values_out


def _polygon_measurements(points: np.ndarray) -> dict[str, Any] | None:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("each building polygon must have shape [vertices, 2]")
    if points.shape[0] >= 2 and np.allclose(points[0], points[-1]):
        points = points[:-1]
    if points.shape[0] < 3 or not np.isfinite(points).all():
        return None
    following = np.roll(points, -1, axis=0)
    cross = points[:, 0] * following[:, 1] - following[:, 0] * points[:, 1]
    signed_double_area = float(cross.sum())
    area = abs(signed_double_area) * 0.5
    edges = following - points
    lengths = np.linalg.norm(edges, axis=1)
    valid_edges = lengths > 1e-12
    perimeter = float(lengths.sum())
    if area > 1e-12 and abs(signed_double_area) > 1e-12:
        centroid = np.asarray(
            [
                ((points[:, 0] + following[:, 0]) * cross).sum(),
                ((points[:, 1] + following[:, 1]) * cross).sum(),
            ],
            dtype=np.float64,
        ) / (3.0 * signed_double_area)
    else:
        centroid = points.mean(axis=0)
    compactness = (
        float(4.0 * math.pi * area / max(perimeter * perimeter, 1e-20))
        if area > 0.0
        else 0.0
    )
    return {
        "area": area,
        "perimeter": perimeter,
        "compactness": compactness,
        "centroid": centroid,
        "edge_lengths": lengths[valid_edges],
        "edge_angles_rad": np.mod(
            np.arctan2(edges[valid_edges, 1], edges[valid_edges, 0]), math.pi
        ),
    }


def geometry_feature_names(
    *, orientation_bins: int, layout_grid_size: int
) -> tuple[str, ...]:
    names = [
        "building_count_per_10000_source_units2",
        "summed_footprint_fraction",
        "perimeter_per_source_unit2",
    ]
    for prefix in ("building_area", "building_perimeter", "edge_length"):
        distribution_names, _ = _distribution_features(np.empty(0), prefix)
        names.extend(distribution_names)
    names.extend(
        [
            "building_compactness_mean",
            "building_compactness_std",
            "building_compactness_q25",
            "building_compactness_q50",
            "building_compactness_q75",
            "building_compactness_q90",
        ]
    )
    names.extend(f"edge_orientation_length_fraction_bin_{index}" for index in range(orientation_bins))
    names.extend(
        f"area_weighted_centroid_layout_row_{row}_col_{column}"
        for row in range(layout_grid_size)
        for column in range(layout_grid_size)
    )
    return tuple(names)


def extract_map_geometry_features(
    polygons: Iterable[Any],
    *,
    map_size: float = 256.0,
    orientation_bins: int = 12,
    layout_grid_size: int = 4,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract fixed-length, path-label-free features from building footprints."""
    if map_size <= 0 or orientation_bins < 2 or layout_grid_size < 1:
        raise ValueError("map_size must be positive and histogram sizes must be valid")
    measurements = []
    invalid_polygons = 0
    for polygon in polygons:
        measurement = _polygon_measurements(np.asarray(polygon))
        if measurement is None:
            invalid_polygons += 1
        else:
            measurements.append(measurement)
    areas = np.asarray([item["area"] for item in measurements], dtype=np.float64)
    perimeters = np.asarray([item["perimeter"] for item in measurements], dtype=np.float64)
    compactness = np.asarray([item["compactness"] for item in measurements], dtype=np.float64)
    edge_lengths = (
        np.concatenate([item["edge_lengths"] for item in measurements])
        if measurements
        else np.empty(0, dtype=np.float64)
    )
    edge_angles = (
        np.concatenate([item["edge_angles_rad"] for item in measurements])
        if measurements
        else np.empty(0, dtype=np.float64)
    )
    map_area = map_size * map_size
    names = list(
        geometry_feature_names(
            orientation_bins=orientation_bins, layout_grid_size=layout_grid_size
        )
    )
    features = [
        len(measurements) / map_area * 10_000.0,
        float(areas.sum()) / map_area,
        float(perimeters.sum()) / map_area,
    ]
    for values, prefix in (
        (areas, "building_area"),
        (perimeters, "building_perimeter"),
        (edge_lengths, "edge_length"),
    ):
        _, distribution = _distribution_features(values, prefix)
        features.extend(distribution)
    q25, q50, q75, q90 = _quantiles(compactness)
    features.extend(
        [
            float(compactness.mean()) if compactness.size else 0.0,
            float(compactness.std()) if compactness.size else 0.0,
            q25,
            q50,
            q75,
            q90,
        ]
    )
    orientation_histogram, _ = np.histogram(
        edge_angles,
        bins=np.linspace(0.0, math.pi, orientation_bins + 1),
        weights=edge_lengths,
    )
    orientation_histogram = orientation_histogram.astype(np.float64)
    orientation_histogram /= max(float(orientation_histogram.sum()), 1e-20)
    features.extend(orientation_histogram.tolist())

    layout = np.zeros((layout_grid_size, layout_grid_size), dtype=np.float64)
    for area, item in zip(areas, measurements):
        normalized = np.clip(item["centroid"] / map_size, 0.0, 1.0 - 1e-12)
        column = min(int(normalized[0] * layout_grid_size), layout_grid_size - 1)
        row = min(int(normalized[1] * layout_grid_size), layout_grid_size - 1)
        layout[row, column] += area
    layout /= max(float(layout.sum()), 1e-20)
    features.extend(layout.flatten().tolist())
    result = np.asarray(features, dtype=np.float32)
    if result.shape != (len(names),) or not np.isfinite(result).all():
        raise RuntimeError("geometry feature extraction produced an invalid feature vector")
    summary = {
        "valid_buildings": len(measurements),
        "invalid_buildings": invalid_polygons,
        "summed_building_area": float(areas.sum()),
        "total_perimeter": float(perimeters.sum()),
        "source_coordinate_bounds": [0.0, float(map_size)],
        "coverage_semantics": (
            "sum of polygon footprint areas divided by map area; polygons are assumed not to "
            "overlap, so this is not a computational-geometry union"
        ),
    }
    return result, summary


def build_geometry_feature_artifact(
    geometry_root: str | Path,
    map_ids: Iterable[int],
    *,
    map_size: float = 256.0,
    orientation_bins: int = 12,
    layout_grid_size: int = 4,
) -> dict[str, Any]:
    root = Path(geometry_root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    names = geometry_feature_names(
        orientation_bins=orientation_bins, layout_grid_size=layout_grid_size
    )
    maps: dict[str, Any] = {}
    combined_hash = hashlib.sha256()
    for map_id in sorted({int(value) for value in map_ids}):
        source = root / f"{map_id}.json"
        if not source.is_file():
            raise FileNotFoundError(f"geometry for map_{map_id} is missing: {source}")
        source_bytes = source.read_bytes()
        combined_hash.update(str(map_id).encode())
        combined_hash.update(source_bytes)
        polygons = json.loads(source_bytes)
        if not isinstance(polygons, list):
            raise ValueError(f"geometry must be a list of polygons: {source}")
        features, summary = extract_map_geometry_features(
            polygons,
            map_size=map_size,
            orientation_bins=orientation_bins,
            layout_grid_size=layout_grid_size,
        )
        maps[str(map_id)] = {
            "source_file": str(source),
            "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "features": features.tolist(),
            "summary": summary,
        }
    return {
        "format_version": GEOMETRY_FORMAT_VERSION,
        "kind": "map_global_building_geometry_statistics",
        "geometry_root": str(root),
        "source_sha256": combined_hash.hexdigest(),
        "map_size_source_units": float(map_size),
        "orientation_bins": int(orientation_bins),
        "layout_grid_size": int(layout_grid_size),
        "feature_names": list(names),
        "maps": maps,
        "cautions": [
            "Features are map-global and contain no Tx/Rx coordinates or propagation labels.",
            "Edge orientation is axial over [0, 180 degrees) and weighted by edge length.",
            "Geometry must use the same coordinate frame as the RayVerse map for orientation-sensitive claims.",
        ],
    }


def load_geometry_feature_artifact(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if (
        payload.get("format_version") != GEOMETRY_FORMAT_VERSION
        or payload.get("kind") != "map_global_building_geometry_statistics"
    ):
        raise ValueError("unsupported geometry feature artifact")
    names = payload.get("feature_names")
    maps = payload.get("maps")
    if not isinstance(names, list) or not names or not isinstance(maps, Mapping):
        raise ValueError("geometry feature artifact is incomplete")
    for map_id, item in maps.items():
        features = torch.as_tensor(item.get("features"), dtype=torch.float32)
        if features.shape != (len(names),) or not torch.isfinite(features).all():
            raise ValueError(f"invalid geometry features for map_{map_id}")
    return payload


def geometry_matrix_for_map_ids(
    artifact: Mapping[str, Any], map_ids: torch.Tensor
) -> torch.Tensor:
    maps = artifact["maps"]
    rows = []
    for value in map_ids.long().cpu().tolist():
        key = str(int(value))
        if key not in maps:
            raise KeyError(f"geometry artifact has no map_{key}")
        rows.append(torch.tensor(maps[key]["features"], dtype=torch.float32))
    return torch.stack(rows)
