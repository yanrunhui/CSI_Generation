#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_csi.config import load_yaml
from physics_csi.data.dataset import PreparedShardDataset
from physics_csi.data.geometry import load_geometry_feature_artifact
from physics_csi.data.scene_id import load_seen_map_spatial_plan, make_scene_id_datasets
from physics_csi.evaluation.geometry_probe import (
    collect_probe_arrays,
    cyclic_context_overrides,
    fit_probe,
    predict_probe,
    prepare_probe_arrays,
    probe_metrics,
)
from physics_csi.training.trainer import resolve_device
from physics_csi.utils.io import atomic_json_dump


def _datasets_for_protocol(
    protocol: str, prepared_root: str, scene_plan: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    if protocol == "seen_map_spatial":
        plan = load_seen_map_spatial_plan(scene_plan)
        return make_scene_id_datasets(prepared_root, plan), {
            "kind": protocol,
            "scene_plan_fingerprint": plan["fingerprint"],
        }
    datasets = {
        split: PreparedShardDataset(prepared_root, split)
        for split in ("train", "validation", "test")
    }
    manifest = datasets["train"].manifest
    if manifest.get("split_mode") != "cross_map" or not manifest.get("map_split"):
        raise ValueError("unseen_map protocol requires a prepared cross-map split")
    return datasets, {
        "kind": protocol,
        "prepared_map_split": manifest["map_split"],
    }


def _comparison(
    correct: dict[str, Any], comparison: dict[str, Any]
) -> dict[str, Any]:
    groups = {
        group: (
            comparison["groups"][group]["standardized_mse"]
            - correct["groups"][group]["standardized_mse"]
        )
        for group in correct["groups"]
    }
    correct_maps = correct["per_map"]
    comparison_maps = comparison["per_map"]
    return {
        "micro_mse_reduction": (
            comparison["standardized_mse_micro"] - correct["standardized_mse_micro"]
        ),
        "map_macro_mse_reduction": (
            comparison["standardized_mse_map_macro"]
            - correct["standardized_mse_map_macro"]
        ),
        "group_mse_reduction": groups,
        "maps_correct_wins": sum(
            correct_maps[name]["standardized_mse"]
            < comparison_maps[name]["standardized_mse"]
            for name in correct_maps
        ),
        "maps_total": len(correct_maps),
    }


def _run_protocol(
    protocol: str,
    *,
    prepared_root: str,
    scene_plan: str,
    geometry_artifact: dict[str, Any],
    tx_array_shape: tuple[int, int],
    feature_batch_size: int,
    num_workers: int,
    hidden_dim: int,
    context_dim: int,
    geometry_hidden_dim: int,
    train_batch_size: int,
    epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    seeds: list[int],
    device: torch.device,
) -> dict[str, Any]:
    datasets, split_metadata = _datasets_for_protocol(
        protocol, prepared_root, scene_plan
    )
    raw = {
        split: collect_probe_arrays(
            dataset,
            tx_array_shape=tx_array_shape,
            batch_size=feature_batch_size,
            num_workers=num_workers,
        )
        for split, dataset in datasets.items()
    }
    feature_names = raw["train"]["target_names"]
    if any(values["target_names"] != feature_names for values in raw.values()):
        raise RuntimeError("probe target schemas differ across splits")
    split_maps = {
        split: sorted(int(value) for value in values["map_id"].unique())
        for split, values in raw.items()
    }
    if protocol == "unseen_map":
        if set(split_maps["train"]) & set(split_maps["validation"]):
            raise RuntimeError("unseen-map train and validation Map IDs overlap")
        if set(split_maps["train"]) & set(split_maps["test"]):
            raise RuntimeError("unseen-map train and test Map IDs overlap")
        if set(split_maps["validation"]) & set(split_maps["test"]):
            raise RuntimeError("unseen-map validation and test Map IDs overlap")
    arrays, normalization = prepare_probe_arrays(raw, geometry_artifact)
    wrong_map, wrong_geometry = cyclic_context_overrides(arrays["test"])
    modes = ["physics", "geometry"]
    if protocol == "seen_map_spatial":
        modes.insert(1, "map")
    runs: list[dict[str, Any]] = []
    for seed in seeds:
        models = {}
        fit_reports = {}
        for mode in modes:
            model, fit_report = fit_probe(
                arrays,
                mode=mode,
                map_count=len(normalization["training_maps"]),
                hidden_dim=hidden_dim,
                context_dim=context_dim,
                geometry_hidden_dim=geometry_hidden_dim,
                batch_size=train_batch_size,
                epochs=epochs,
                patience=patience,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                device=device,
                seed=seed,
            )
            models[mode] = model
            fit_reports[mode] = fit_report
        test_metrics = {}
        for mode, model in models.items():
            prediction = predict_probe(
                model, arrays["test"], device=device, batch_size=train_batch_size
            )
            test_metrics[mode] = probe_metrics(
                prediction,
                arrays["test"]["y"],
                feature_names,
                arrays["test"]["map_id"],
            )
        wrong_geometry_prediction = predict_probe(
            models["geometry"],
            arrays["test"],
            device=device,
            batch_size=train_batch_size,
            geometry_override=wrong_geometry,
        )
        test_metrics["geometry_shuffled"] = probe_metrics(
            wrong_geometry_prediction,
            arrays["test"]["y"],
            feature_names,
            arrays["test"]["map_id"],
        )
        if "map" in models:
            wrong_map_prediction = predict_probe(
                models["map"],
                arrays["test"],
                device=device,
                batch_size=train_batch_size,
                map_override=wrong_map,
            )
            test_metrics["map_shuffled"] = probe_metrics(
                wrong_map_prediction,
                arrays["test"]["y"],
                feature_names,
                arrays["test"]["map_id"],
            )
        comparisons = {
            "geometry_vs_physics": _comparison(
                test_metrics["geometry"], test_metrics["physics"]
            ),
            "geometry_vs_shuffled_geometry": _comparison(
                test_metrics["geometry"], test_metrics["geometry_shuffled"]
            ),
        }
        if "map" in test_metrics:
            comparisons["map_vs_physics"] = _comparison(
                test_metrics["map"], test_metrics["physics"]
            )
            comparisons["geometry_vs_map"] = _comparison(
                test_metrics["geometry"], test_metrics["map"]
            )
            comparisons["map_vs_shuffled_map"] = _comparison(
                test_metrics["map"], test_metrics["map_shuffled"]
            )
        runs.append(
            {
                "seed": seed,
                "fit": fit_reports,
                "test": test_metrics,
                "comparisons": comparisons,
            }
        )
    comparison_names = list(runs[0]["comparisons"])
    summary = {}
    for comparison_name in comparison_names:
        comparison_runs = [run["comparisons"][comparison_name] for run in runs]
        groups = comparison_runs[0]["group_mse_reduction"]
        summary[comparison_name] = {
            "mean_micro_mse_reduction": float(
                torch.tensor([item["micro_mse_reduction"] for item in comparison_runs]).mean()
            ),
            "mean_map_macro_mse_reduction": float(
                torch.tensor(
                    [item["map_macro_mse_reduction"] for item in comparison_runs]
                ).mean()
            ),
            "seeds_with_positive_micro_reduction": sum(
                item["micro_mse_reduction"] > 0 for item in comparison_runs
            ),
            "seeds_total": len(comparison_runs),
            "mean_map_win_rate": float(
                torch.tensor(
                    [item["maps_correct_wins"] / max(item["maps_total"], 1) for item in comparison_runs]
                ).mean()
            ),
            "mean_group_mse_reduction": {
                group: float(
                    torch.tensor(
                        [item["group_mse_reduction"][group] for item in comparison_runs]
                    ).mean()
                )
                for group in groups
            },
            "seeds_with_positive_group_reduction": {
                group: sum(
                    item["group_mse_reduction"][group] > 0 for item in comparison_runs
                )
                for group in groups
            },
        }
    return {
        "split": split_metadata,
        "split_maps": split_maps,
        "split_samples": {
            split: int(values["y"].shape[0]) for split, values in arrays.items()
        },
        "normalization": normalization,
        "target_features": list(feature_names),
        "summary": summary,
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run P0/P1/P2 building-geometry probes without diffusion."
    )
    parser.add_argument("--prepared-root", default="artifacts/prepared_d2los")
    parser.add_argument("--scene-plan", required=True)
    parser.add_argument("--geometry-features", required=True)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument(
        "--protocol",
        choices=["seen_map_spatial", "unseen_map", "both"],
        default="both",
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--context-dim", type=int, default=16)
    parser.add_argument("--geometry-hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--feature-batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seeds", default="2026,2027,2028")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    positive = (
        args.hidden_dim,
        args.context_dim,
        args.geometry_hidden_dim,
        args.epochs,
        args.patience,
        args.batch_size,
        args.feature_batch_size,
    )
    if min(positive) < 1:
        parser.error("model dimensions, epochs, patience, and batch sizes must be positive")
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        parser.error("--seeds cannot be empty")
    device = resolve_device(args.device)
    geometry_artifact = load_geometry_feature_artifact(args.geometry_features)
    data_config = load_yaml(args.data_config)["data"]
    tx_array_shape = tuple(int(value) for value in data_config["system"]["tx_array_shape"])
    protocols = (
        ["seen_map_spatial", "unseen_map"]
        if args.protocol == "both"
        else [args.protocol]
    )
    results = {
        protocol: _run_protocol(
            protocol,
            prepared_root=args.prepared_root,
            scene_plan=args.scene_plan,
            geometry_artifact=geometry_artifact,
            tx_array_shape=tx_array_shape,
            feature_batch_size=args.feature_batch_size,
            num_workers=args.num_workers,
            hidden_dim=args.hidden_dim,
            context_dim=args.context_dim,
            geometry_hidden_dim=args.geometry_hidden_dim,
            train_batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seeds=seeds,
            device=device,
        )
        for protocol in protocols
    }
    report = {
        "format_version": 1,
        "diagnostic": "b3_s_geo_lite_geometry_probe",
        "prepared_root": str(Path(args.prepared_root).resolve()),
        "scene_plan": str(Path(args.scene_plan).resolve()),
        "geometry_features": str(Path(args.geometry_features).resolve()),
        "geometry_source_sha256": geometry_artifact["source_sha256"],
        "geometry_feature_names": geometry_artifact["feature_names"],
        "device": str(device),
        "seeds": seeds,
        "models": {
            "P0": "physical conditions + learned global null context",
            "P1": "physical conditions + categorical Map embedding; seen-map protocol only",
            "P2": "physical conditions + standardized building statistics + Geometry Encoder",
        },
        "protocols": results,
        "decision_rule": (
            "Proceed to diffusion only if correct P2 geometry consistently improves AoD, "
            "Power-AoD, and spatial-correlation targets over P0 and shuffled geometry. "
            "Unseen-map improvements must be judged only on complete held-out Maps."
        ),
    }
    atomic_json_dump(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
