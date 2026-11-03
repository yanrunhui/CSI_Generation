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
    fit_residual_geometry_probe,
    geometry_ood_audit,
    geometry_variant_indices,
    predict_probe,
    predict_residual_geometry_probe,
    prepare_probe_arrays,
    probe_metrics,
    subset_geometry_arrays,
)
from physics_csi.training.trainer import resolve_device
from physics_csi.utils.io import atomic_json_dump


def _datasets(
    protocol: str, prepared_root: str, scene_plan: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    if protocol == "seen_map_spatial":
        plan = load_seen_map_spatial_plan(scene_plan)
        return make_scene_id_datasets(prepared_root, plan), {
            "kind": protocol,
            "scene_plan_fingerprint": plan["fingerprint"],
        }
    result = {
        split: PreparedShardDataset(prepared_root, split)
        for split in ("train", "validation", "test")
    }
    manifest = result["train"].manifest
    if manifest.get("split_mode") != "cross_map" or not manifest.get("map_split"):
        raise ValueError("unseen_map requires a prepared cross-map split")
    return result, {"kind": protocol, "prepared_map_split": manifest["map_split"]}


def _comparison(
    correct: dict[str, Any], comparison: dict[str, Any]
) -> dict[str, Any]:
    return {
        "micro_mse_reduction": (
            comparison["standardized_mse_micro"] - correct["standardized_mse_micro"]
        ),
        "map_macro_mse_reduction": (
            comparison["standardized_mse_map_macro"]
            - correct["standardized_mse_map_macro"]
        ),
        "group_mse_reduction": {
            group: (
                comparison["groups"][group]["standardized_mse"]
                - correct["groups"][group]["standardized_mse"]
            )
            for group in correct["groups"]
        },
        "maps_correct_wins": sum(
            correct["per_map"][name]["standardized_mse"]
            < comparison["per_map"][name]["standardized_mse"]
            for name in correct["per_map"]
        ),
        "maps_total": len(correct["per_map"]),
    }


def _summary(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    groups = comparisons[0]["group_mse_reduction"]
    return {
        "mean_micro_mse_reduction": float(
            torch.tensor([item["micro_mse_reduction"] for item in comparisons]).mean()
        ),
        "mean_map_macro_mse_reduction": float(
            torch.tensor([item["map_macro_mse_reduction"] for item in comparisons]).mean()
        ),
        "seeds_with_positive_micro_reduction": sum(
            item["micro_mse_reduction"] > 0 for item in comparisons
        ),
        "seeds_total": len(comparisons),
        "mean_map_win_rate": float(
            torch.tensor(
                [item["maps_correct_wins"] / max(item["maps_total"], 1) for item in comparisons]
            ).mean()
        ),
        "mean_group_mse_reduction": {
            group: float(
                torch.tensor(
                    [item["group_mse_reduction"][group] for item in comparisons]
                ).mean()
            )
            for group in groups
        },
        "seeds_with_positive_group_reduction": {
            group: sum(
                item["group_mse_reduction"][group] > 0 for item in comparisons
            )
            for group in groups
        },
    }


def _ood_improvement_relationship(
    runs: list[dict[str, Any]], variant: str, ood: dict[str, Any]
) -> dict[str, Any]:
    map_names = sorted(ood["per_map"])
    distances = torch.tensor(
        [ood["per_map"][name]["nearest_standardized_rms_distance"] for name in map_names]
    )
    improvements = []
    per_map = {}
    for name in map_names:
        values = [
            run["base_test"]["per_map"][name]["standardized_mse"]
            - run["variants"][variant]["correct_test"]["per_map"][name]["standardized_mse"]
            for run in runs
        ]
        mean_value = float(torch.tensor(values).mean())
        improvements.append(mean_value)
        per_map[name] = {
            "nearest_training_geometry_distance": ood["per_map"][name][
                "nearest_standardized_rms_distance"
            ],
            "mean_residual_mse_reduction_vs_p0": mean_value,
        }
    improvement_tensor = torch.tensor(improvements)
    if (
        distances.numel() >= 2
        and float(distances.std(unbiased=False)) > 1e-12
        and float(improvement_tensor.std(unbiased=False)) > 1e-12
    ):
        correlation = float(torch.corrcoef(torch.stack([distances, improvement_tensor]))[0, 1])
    else:
        correlation = None
    return {
        "distance_vs_improvement_pearson": correlation,
        "note": "Negative correlation means geometrically distant Maps benefit less from the residual.",
        "per_map": per_map,
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
    residual_hidden_dim: int,
    geometry_dropout: float,
    maximum_gate: float,
    initial_gate: float,
    correction_penalty: float,
    batch_size: int,
    epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    seeds: list[int],
    device: torch.device,
) -> dict[str, Any]:
    datasets, split_metadata = _datasets(protocol, prepared_root, scene_plan)
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
    arrays, normalization = prepare_probe_arrays(raw, geometry_artifact)
    split_maps = {
        split: sorted(int(value) for value in values["map_id"].unique())
        for split, values in raw.items()
    }
    if protocol == "unseen_map":
        split_sets = {key: set(value) for key, value in split_maps.items()}
        if split_sets["train"] & split_sets["validation"]:
            raise RuntimeError("unseen-map train and validation maps overlap")
        if split_sets["train"] & split_sets["test"]:
            raise RuntimeError("unseen-map train and test maps overlap")
        if split_sets["validation"] & split_sets["test"]:
            raise RuntimeError("unseen-map validation and test maps overlap")
    _, shuffled_geometry = cyclic_context_overrides(arrays["test"])
    geometry_names = geometry_artifact["feature_names"]
    variant_indices = {
        variant: geometry_variant_indices(geometry_names, variant)
        for variant in ("G1", "G2", "G3")
    }
    ood = {
        variant: geometry_ood_audit(
            geometry_artifact,
            training_maps=split_maps["train"],
            evaluation_maps=split_maps["test"],
            feature_indices=indices,
        )
        for variant, indices in variant_indices.items()
    }
    runs = []
    for seed in seeds:
        base_model, base_fit = fit_probe(
            arrays,
            mode="physics",
            map_count=len(normalization["training_maps"]),
            hidden_dim=hidden_dim,
            context_dim=context_dim,
            geometry_hidden_dim=geometry_hidden_dim,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=device,
            seed=seed,
        )
        base_prediction = predict_probe(
            base_model, arrays["test"], device=device, batch_size=batch_size
        )
        base_test = probe_metrics(
            base_prediction, arrays["test"]["y"], feature_names, arrays["test"]["map_id"]
        )
        variants = {}
        for variant, indices in variant_indices.items():
            selected = subset_geometry_arrays(arrays, indices)
            selected_shuffled = shuffled_geometry[:, indices]
            residual_model, fit_report = fit_residual_geometry_probe(
                base_model,
                selected,
                context_dim=context_dim,
                geometry_hidden_dim=geometry_hidden_dim,
                residual_hidden_dim=residual_hidden_dim,
                geometry_dropout=geometry_dropout,
                maximum_gate=maximum_gate,
                initial_gate=initial_gate,
                correction_penalty=correction_penalty,
                batch_size=batch_size,
                epochs=epochs,
                patience=patience,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                device=device,
                seed=seed,
            )
            correct_prediction, correct_correction = predict_residual_geometry_probe(
                residual_model, selected["test"], device=device, batch_size=batch_size
            )
            shuffled_prediction, shuffled_correction = predict_residual_geometry_probe(
                residual_model,
                selected["test"],
                device=device,
                batch_size=batch_size,
                geometry_override=selected_shuffled,
            )
            correct_test = probe_metrics(
                correct_prediction,
                selected["test"]["y"],
                feature_names,
                selected["test"]["map_id"],
            )
            shuffled_test = probe_metrics(
                shuffled_prediction,
                selected["test"]["y"],
                feature_names,
                selected["test"]["map_id"],
            )
            variants[variant] = {
                "selected_geometry_features": [geometry_names[index] for index in indices],
                "fit": fit_report,
                "correct_test": correct_test,
                "shuffled_test": shuffled_test,
                "correct_correction": correct_correction,
                "shuffled_correction": shuffled_correction,
                "correct_vs_p0": _comparison(correct_test, base_test),
                "correct_vs_shuffled": _comparison(correct_test, shuffled_test),
            }
        runs.append(
            {
                "seed": seed,
                "base_fit": base_fit,
                "base_test": base_test,
                "variants": variants,
            }
        )
    summaries = {}
    for variant in variant_indices:
        summaries[variant] = {
            "correct_vs_p0": _summary(
                [run["variants"][variant]["correct_vs_p0"] for run in runs]
            ),
            "correct_vs_shuffled": _summary(
                [run["variants"][variant]["correct_vs_shuffled"] for run in runs]
            ),
            "mean_gate": float(
                torch.tensor([run["variants"][variant]["fit"]["gate"] for run in runs]).mean()
            ),
            "mean_correction_rms": float(
                torch.tensor(
                    [
                        run["variants"][variant]["correct_correction"]["correction_rms"]
                        for run in runs
                    ]
                ).mean()
            ),
            "ood_relationship": _ood_improvement_relationship(runs, variant, ood[variant]),
        }
    return {
        "split": split_metadata,
        "split_maps": split_maps,
        "split_samples": {
            split: int(values["y"].shape[0]) for split, values in arrays.items()
        },
        "target_features": list(feature_names),
        "normalization": normalization,
        "geometry_ood": ood,
        "summary": summaries,
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Frozen-P0 residual geometry probe with G1/G2/G3 ablations."
    )
    parser.add_argument("--prepared-root", default="artifacts/prepared_d2los")
    parser.add_argument("--scene-plan", required=True)
    parser.add_argument("--geometry-features", required=True)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument(
        "--protocol", choices=["seen_map_spatial", "unseen_map", "both"], default="both"
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--context-dim", type=int, default=16)
    parser.add_argument("--geometry-hidden-dim", type=int, default=48)
    parser.add_argument("--residual-hidden-dim", type=int, default=96)
    parser.add_argument("--geometry-dropout", type=float, default=0.20)
    parser.add_argument("--maximum-gate", type=float, default=0.50)
    parser.add_argument("--initial-gate", type=float, default=0.05)
    parser.add_argument("--correction-penalty", type=float, default=0.05)
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
    if not 0.0 < args.initial_gate < args.maximum_gate:
        parser.error("--initial-gate must be between zero and --maximum-gate")
    if args.correction_penalty < 0:
        parser.error("--correction-penalty cannot be negative")
    positive = (
        args.hidden_dim,
        args.context_dim,
        args.geometry_hidden_dim,
        args.residual_hidden_dim,
        args.epochs,
        args.patience,
        args.batch_size,
        args.feature_batch_size,
    )
    if min(positive) < 1:
        parser.error("dimensions, epochs, patience, and batch sizes must be positive")
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
            residual_hidden_dim=args.residual_hidden_dim,
            geometry_dropout=args.geometry_dropout,
            maximum_gate=args.maximum_gate,
            initial_gate=args.initial_gate,
            correction_penalty=args.correction_penalty,
            batch_size=args.batch_size,
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
        "diagnostic": "b3_s_geo_lite_residual_geometry_probe",
        "prepared_root": str(Path(args.prepared_root).resolve()),
        "scene_plan": str(Path(args.scene_plan).resolve()),
        "geometry_features": str(Path(args.geometry_features).resolve()),
        "geometry_source_sha256": geometry_artifact["source_sha256"],
        "device": str(device),
        "seeds": seeds,
        "formula": "y_hat = frozen_P0(c) + alpha * tanh(Delta_geo(c, g))",
        "hyperparameters": {
            "geometry_dropout": args.geometry_dropout,
            "maximum_gate": args.maximum_gate,
            "initial_gate": args.initial_gate,
            "correction_penalty": args.correction_penalty,
            "geometry_hidden_dim": args.geometry_hidden_dim,
            "residual_hidden_dim": args.residual_hidden_dim,
        },
        "geometry_variants": {
            "G1": "density, footprint/edge size distributions, compactness",
            "G2": "G1 plus length-weighted edge orientation",
            "G3": "G2 plus 4x4 absolute layout",
        },
        "protocols": results,
        "decision_rule": (
            "Proceed to diffusion only if a residual variant beats frozen P0 and shuffled "
            "geometry across seeds on completely unseen Maps without spatial-target regression."
        ),
    }
    atomic_json_dump(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
