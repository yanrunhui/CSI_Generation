#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _flatten(value: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(_flatten(item, name))
        elif isinstance(item, (str, int, float, bool)) or item is None:
            result[name] = item
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine Model B/C evaluation reports.")
    parser.add_argument("--model-b", required=True)
    parser.add_argument("--model-c", required=True)
    parser.add_argument("--output-prefix", default="artifacts/reports/model_B_vs_model_C")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    with Path(args.model_b).open("r", encoding="utf-8") as stream:
        model_b = json.load(stream)
    with Path(args.model_c).open("r", encoding="utf-8") as stream:
        model_c = json.load(stream)
    payload = {
        "model_B": model_b,
        "model_C": model_c,
        "comparability": {
            "directly_comparable": [
                "CSI summary MMD", "CSI summary sliced Wasserstein", "CSI diversity"
            ],
            "model_B_only": ["exact path-condition accuracy", "constraint projection change"],
            "estimator_dependent": ["Model C path count", "Model C K-factor", "Model C LoS"],
        },
    }
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with prefix.with_suffix(".json").open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    rows = [
        {"model": "model_B", **_flatten(model_b)},
        {"model": "model_C", **_flatten(model_c)},
    ]
    fields = sorted({key for row in rows for key in row})
    with prefix.with_suffix(".csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    if not args.no_plot:
        import matplotlib.pyplot as plt

        metric_names = (
            "csi_summary_mmd_rbf",
            "csi_summary_sliced_wasserstein",
            "generated_pairwise_diversity",
        )
        labels = ("CSI MMD", "Sliced Wasserstein", "Diversity")
        model_b_values = [model_b["distribution"][name] for name in metric_names]
        model_c_values = [model_c["distribution"][name] for name in metric_names]
        positions = list(range(len(labels)))
        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.bar([value - 0.2 for value in positions], model_b_values, 0.4, label="Model B")
        axis.bar([value + 0.2 for value in positions], model_c_values, 0.4, label="Model C")
        axis.set_xticks(positions, labels)
        axis.set_ylabel("metric value (lower is better except diversity)")
        axis.set_title("Conditional diffusion generator comparison")
        axis.legend()
        figure.tight_layout()
        figure.savefig(prefix.with_suffix(".png"), dpi=180)
        plt.close(figure)


if __name__ == "__main__":
    main()
