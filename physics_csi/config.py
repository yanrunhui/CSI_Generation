from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return payload


def deep_merge(*configs: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for config in configs:
        _merge_into(result, config)
    return result


def _merge_into(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge_into(target[key], value)
        else:
            target[key] = deepcopy(value)


def load_project_config(config_dir: str | Path = "configs") -> dict[str, Any]:
    root = Path(config_dir)
    return deep_merge(
        load_yaml(root / "data.yaml"),
        load_yaml(root / "model.yaml"),
        load_yaml(root / "train.yaml"),
        load_yaml(root / "evaluation.yaml"),
    )

