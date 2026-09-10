"""Configuration loading and project path utilities."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge dictionaries without mutating either input."""
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML config and resolve an optional local ``extends`` file."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    parent = config.pop("extends", None)
    if parent:
        parent_path = config_path.parent / parent
        return deep_merge(load_config(parent_path), config)
    return config


def project_path(value: str | Path) -> Path:
    """Resolve a project-relative path against the repository root."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path
