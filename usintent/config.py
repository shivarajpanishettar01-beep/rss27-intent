"""YAML config loading with explicit handling of unknown rig facts.

Unknown values are written as `null` in configs/*.yaml. Code that needs one
calls `require(cfg, "dotted.key")`, which raises MissingRigFact instead of
silently using a guess (CLAUDE.md hard rule 1).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from usintent.paths import CONFIGS


class MissingRigFact(RuntimeError):
    """Raised when a required config value is still null."""


def load(name: str | Path) -> dict[str, Any]:
    path = Path(name)
    if not path.is_absolute():
        path = CONFIGS / path
    if path.suffix == "":
        path = path.with_suffix(".yaml")
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a mapping")
    return data


def require(cfg: dict[str, Any], dotted: str) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"config key missing: {dotted}")
        node = node[part]
    if node is None:
        raise MissingRigFact(
            f"'{dotted}' is null. Measure it, record it in docs/RIG_FACTS.md, then set it in configs/."
        )
    return node
