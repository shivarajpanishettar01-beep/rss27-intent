"""Single source of truth for every filesystem path in the repo.

Rules (see docs/LAYOUT.md):
- Never hard-code paths elsewhere; import from here.
- Large/private data lives outside git; point to it with the env var USINTENT_DATA.
"""
from __future__ import annotations

import os
from pathlib import Path


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "usintent").is_dir():
            return parent
    raise RuntimeError("Repo root not found: is usintent installed with `pip install -e .`?")


REPO_ROOT: Path = _find_repo_root()
CONFIGS: Path = REPO_ROOT / "configs"
ARM_CONFIGS: Path = CONFIGS / "arms"
REFERENCE: Path = REPO_ROOT / "reference"
REPORTS: Path = REPO_ROOT / "reports"
FIGURES: Path = REPO_ROOT / "figures"
SCRIPTS: Path = REPO_ROOT / "scripts"
ROS2_WS: Path = REPO_ROOT / "ros2_ws"
DATA: Path = Path(os.environ.get("USINTENT_DATA", REPO_ROOT / "data"))
RESULTS: Path = Path(os.environ.get("USINTENT_RESULTS", REPO_ROOT / "results"))
SYNTHETIC_DATA: Path = DATA / "synthetic"
OPENH_DATA: Path = DATA / "openh"
CUED_DATA: Path = DATA / "cued"
RAW_BAGS: Path = DATA / "bags"
