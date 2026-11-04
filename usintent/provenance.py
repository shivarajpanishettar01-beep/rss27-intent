"""Provenance stamps for every figure/table (CLAUDE.md hard rule 7, SPEC section 13).

Every output of ``scripts/make_*.py`` embeds ``stamp(...)``: git hash, dirty flag, a hash of the
configs that were read, the data source label and the UTC time. Synthetic outputs carry
``"data_source": "synthetic"`` so they can never be mistaken for experimental results.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from usintent.config import load
from usintent.paths import REPO_ROOT

DATA_SOURCES = frozenset({"theory", "synthetic", "openh", "cued", "lab"})


def git_hash() -> str:
    """Current commit hash, suffixed with '+dirty' if tracked files are modified."""
    try:
        h = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--"],
            check=False,
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
        ).returncode
    except (OSError, subprocess.CalledProcessError):
        return "unknown (not a git checkout)"
    return h + ("+dirty" if dirty else "")


def config_hash(names: list[str]) -> str:
    """sha256 over the canonical JSON of the named configs (e.g. ["protocol", "arms/A1"])."""
    blob = json.dumps({n: load(n) for n in sorted(names)}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def stamp(config_names: list[str], data_source: str, **extra: Any) -> dict[str, Any]:
    """Provenance dict to embed in a figure/table JSON."""
    if data_source not in DATA_SOURCES:
        raise ValueError(f"data_source must be one of {sorted(DATA_SOURCES)}")
    out: dict[str, Any] = {
        "git_hash": git_hash(),
        "config_hash": config_hash(config_names),
        "configs": sorted(config_names),
        "data_source": data_source,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if data_source == "synthetic":
        out["warning"] = "SYNTHETIC: test-oracle output, not an experimental result"
    out.update(extra)
    return out


def output_dir(data_source: str, requested: Path | None, default: Path) -> Path:
    """Where a make_* script may write. Synthetic outputs must go to an explicit directory
    outside figures/ and results/ (they must never feed a paper figure)."""
    from usintent.paths import FIGURES, RESULTS

    if data_source == "synthetic":
        if requested is None:
            raise ValueError("synthetic runs need an explicit --out outside figures/ and results/")
        req = Path(requested).resolve()
        for forbidden in (FIGURES.resolve(), RESULTS.resolve()):
            if req == forbidden or forbidden in req.parents:
                raise ValueError(f"synthetic outputs may not be written under {forbidden}")
        return req
    return Path(requested) if requested is not None else default


def data_hash(root: Path) -> str:
    """sha256 over a LeRobot dataset's metadata files (info, episodes, stats, holds sidecar)."""
    root = Path(root)
    h = hashlib.sha256()
    for f in sorted((root / "meta").rglob("*")):
        if f.is_file():
            h.update(f.relative_to(root).as_posix().encode())
            h.update(f.read_bytes())
    return h.hexdigest()[:16]
