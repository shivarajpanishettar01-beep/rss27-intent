"""Session/day splits; assert no session crosses splits. SPEC section 4, CLAUDE.md rule 5.

The grouping unit is the recording *day*: all sessions of one day go to one split, so neither a
session nor a day can appear in two splits. Never split by episode or frame.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SPLITS = ("train", "val", "test")


class SplitLeakageError(AssertionError):
    """A session or day appears in more than one split."""


def assign_splits(days: Iterable[str], fractions: dict[str, float], seed: int) -> dict[str, str]:
    """Deterministically assign each day to a split; returns {day: split}.

    Days are shuffled with ``seed`` and cut by cumulative ``fractions`` (rounded, each listed
    split with a positive fraction gets at least one day when enough days exist).
    """
    uniq = sorted(set(days))
    if abs(sum(fractions.values()) - 1.0) > 1e-9 or set(fractions) - set(SPLITS):
        raise ValueError(f"fractions must be over {SPLITS} and sum to 1")
    order = list(np.random.default_rng(seed).permutation(uniq))
    names = [s for s in SPLITS if fractions.get(s, 0) > 0]
    counts = [max(1, round(fractions[s] * len(uniq))) if len(uniq) >= len(names) else 0
              for s in names]
    while sum(counts) > len(uniq):
        counts[int(np.argmax(counts))] -= 1
    counts[0] += len(uniq) - sum(counts)
    out, i = {}, 0
    for name, c in zip(names, counts):
        for d in order[i:i + c]:
            out[str(d)] = name
        i += c
    return out


def check_no_leakage(table: pd.DataFrame) -> None:
    """``table`` has columns session_id, day, split (one row per episode or step).

    Raises SplitLeakageError if any session or day is in two splits.
    """
    for unit in ("session_id", "day"):
        n = table.groupby(unit)["split"].nunique()
        bad = n[n > 1]
        if len(bad):
            raise SplitLeakageError(
                f"{unit}(s) in more than one split: {sorted(bad.index.astype(str).tolist())}")


def episode_split_table(episodes: Iterable[object], day_to_split: dict[str, str]) -> pd.DataFrame:
    """Per-episode table (episode_index, session_id, day, split) for Episode-like objects."""
    rows = [{"episode_index": e.episode_index, "session_id": e.session_id, "day": e.day,  # type: ignore[attr-defined]
             "split": day_to_split[e.day]} for e in episodes]  # type: ignore[attr-defined]
    table = pd.DataFrame(rows)
    check_no_leakage(table)
    return table


def save_assignment(day_to_split: dict[str, str], path: Path) -> None:
    Path(path).write_text(yaml.safe_dump({"by": "day", "assignment": day_to_split}, sort_keys=True))


def load_assignment(path: Path) -> dict[str, str]:
    data = yaml.safe_load(Path(path).read_text())
    return {str(k): str(v) for k, v in data["assignment"].items()}
