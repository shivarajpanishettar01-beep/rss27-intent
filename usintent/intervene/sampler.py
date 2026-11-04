"""Weighted chunk sampler setting the departure fraction of hold chunks (H2 causal test). SPEC 8. M6.

Three groups of chunk starts in a ChunkSampler index: A = hold chunks containing a departure,
B = hold chunks without one, C = all other chunks. Natural sampling is uniform over chunk starts.
The intervention keeps every chunk (no deletion), keeps the total probability mass of holds and
of C at their natural values, and splits the hold mass between A and B so that
P(A | hold) = p:
    w_A = p (n_A + n_B) / n_A,   w_B = (1 - p)(n_A + n_B) / n_B,   w_C = 1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass(frozen=True)
class InterventionWeights:
    weights: NDArray[np.float64]      # one per index row, sums to len(index)
    natural_fraction: float           # n_A / (n_A + n_B)
    target_fraction: float


def natural_departure_fraction(index: pd.DataFrame) -> float:
    hold = index["phase"] == "hold"
    return float(index.loc[hold, "contains_departure"].mean())


def intervention_weights(index: pd.DataFrame, target_fraction: float) -> InterventionWeights:
    """Per-row sampling weights giving P(departure | hold chunk) = target_fraction."""
    if not 0.0 < target_fraction < 1.0:
        raise ValueError("target fraction must be in (0, 1)")
    hold = (index["phase"] == "hold").to_numpy()
    dep = index["contains_departure"].to_numpy(dtype=bool) & hold
    n_a, n_b = int(dep.sum()), int((hold & ~dep).sum())
    if n_a == 0 or n_b == 0:
        raise ValueError("need hold chunks with and without departures")
    w = np.ones(len(index))
    w[dep] = target_fraction * (n_a + n_b) / n_a
    w[hold & ~dep] = (1.0 - target_fraction) * (n_a + n_b) / n_b
    return InterventionWeights(w, n_a / (n_a + n_b), target_fraction)


def epoch_indices(weights: NDArray[np.float64], rng: np.random.Generator,
                  n: int | None = None) -> NDArray[np.int64]:
    """One epoch of chunk-start row positions drawn with replacement by weight (default: as
    many draws as rows)."""
    n = len(weights) if n is None else n
    return np.asarray(rng.choice(len(weights), size=n, replace=True, p=weights / weights.sum()))


def realized_fraction(index: pd.DataFrame, picks: NDArray[np.int64]) -> float:
    """Fraction of drawn hold chunks that contain a departure."""
    rows = index.iloc[picks]
    hold = rows["phase"] == "hold"
    return float(rows.loc[hold, "contains_departure"].mean())
