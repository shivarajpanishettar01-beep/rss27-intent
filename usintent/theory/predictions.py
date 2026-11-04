"""Closed-form / Monte-Carlo predictions from the paper (Eqs. 1-3, Fig. 1c).

Golden values are tested in tests/test_theory.py against reference/rss_predictions.py.
"""
from __future__ import annotations

import numpy as np

from usintent.theory.distributions import HoldDistribution

_DEFAULT_N = 2_000_000


def _draw(dist: HoldDistribution, n: int, seed: int) -> np.ndarray:
    return dist.sample(n, np.random.default_rng(seed))


def hazard(dist: HoldDistribution, e: float, h: float, n: int = _DEFAULT_N, seed: int = 0) -> float:
    """P(D <= e+h | D > e), Eq. 1."""
    d = _draw(dist, n, seed)
    alive = d > e
    if not np.any(alive):
        return float("nan")
    return float(np.mean(d[alive] <= e + h))


def bound_median(dist: HoldDistribution, n: int = _DEFAULT_N, seed: int = 0) -> float:
    """E|D - med(D)|, Eq. 2 (lower bound for intent-blind timing error)."""
    d = _draw(dist, n, seed)
    return float(np.mean(np.abs(d - np.median(d))))


def err_calibrated_imitator(dist: HoldDistribution, n: int = _DEFAULT_N, seed: int = 0) -> float:
    """E|D - D'| with D, D' i.i.d. (imitator that knows e and samples T ~ p_D)."""
    d1 = _draw(dist, n, seed)
    d2 = _draw(dist, n, seed + 1)
    return float(np.mean(np.abs(d1 - d2)))


def err_memoryless_hedger(
    dist: HoldDistribution, n: int = _DEFAULT_N, seed: int = 0
) -> tuple[float, float]:
    """(E|T - D|, P(T < D)) with T ~ Exp(mean=E[D]) independent of D (idealized)."""
    rng = np.random.default_rng(seed)
    d = dist.sample(n, rng)
    t = rng.exponential(float(np.mean(d)), n)
    return float(np.mean(np.abs(t - d))), float(np.mean(t < d))


def pibar(dist: HoldDistribution, horizon: float, n: int = _DEFAULT_N, seed: int = 0) -> float:
    """E[min(H, D)] / E[D], Eq. 3."""
    d = _draw(dist, n, seed)
    return float(np.mean(np.minimum(horizon, d)) / np.mean(d))


def pibar_tau(
    dist: HoldDistribution, horizon: float, tau: float, n: int = _DEFAULT_N, seed: int = 0
) -> float:
    """E[max(0, min(H, D) - tau)] / E[D]: rate of threshold-crossing departure chunks."""
    d = _draw(dist, n, seed)
    return float(np.mean(np.maximum(0.0, np.minimum(horizon, d) - tau)) / np.mean(d))
