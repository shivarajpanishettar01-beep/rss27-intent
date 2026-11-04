"""Closed-form / Monte-Carlo predictions from the paper (Eqs. 1-3, Fig. 1c).

Golden values are tested in tests/test_theory.py against reference/rss_predictions.py.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from usintent.theory.distributions import HoldDistribution, ShiftedExp, Uniform

_DEFAULT_N = 2_000_000


def _draw(dist: HoldDistribution, n: int, seed: int) -> NDArray[np.float64]:
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


# ---------------------------------------------------------------------------------------------
# Closed forms (SPEC section 2: "provide closed forms where they exist"). Tested against the
# Monte-Carlo estimators above in tests/test_theory.py.
# ---------------------------------------------------------------------------------------------


def bound_median_closed(dist: HoldDistribution) -> float:
    """E|D - med(D)| in closed form: (b-a)/4 for Uniform, scale*ln 2 for ShiftedExp."""
    if isinstance(dist, Uniform):
        return (dist.b - dist.a) / 4.0
    if isinstance(dist, ShiftedExp):
        return dist.scale * float(np.log(2.0))
    raise TypeError(f"no closed form for {type(dist).__name__}")


def err_calibrated_imitator_closed(dist: HoldDistribution) -> float:
    """E|D - D'| in closed form: (b-a)/3 for Uniform, scale for ShiftedExp (Laplace difference)."""
    if isinstance(dist, Uniform):
        return (dist.b - dist.a) / 3.0
    if isinstance(dist, ShiftedExp):
        return dist.scale
    raise TypeError(f"no closed form for {type(dist).__name__}")


def err_memoryless_hedger_closed(dist: HoldDistribution) -> tuple[float, float]:
    """(E|T - D|, P(T < D)) with T ~ Exp(mu = E[D]) independent of D.

    For fixed d, E|T - d| = d - mu + 2 mu exp(-d/mu). Averaging over D with mu = E[D] gives
    E|T - D| = 2 mu L(1/mu), with L the Laplace transform of D; P(T < D) = 1 - L(1/mu).
    """
    mu = dist.mean()
    lap = dist.laplace(1.0 / mu)
    return 2.0 * mu * lap, 1.0 - lap


def pibar_closed(dist: HoldDistribution, horizon: float) -> float:
    """E[min(H, D)]/E[D] in closed form for Uniform and ShiftedExp (any H >= 0)."""
    if isinstance(dist, Uniform):
        a, b, h = dist.a, dist.b, horizon
        if h <= a:
            emin = h
        elif h >= b:
            emin = dist.mean()
        else:  # integral of survival function from 0 to H
            emin = a + ((h - a) - (h - a) ** 2 / (2 * (b - a)))
        return emin / dist.mean()
    if isinstance(dist, ShiftedExp):
        s, m, h = dist.shift, dist.scale, horizon
        emin = h if h <= s else s + m * (1.0 - float(np.exp(-(h - s) / m)))
        return emin / dist.mean()
    raise TypeError(f"no closed form for {type(dist).__name__}")


def pibar_tau_closed(dist: HoldDistribution, horizon: float, tau: float) -> float:
    """(H - tau)/E[D], valid when P(D >= H) = 1 (both protocol distributions with H <= 2 s)."""
    if float(dist.cdf(horizon)) > 0.0:
        raise ValueError("closed form (H - tau)/E[D] needs D >= H almost surely")
    return max(0.0, horizon - tau) / dist.mean()


def hazard_closed(dist: HoldDistribution, e: float, h: float) -> float:
    """P(D <= e+h | D > e) from the CDF (exact for any distribution with a cdf)."""
    surv = 1.0 - float(dist.cdf(e))
    if surv <= 0.0:
        return float("nan")
    return (float(dist.cdf(e + h)) - float(dist.cdf(e))) / surv
