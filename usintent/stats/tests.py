"""Seed noise floor, permutation tests, cluster bootstrap, rank correlation. SPEC section 11.

The unit of analysis for a training configuration is the run (seed), never the episode
(CLAUDE.md rule 6). In closed loop the resampling unit is the whole trial.
"""
from __future__ import annotations

import itertools
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy import stats


def seed_noise_floor(values: NDArray[np.float64]) -> dict[str, float]:
    """Spread of one arm's metric across seeds: sample std, range and n (one value per seed)."""
    v = np.asarray(values, dtype=np.float64)
    if v.size < 2:
        raise ValueError("need at least 2 seeds")
    return {"n": float(v.size), "mean": float(v.mean()), "std": float(v.std(ddof=1)),
            "range": float(v.max() - v.min())}


def permutation_test_runs(a: NDArray[np.float64], b: NDArray[np.float64], n_perm: int = 10000,
                          seed: int = 0) -> dict[str, float]:
    """Two-sided permutation test on the difference of means between two arms (one value per
    run). Exact enumeration when the number of relabelings is <= n_perm."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    pooled = np.concatenate([a, b])
    obs = float(a.mean() - b.mean())
    n_a, n = len(a), len(pooled)
    n_comb = 1
    for k in range(n_a):
        n_comb = n_comb * (n - k) // (k + 1)
    if n_comb <= n_perm:
        diffs = []
        for idx in itertools.combinations(range(n), n_a):
            m = np.zeros(n, bool)
            m[list(idx)] = True
            diffs.append(pooled[m].mean() - pooled[~m].mean())
        d = np.asarray(diffs)
        p = float(np.mean(np.abs(d) >= abs(obs) - 1e-12))
        exact = 1.0
    else:
        rng = np.random.default_rng(seed)
        d = np.empty(n_perm)
        for i in range(n_perm):
            perm = rng.permutation(pooled)
            d[i] = perm[:n_a].mean() - perm[n_a:].mean()
        p = float((1 + np.sum(np.abs(d) >= abs(obs) - 1e-12)) / (1 + n_perm))
        exact = 0.0
    return {"diff": obs, "p_value": p, "exact": exact}


def effect_vs_noise_floor(a: NDArray[np.float64], b: NDArray[np.float64],
                          floor_std: float) -> float:
    """Difference of arm means in units of the seed noise floor (A1 std across seeds)."""
    return float((np.mean(a) - np.mean(b)) / floor_std) if floor_std > 0 else float("inf")


def cluster_bootstrap(values: NDArray[np.float64], clusters: NDArray[np.str_],
                      stat: Callable[[NDArray[np.float64]], float] = np.mean,
                      n_boot: int = 2000, seed: int = 0, ci: float = 0.95) -> dict[str, float]:
    """Bootstrap CI resampling whole clusters (trials or sessions) with replacement."""
    v = np.asarray(values, dtype=np.float64)
    c = np.asarray(clusters)
    uniq = np.unique(c)
    groups = [v[c == u] for u in uniq]
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        boots[i] = stat(np.concatenate([groups[j] for j in pick]))
    lo, hi = np.quantile(boots, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return {"estimate": float(stat(v)), "lo": float(lo), "hi": float(hi),
            "n_clusters": float(len(groups))}


def paired_cluster_bootstrap(a: NDArray[np.float64], b: NDArray[np.float64],
                             clusters: NDArray[np.str_], n_boot: int = 2000, seed: int = 0,
                             ci: float = 0.95) -> dict[str, float]:
    """CI of mean(a - b) for paired observations (same holds/station sequences), by cluster."""
    return cluster_bootstrap(np.asarray(a) - np.asarray(b), clusters, n_boot=n_boot, seed=seed,
                             ci=ci)


def rank_correlation(x: NDArray[np.float64], y: NDArray[np.float64]) -> dict[str, float]:
    """Spearman rank correlation (e.g. open-loop hold metric vs closed-loop timing per run)."""
    r = stats.spearmanr(x, y)
    return {"rho": float(r.statistic), "p_value": float(r.pvalue), "n": float(len(x))}


def slope_with_permutation(x: NDArray[np.float64], y: NDArray[np.float64], n_perm: int = 10000,
                           seed: int = 0, ci: float = 0.95) -> dict[str, float]:
    """OLS slope of y on x (one point per run) with a permutation null interval and p-value."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    def slope(yy: NDArray[np.float64]) -> float:
        xc = x - x.mean()
        return float(np.sum(xc * (yy - yy.mean())) / np.sum(xc * xc))

    obs = slope(y)
    rng = np.random.default_rng(seed)
    null = np.array([slope(rng.permutation(y)) for _ in range(n_perm)])
    lo, hi = np.quantile(null, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    p = float((1 + np.sum(np.abs(null) >= abs(obs))) / (1 + n_perm))
    return {"slope": obs, "null_lo": float(lo), "null_hi": float(hi), "p_value": p}
