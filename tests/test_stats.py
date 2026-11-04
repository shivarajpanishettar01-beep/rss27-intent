"""Statistics helpers with known answers (SPEC section 11)."""
import numpy as np
import pytest

from usintent.stats.tests import (
    cluster_bootstrap,
    effect_vs_noise_floor,
    paired_cluster_bootstrap,
    permutation_test_runs,
    rank_correlation,
    seed_noise_floor,
    slope_with_permutation,
)


def test_noise_floor_known() -> None:
    f = seed_noise_floor(np.array([1.0, 2.0, 3.0]))
    assert f["std"] == pytest.approx(1.0) and f["range"] == 2.0
    assert effect_vs_noise_floor(np.array([3.0]), np.array([1.0]), 1.0) == 2.0


def test_exact_permutation_known() -> None:
    # 3 vs 3 fully separated: only 2 of 20 labelings are as extreme -> p = 0.1
    r = permutation_test_runs(np.array([10.0, 11, 12]), np.array([0.0, 1, 2]))
    assert r["exact"] == 1.0 and r["p_value"] == pytest.approx(0.1)
    r = permutation_test_runs(np.array([1.0, 2, 3]), np.array([1.0, 2, 3]))
    assert r["p_value"] == 1.0


def test_cluster_bootstrap_covers_and_respects_clusters() -> None:
    rng = np.random.default_rng(0)
    clusters = np.repeat([f"t{i}" for i in range(40)], 10)
    trial_effect = np.repeat(rng.normal(0, 1, 40), 10)
    v = 5.0 + trial_effect + rng.normal(0, 0.1, 400)
    ci = cluster_bootstrap(v, clusters)
    assert ci["lo"] < 5.0 < ci["hi"] and ci["n_clusters"] == 40
    naive = cluster_bootstrap(v, np.arange(400).astype(str))
    assert (ci["hi"] - ci["lo"]) > 2 * (naive["hi"] - naive["lo"])   # clustering widens the CI
    d = paired_cluster_bootstrap(v + 1.0, v, clusters)
    assert d["estimate"] == pytest.approx(1.0) and d["lo"] == pytest.approx(1.0)


def test_rank_and_slope_known() -> None:
    x = np.arange(10.0)
    assert rank_correlation(x, x**3)["rho"] == pytest.approx(1.0)
    s = slope_with_permutation(x, 2 * x + 1, n_perm=2000)
    assert s["slope"] == pytest.approx(2.0) and s["p_value"] < 0.01
    assert s["null_lo"] < 0 < s["null_hi"]
