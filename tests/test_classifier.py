"""M4 leakage classifier acceptance on SYNTHETIC data (SPEC section 7)."""
import numpy as np
import pytest

from usintent.leakage.classifier import (
    build_dataset,
    evaluate,
    evaluate_remaining,
    permutation_test,
    permutation_test_remaining,
)
from usintent.sim.holdworld import HoldWorldConfig, generate
from usintent.whenhead.features import coupling_indicator, force_variance, pooled_embedding


def _data(leak: bool):
    eps = generate(HoldWorldConfig(leak=leak, image_shape=(24, 24), fps=10.0, seed=5), 60, sessions=6)
    return build_dataset(eps, horizon_s=1.0, subsample_s=0.5, coupling_threshold=60.0)


@pytest.fixture(scope="module")
def leak_off():
    return _data(False)


@pytest.fixture(scope="module")
def leak_on():
    return _data(True)


def test_leak_off_no_improvement(leak_off) -> None:
    res = permutation_test(leak_off, n_perm=200)
    assert res["p_value"] > 0.2
    tab = evaluate(leak_off, models=("logreg",)).set_index("features")
    assert tab.loc["E+ctx+obs", "auroc"] < tab.loc["E", "auroc"] + 0.02


def test_leak_on_clear_improvement(leak_on) -> None:
    res = permutation_test(leak_on, n_perm=200)
    assert res["p_value"] < 0.01
    tab = evaluate(leak_on).set_index(["model", "features"])
    for m in ("logreg", "gbm"):
        assert tab.loc[(m, "E+ctx+obs"), "logloss"] < tab.loc[(m, "E"), "logloss"] - 0.05


@pytest.mark.slow
def test_full_1000_permutations(leak_off, leak_on) -> None:
    assert permutation_test(leak_off, n_perm=1000)["p_value"] > 0.2
    assert permutation_test(leak_on, n_perm=1000)["p_value"] < 0.01


def test_dataset_structure(leak_off) -> None:
    d = leak_off
    assert np.all(d.e >= 0) and np.all((d.remaining_s > 1.0) | d.y)
    assert np.array_equal(d.y, d.remaining_s <= 1.0)
    assert len(np.unique(d.groups)) == 6
    assert d.features("E+ctx+obs").shape[1] == 1 + 3 + d.obs.shape[1]


def test_feature_functions_known() -> None:
    f = np.zeros((20, 10), np.uint8)
    f[:, :4] = 200
    assert coupling_indicator(f, 100.0) == pytest.approx(0.4)
    fz = np.array([1.0, 1.0, 3.0, 3.0])
    v = force_variance(fz, np.array([0.0, 0.5, 1.0, 1.5]), window_s=0.6)
    assert v[0] == 0.0 and v[1] == 0.0 and v[2] == pytest.approx(1.0) and v[3] == 0.0
    emb = pooled_embedding(np.full((8, 8), 255, np.uint8), grid=2)
    assert np.allclose(emb, 1.0) and emb.shape == (4,)


def test_remaining_time_regression(leak_off, leak_on) -> None:
    off = permutation_test_remaining(leak_off, n_perm=100)
    on = permutation_test_remaining(leak_on, n_perm=100)
    assert off["p_value"] > 0.2 and on["p_value"] < 0.02
    tab = evaluate_remaining(leak_on).set_index(["model", "features"])
    # the synthetic leak exists only in the last 1.5 s of a hold, which bounds the MAE gain
    assert tab.loc[("gbm", "E+ctx+obs"), "mae_s"] < tab.loc[("gbm", "E"), "mae_s"] - 0.1
