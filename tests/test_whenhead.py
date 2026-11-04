"""M7 when-head acceptance on SYNTHETIC data (SPEC section 9)."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from usintent.sim.holdworld import HoldWorldConfig, generate
from usintent.theory import Uniform, bound_median
from usintent.whenhead.commit import CommitRule, commit_index, commit_times
from usintent.whenhead.confidence import karamalis_confidence, measure_runtime, proxy_confidence
from usintent.whenhead.features import FeatureParams, hold_bins
from usintent.whenhead.model import WhenHead, image_share, survival_nll

DELTA = 0.1            # synthetic bin width (= policy period) at 10 fps
P = FeatureParams(DELTA, coupling_threshold=60.0, confidence_impl="proxy")
# attenuation 150 makes the adequacy leak visible in the vessel-ROI confidence (synthetic choice)
CFG = HoldWorldConfig(image_shape=(32, 32), fps=10.0, attenuation=150.0)
HIDDEN = [32, 32]
BOUND = bound_median(Uniform(2, 10))


def _fit_eval(leak: bool):
    tr = hold_bins(generate(replace(CFG, leak=leak, seed=1), 250), P)
    te = hold_bins(generate(replace(CFG, leak=leak, seed=2), 150), P)
    full = WhenHead(HIDDEN).fit(tr)
    eonly = WhenHead(HIDDEN, image=False).fit(tr)
    return tr, te, full, eonly


@pytest.fixture(scope="module")
def leak_off():
    return _fit_eval(False)


@pytest.fixture(scope="module")
def leak_on():
    return _fit_eval(True)


def test_leak_off_share_and_timing(leak_off) -> None:
    tr, te, full, _ = leak_off
    assert te["hold_key"].nunique() > 600
    assert image_share(full, tr) < 0.05
    ct = commit_times(full, te, DELTA)
    assert ct["T"].notna().all()
    assert ct["err"].mean() == pytest.approx(BOUND, abs=0.1)


def test_leak_on_share_and_timing(leak_on) -> None:
    tr, te, full, eonly = leak_on
    assert image_share(full, tr) > 0.2
    ct = commit_times(full, te, DELTA)
    assert ct["err"].mean() < BOUND
    assert eonly.nll_per_hold(te) - full.nll_per_hold(te) > 0.1      # log-likelihood gain


def test_censoring_handled() -> None:
    """Truncated episodes: censored holds contribute no event; the hazard fit stays unbiased."""
    rng = np.random.default_rng(0)
    cfg = replace(CFG, seed=7, n_stations=(1, 1), approach_s=0.5)
    from usintent.sim.holdworld import generate_episode

    eps = []
    for i in range(500):
        cut = 0.5 + rng.uniform(1.0, 14.0)            # random end: many holds censored
        eps.append(generate_episode(cfg, i, rng, first_hold_id=i, truncate_at_s=cut))
    bins = hold_bins(eps, P)
    cens = bins.groupby("hold_key")["censored_hold"].first()
    assert 0.3 < cens.mean() < 0.8
    assert bins.loc[bins["censored_hold"], "event"].sum() == 0
    good = WhenHead(HIDDEN, image=False).fit(bins)
    naive_bins = bins.copy()
    last = naive_bins.groupby("hold_key")["k"].transform("max") == naive_bins["k"]
    naive_bins.loc[naive_bins["censored_hold"] & last, "event"] = 1   # wrong: censoring as events
    naive = WhenHead(HIDDEN, image=False).fit(naive_bins)
    grid = pd.DataFrame({"e": np.arange(0, 12, DELTA), "kappa": 0.0, "s": 1.0, "f": 0.0})
    med_good = grid["e"].iloc[commit_index(good.hazards(grid))]
    med_naive = grid["e"].iloc[commit_index(naive.hazards(grid))]
    assert med_good == pytest.approx(Uniform(2, 10).median(), abs=0.4)
    assert med_naive < med_good - 0.5


def test_survival_nll_known() -> None:
    h = np.array([0.5, 0.5, 0.2, 0.2])
    y = np.array([0.0, 1.0, 0.0, 0.0])                 # hold A: event in bin 1; hold B: censored
    keys = np.array(["a", "a", "b", "b"])
    expect = -(np.log(0.5) + np.log(0.5) + 2 * np.log(0.8)) / 2
    assert survival_nll(h, y, keys) == pytest.approx(expect)


def test_commit_rule_online_matches_offline(leak_on) -> None:
    _, te, full, _ = leak_on
    grp = te[te["hold_key"] == te["hold_key"].iloc[0]].sort_values("k")
    k_off = commit_index(full.hazards(grp))
    rule = CommitRule(full)
    out = [rule.step(r[["e", "kappa", "s", "f"]].to_dict()) for _, r in grp.iterrows()]
    first = next((i for i, (m, _) in enumerate(out) if m != "HOLD"), -1)
    assert first == k_off
    assert all(zero for m, zero in out[: max(first, 0)])            # zero motion while HOLD
    assert commit_index(np.array([0.1, 0.3, 0.5])) == 2 and commit_index(np.zeros(3)) == -1


def test_confidence_maps_properties_and_runtime() -> None:
    ep = generate(replace(CFG, seed=3, n_stations=(2, 2), p_lift=0.0), 1)[0]
    frame = ep.bmode[np.flatnonzero(ep.hold_id >= 0)[5]]
    c = karamalis_confidence(frame)
    assert c.shape == (32, 32) and np.all((c >= 0) & (c <= 1))
    assert np.allclose(c[0], 1.0) and np.allclose(c[-1], 0.0)
    prof = c.mean(axis=1)
    assert np.all(np.diff(prof) <= 1e-9)                              # decreases with depth
    assert proxy_confidence(frame).shape == frame.shape
    rep = measure_runtime(ep.bmode[:10], "karamalis", (32, 32))
    assert rep.n_frames == 10 and 0 < rep.ms_per_frame_median < 1e4
