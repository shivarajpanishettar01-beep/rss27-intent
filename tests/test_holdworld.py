"""M1 acceptance tests for the synthetic hold-world (SPEC section 3). All data here is SYNTHETIC."""
from dataclasses import replace

import numpy as np
import pytest
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from usintent.data.chunks import chunks_from_poses, integrate_actions
from usintent.data.loaders import holds_table, read_lerobot, write_lerobot
from usintent.sim.holdworld import (
    HoldWorldConfig,
    departure_tau,
    generate,
    generate_holds,
    sweep_profile,
)
from usintent.theory import ShiftedExp, Uniform, pibar, pibar_tau

H = 1.0
DISTS = {"uniform": Uniform(2, 10), "shifted_exp": ShiftedExp(2, 4)}


def _hold_chunks(episodes, horizon):
    """For every frame inside a hold (start <= t < key): (contains_departure, max |dxy| of GT chunk)."""
    contains, maxdisp, speeds = [], [], []
    for ep in episodes:
        n = round(horizon * ep.fps)
        idx = np.flatnonzero(ep.hold_id >= 0)
        idx = idx[idx + n < len(ep)]
        key = ep.holds.set_index("hold_id").loc[ep.hold_id[idx], "key_time"].to_numpy()
        v = ep.holds.set_index("hold_id").loc[ep.hold_id[idx], "gt_exit_speed_mps"].to_numpy()
        ch = chunks_from_poses(ep.state[:, :6], idx, n)
        contains.append(key < ep.time[idx] + horizon)
        maxdisp.append(np.linalg.norm(ch[:, :, :2], axis=2).max(axis=1))
        speeds.append(v)
    return np.concatenate(contains), np.concatenate(maxdisp), np.concatenate(speeds)


@pytest.fixture(scope="module", params=list(DISTS))
def big_world(request):
    name = request.param
    cfg = HoldWorldConfig(hold_dist=DISTS[name], distribution_name=name, render_images=False, seed=11)
    eps = generate_holds(cfg, 10_000)
    return name, cfg, eps


def test_departure_fraction_equals_pibar(big_world) -> None:
    name, _, eps = big_world
    assert len(holds_table(eps)) >= 10_000
    contains, maxdisp, _ = _hold_chunks(eps, H)
    assert contains.mean() == pytest.approx(pibar(DISTS[name], H), abs=0.01)
    # no motion before the key press: chunks without a departure are exactly still
    assert np.all(maxdisp[~contains] < 1e-12)


def test_threshold_crossing_rate_equals_pibar_tau(big_world) -> None:
    name, cfg, eps = big_world
    thr = 0.002
    _, maxdisp, _ = _hold_chunks(eps, H)
    taus = {s: departure_tau(thr, s, cfg.t_acc_s) for s in cfg.speeds_mps}
    assert all(t < H for t in taus.values())
    expected = np.mean([pibar_tau(DISTS[name], H, taus[s]) for s in cfg.speeds_mps])
    assert (maxdisp > thr).mean() == pytest.approx(expected, abs=0.01)


def test_hold_durations_follow_p_d(big_world) -> None:
    name, _, eps = big_world
    holds = holds_table(eps)
    d = holds["D"].to_numpy()
    assert np.allclose(d, holds["key_time"] - holds["start_time"])
    assert np.all(holds["reaction_time"] > 0) and np.allclose(holds["cue_time"] + holds["reaction_time"],
                                                             holds["key_time"])
    dist = DISTS[name]
    assert stats.kstest(d, lambda x: dist.cdf(x)).pvalue > 0.01


def test_departure_tau_matches_profile() -> None:
    for v in (0.01, 0.02, 0.03):
        for thr in (0.0005, 0.002, 0.006):
            tau = departure_tau(thr, v, 0.5)
            x, _ = sweep_profile(np.array([tau - 1e-6, tau + 1e-6]), 0.02, v, 0.5)
            assert x[0] < thr < x[1]


def _frame_dataset(leak: bool, seed: int):
    cfg = HoldWorldConfig(leak=leak, image_shape=(24, 24), fps=10.0, seed=seed,
                          hold_dist=Uniform(2, 10))
    eps = generate(cfg, 50)
    xs, ys, groups = [], [], []
    for ep in eps:
        keys = ep.holds.set_index("hold_id")["key_time"]
        idx = np.flatnonzero(ep.hold_id >= 0)
        xs.append(ep.bmode[idx].reshape(len(idx), -1).astype(np.float64) / 255.0)
        ys.append(keys.loc[ep.hold_id[idx]].to_numpy() <= ep.time[idx] + H)
        groups.append(np.full(len(idx), ep.episode_index))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(groups)


def _cv_auroc(x, y, g) -> float:
    scores = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(x, y, g):
        m = LogisticRegression(C=0.05, max_iter=2000).fit(x[tr], y[tr])
        scores[te] = m.predict_proba(x[te])[:, 1]
    return float(roc_auc_score(y, scores))


def test_leak_off_frames_carry_no_hold_end_information() -> None:
    x, y, g = _frame_dataset(leak=False, seed=3)
    auc = _cv_auroc(x, y, g)
    assert 0.44 < auc < 0.56, auc


def test_leak_on_frames_predict_hold_end() -> None:
    x, y, g = _frame_dataset(leak=True, seed=3)
    assert _cv_auroc(x, y, g) > 0.7


def test_deterministic_given_seed() -> None:
    cfg = HoldWorldConfig(image_shape=(16, 16), seed=5)
    a, b = generate(cfg, 2), generate(cfg, 2)
    c = generate(replace(cfg, seed=6), 2)
    for ea, eb in zip(a, b):
        assert np.array_equal(ea.state, eb.state) and np.array_equal(ea.bmode, eb.bmode)
        assert np.array_equal(ea.wrench, eb.wrench) and ea.holds.equals(eb.holds)
    assert not np.array_equal(a[0].bmode[:50], c[0].bmode[:50])


def test_actions_integrate_back_to_poses() -> None:
    cfg = HoldWorldConfig(render_images=False, tool_yaw_rad=0.7, seed=2)
    ep = generate(cfg, 1)[0]
    pose = integrate_actions(ep.state[0, :6], ep.action)
    assert np.max(np.abs(pose[:, :3] - ep.state[:, :3])) < 1e-9


def test_lift_exits_break_contact_and_darken_image() -> None:
    cfg = HoldWorldConfig(p_lift=1.0, image_shape=(16, 16), seed=4)
    ep = generate(cfg, 1)[0]
    lifted = ep.extras["gt.in_contact"] == 0
    after_approach = ep.time > cfg.approach_s
    assert np.any(lifted & after_approach)
    assert ep.wrench[lifted & after_approach, 2].mean() < 0.5 * cfg.force_n
    assert ep.bmode[lifted & after_approach].mean() < 0.5 * ep.bmode[ep.hold_id >= 0].mean()


def test_lerobot_round_trip(tmp_path) -> None:
    cfg = HoldWorldConfig(image_shape=(16, 16), n_stations=(2, 3), seed=8)
    eps = generate(cfg, 2, sessions=2)
    write_lerobot(eps, tmp_path / "ds")
    back = read_lerobot(tmp_path / "ds")
    assert len(back) == 2
    for a, b in zip(eps, back):
        assert np.array_equal(a.bmode, b.bmode)
        assert np.allclose(a.state, b.state, atol=1e-6) and np.allclose(a.action, b.action, atol=1e-7)
        assert np.array_equal(a.time, b.time)
        assert list(a.mode) == list(b.mode) and list(a.contact_cmd) == list(b.contact_cmd)
        assert np.array_equal(a.hold_id, b.hold_id) and a.session_id == b.session_id
        assert np.allclose(a.extras["gt.adequacy"], b.extras["gt.adequacy"])
        cols = ["hold_id", "start_time", "key_time", "D", "exit_type", "backtrack"]
        assert a.holds[cols].reset_index(drop=True).equals(b.holds[cols].reset_index(drop=True))
