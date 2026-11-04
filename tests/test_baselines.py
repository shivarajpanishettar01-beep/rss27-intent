"""M3 acceptance for baselines B1-B6 on SYNTHETIC cued data (SPEC section 6)."""
from dataclasses import replace

import numpy as np
import pytest

from usintent.baselines.b1_zero import ZeroMotion
from usintent.baselines.b2_const_vel import ConstantVelocity
from usintent.baselines.b3_imm import (
    IMMBaseline,
    IMMParams,
    detection_lags,
    estimate_transition_matrix,
)
from usintent.baselines.b4_hazard_optimal import HazardOptimalBlind
from usintent.baselines.b5_speckle import SpeckleDecorrelation, patch_ncc
from usintent.baselines.b6_stop_on_hold import StopOnHold
from usintent.data.chunks import ChunkSampler, policy_target
from usintent.metrics import open_loop as ol
from usintent.policies.evaluate import EvalContext, evaluate_open_loop, hold_timing, predict_chunks
from usintent.sim.holdworld import HoldWorldConfig, generate
from usintent.theory import Uniform, bound_median

THR = 0.002          # synthetic departure threshold (m)
V_HOLD = 0.002       # synthetic hold speed threshold (m/s)
H = 1.0
CFG = HoldWorldConfig(render_images=False, seed=31)
N = 30               # steps per chunk at 30 fps


@pytest.fixture(scope="module")
def data():
    train = generate(replace(CFG, seed=31), 150, sessions=3)
    test = generate(replace(CFG, seed=32), 150, sessions=3)
    s_tr = ChunkSampler(train, H, history_k=2)
    s_te = ChunkSampler(test, H, history_k=2)
    return train, test, s_tr, s_te


def _b4(s_tr):
    idx = s_tr.index
    rows = []
    for ep in s_tr.episodes:
        for _, h in ep.holds.iterrows():
            k = int(np.searchsorted(ep.time, h["key_time"]))
            rows.append((ep.episode_index, k, h["exit_type"]))
    sel = idx.set_index(["episode_index", "start"]).index
    keep = [(e, k) in sel for e, k, _ in rows]
    rows = [r for r, ok in zip(rows, keep) if ok]
    import pandas as pd

    df = pd.DataFrame({"episode_index": [r[0] for r in rows], "start": [r[1] for r in rows]})
    chunks = policy_target(s_tr.chunks6(df))
    return HazardOptimalBlind.fit(Uniform(2, 10).median(), chunks, [r[2] for r in rows])


def test_b4_timing_error_equals_bound(data) -> None:
    _, _, s_tr, s_te = data
    b4 = _b4(s_tr)
    timing = hold_timing(b4, EvalContext(s_te, V_HOLD), THR)
    assert timing["T"].notna().all()
    assert len(timing) > 500
    d = timing["D"].to_numpy()
    sample_bound = float(np.mean(np.abs(d - Uniform(2, 10).median())))
    assert timing["err"].mean() == pytest.approx(sample_bound, abs=0.05)
    assert timing["err"].mean() == pytest.approx(bound_median(Uniform(2, 10)), abs=0.1)


def test_b4_timing_error_equals_bound_large() -> None:
    """Same check against the population bound with enough holds for +-0.05 s."""
    eps = generate(replace(CFG, seed=41), 700)
    s = ChunkSampler(eps, H, history_k=1)
    b4 = HazardOptimalBlind(Uniform(2, 10).median(), np.full((N, 5), 0.01), "SWEEP")
    timing = hold_timing(b4, EvalContext(s, V_HOLD), THR)
    assert len(timing) > 2900
    assert timing["err"].mean() == pytest.approx(bound_median(Uniform(2, 10)), abs=0.05)


def test_b1_zero_drift_and_b2_beats_b1_in_sweeps(data) -> None:
    _, _, _, s_te = data
    ctx = EvalContext(s_te, V_HOLD)
    rows = s_te.index.iloc[::7]
    b1 = evaluate_open_loop(ZeroMotion(N), ctx, rows, THR)
    assert b1["hold_drift_m"] == 0.0
    assert b1["passthrough"]["max_trans_err_m"] < 1e-6
    pred1, _ = predict_chunks(ZeroMotion(N), ctx, rows)
    pred2, _ = predict_chunks(ConstantVelocity(N), ctx, rows)
    gt = policy_target(s_te.chunks6(rows))
    sweep = (rows["phase"] == "sweep").to_numpy()
    e1 = ol.chunk_error_trans(pred1, gt)[sweep].mean()
    e2 = ol.chunk_error_trans(pred2, gt)[sweep].mean()
    assert e2 < e1
    out = evaluate_open_loop(ConstantVelocity(N), ctx, rows, THR, cv_chunks=pred2)
    assert np.allclose(out["ratio_to_cv"]["ratio"].dropna(), 1.0)


def test_b3_detection_lag_reported(data) -> None:
    train, test, _, _ = data
    p = IMMParams(trans=estimate_transition_matrix(train), dt=1 / CFG.fps)
    assert np.allclose(p.trans.sum(axis=1), 1.0) and np.all(np.diag(p.trans) > 0.9)
    lags = detection_lags(test[:20], p)
    assert len(lags) > 50
    assert lags["lag_any_departure_s"].notna().mean() > 0.95
    assert (lags["lag_any_departure_s"].dropna() > 0).all()
    assert lags["lag_any_departure_s"].median() < 1.0
    # as a policy: holds -> zero output, cruising sweep -> CV chunk
    b3 = IMMBaseline(p, N)
    ctx = EvalContext(ChunkSampler(test[:1], H, history_k=1), None)
    rows = ctx.sampler.index
    _, m_hat = predict_chunks(b3, ctx, rows)
    hold_mid = (rows["phase"] == "hold") & (rows["e"] > 1.0) & ~rows["contains_departure"]
    assert np.all(np.array(m_hat)[hold_mid.to_numpy()] == "HOLD")


def test_b5_recovers_out_of_plane_motion() -> None:
    cfg = HoldWorldConfig(seed=3, n_stations=(3, 3), p_lift=0.0)
    calib, ep = generate(cfg, 2)
    hold = np.flatnonzero(calib.hold_id >= 0)
    rho0 = float(np.mean([np.median(patch_ncc(calib.bmode[k - 1].astype(float),
                                              calib.bmode[k].astype(float), 16, 8.0))
                          for k in hold[5:60]]))
    b5 = SpeckleDecorrelation(N, sigma_m=cfg.speckle_sigma_m, rho0=rho0)
    step = np.abs(np.diff(ep.state[:, 0])) * cfg.fps        # true elevational speed per frame pair
    moving = np.flatnonzero(step > 0.005) + 1
    est = np.array([b5.velocity({"bmode": ep.bmode[k - 1:k + 1], "time": ep.time[k - 1:k + 1]})[0]
                    for k in moving])
    rel = np.abs(est - step[moving - 1]) / step[moving - 1]
    assert np.median(rel) < 0.10
    still = hold[5:60]
    est0 = np.array([b5.velocity({"bmode": ep.bmode[k - 1:k + 1], "time": ep.time[k - 1:k + 1]})[0]
                     for k in np.flatnonzero(ep.hold_id >= 0)[5:60]])
    assert np.median(est0) < 0.004 and len(still)


def test_b6_forces_zero_on_hold(data) -> None:
    _, _, _, s_te = data
    ctx = EvalContext(s_te, V_HOLD)
    b6 = StopOnHold(ConstantVelocity(N), N)
    rows = s_te.index.iloc[::11]
    pred6, m6 = predict_chunks(b6, ctx, rows)
    pred2, _ = predict_chunks(ConstantVelocity(N), ctx, rows)
    sched_hold = np.array([ctx.scheduler_mode(int(e), int(k)) == "HOLD"
                           for e, k in zip(rows["episode_index"], rows["start"])])
    assert sched_hold.any() and (~sched_hold).any()
    assert np.all(pred6[sched_hold] == 0) and all(m == "HOLD" for m, h in zip(m6, sched_hold) if h)
    assert np.allclose(pred6[~sched_hold], pred2[~sched_hold])


def test_intent_gating_for_baselines() -> None:
    h = {"state": np.zeros((2, 12)), "time": np.array([0.0, 0.1]), "dt": np.array([0.1])}
    with pytest.raises(PermissionError):
        ZeroMotion(N).predict(h, "HOLD")
    with pytest.raises(ValueError):
        StopOnHold(ZeroMotion(N), N).predict(h, None)
    assert StopOnHold(ZeroMotion(N), N).intent_aware and not ConstantVelocity(N).intent_aware
