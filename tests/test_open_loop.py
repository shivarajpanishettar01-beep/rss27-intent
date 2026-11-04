"""Open-loop metrics with known answers (SPEC section 5)."""
import numpy as np
import pytest

from usintent.data.chunks import ChunkSampler, policy_target
from usintent.metrics import open_loop as ol
from usintent.metrics.closed_loop import timing_error
from usintent.sim.holdworld import HoldWorldConfig, departure_tau, generate
from usintent.theory import Uniform, pibar_tau


def _const(m, n, xy=(0.0, 0.0), rot=(0.0, 0.0, 0.0)):
    c = np.zeros((m, n, 5))
    c[..., :2] = xy
    c[..., 2:] = rot
    return c


def test_chunk_errors_known() -> None:
    gt = _const(4, 10)
    pred = _const(4, 10, xy=(0.003, 0.004), rot=(0.0, 0.0, 0.2))
    assert np.allclose(ol.chunk_error_trans(pred, gt), 0.005)
    assert np.allclose(ol.chunk_error_rot(pred, gt), 0.2)
    with pytest.raises(ValueError):
        ol.chunk_error_trans(pred[:, :, :4], gt[:, :, :4])


def test_ratio_to_cv_known() -> None:
    phase = np.array(["hold", "hold", "sweep", "sweep"])
    out = ol.ratio_to_cv(np.array([1e-6, 3e-6, 0.004, 0.002]), np.array([1e-6, 1e-6, 0.002, 0.002]),
                         phase).set_index("phase")
    assert out.loc["sweep", "ratio"] == pytest.approx(1.5)
    assert out.loc["hold", "ratio"] == pytest.approx(2.0) and out.loc["hold", "unstable"]
    assert not out.loc["sweep", "unstable"]


def test_hold_drift_and_rate_known() -> None:
    pred = np.concatenate([_const(3, 5, xy=(0.001, 0.0)), _const(1, 5, xy=(0.01, 0.0))])
    in_hold = np.array([True, True, False, True])
    dep = np.array([False, False, False, True])
    assert ol.hold_drift(pred, in_hold, dep) == pytest.approx(0.001)
    r = ol.departure_like_rate(pred, in_hold, 0.005, np.array(["SWEEP", "LIFT", "SWEEP", "SWEEP"]))
    assert r["all"] == pytest.approx(1 / 3) and r["LIFT"] == 0.0 and r["SWEEP"] == 0.5


def test_direction_known() -> None:
    pred = np.concatenate([_const(1, 3, xy=(0.01, 0)), _const(1, 3, xy=(0, 0.01)), _const(1, 3)])
    cos = ol.departure_direction(pred, np.array([[1.0, 0], [1.0, 0], [1.0, 0]]))
    assert cos[0] == pytest.approx(1.0) and cos[1] == pytest.approx(0.0) and np.isnan(cos[2])


def test_rate_vs_e_known() -> None:
    e = np.array([0.5, 1.5, 1.6, 2.5])
    flags = np.array([False, True, False, True])
    out = ol.rate_vs_e(flags, e, np.array([0.0, 1.0, 2.0, 3.0]))
    assert list(out["rate"]) == [0.0, 0.5, 1.0] and list(out["n"]) == [1, 2, 1]


def test_timing_error_known() -> None:
    assert np.allclose(timing_error(np.array([5.0, 2.0]), np.array([3.0, 4.0])), [2.0, 2.0])
    assert ol.first_departure_time(np.array([0.0, 0.1, 0.2]), np.array([False, True, True])) == 0.1


def test_ground_truth_chunks_reproduce_pibar_tau() -> None:
    """Scoring the recorded chunks as if they were predictions gives pi-bar_tau (SYNTHETIC)."""
    cfg = HoldWorldConfig(render_images=False, speeds_mps=(0.02,), seed=9)
    s = ChunkSampler(generate(cfg, 500), 1.0)
    rows = s.index[s.index["phase"] == "hold"]
    gt = policy_target(s.chunks6(rows))
    thr = 0.002
    rate = ol.departure_like_rate(gt, np.ones(len(rows), bool), thr)["all"]
    assert rate == pytest.approx(pibar_tau(Uniform(2, 10), 1.0, departure_tau(thr, 0.02, 0.5)), abs=0.01)
    assert ol.hold_drift(gt, np.ones(len(rows), bool), rows["contains_departure"].to_numpy()) == 0.0
