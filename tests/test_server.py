"""M8: policy-server gating, offline harness, rate-vs-e script (SYNTHETIC data)."""
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from usintent.baselines.b1_zero import ZeroMotion
from usintent.baselines.b6_stop_on_hold import StopOnHold
from usintent.data.chunks import ChunkSampler
from usintent.data.passthrough import PassthroughError
from usintent.paths import SCRIPTS
from usintent.policies.evaluate import EvalContext, offline_eval
from usintent.policies.server import MODE_TOPIC, GatingError, PolicyServer, subscriptions
from usintent.sim.holdworld import HoldWorldConfig, departure_tau, generate
from usintent.theory import Uniform, hazard_closed

RIG = {"bmode": "/b", "tcp_pose": "/p", "wrench": "/w", "camera": None}
H = {"state": np.zeros((2, 12)), "time": np.array([0.0, 0.1]), "dt": np.array([0.1])}


class Spy(ZeroMotion):
    """Records every mode a policy receives."""

    def __init__(self, arm_id, n):
        super().__init__(n)
        self.arm_id = arm_id
        self.seen = []

    def _predict(self, history, mode):
        self.seen.append(mode)
        return super()._predict(history, mode)


@pytest.mark.parametrize("arm", ["A1", "A2", "A3", "A6", "A7", "DP-1", "B1", "B2", "B3", "B4", "B5"])
def test_blind_arms_never_receive_mode(arm) -> None:
    assert MODE_TOPIC not in subscriptions(arm, RIG)
    spy = Spy(arm, 5)
    srv = PolicyServer(spy)
    with pytest.raises(GatingError):
        srv.set_scheduler_mode("HOLD")
    srv.step(0.0, H)
    assert spy.seen == [None]


@pytest.mark.parametrize("arm", ["A5", "DP-5", "B6"])
def test_aware_arms_receive_mode(arm, tmp_path) -> None:
    assert MODE_TOPIC in subscriptions(arm, RIG)
    spy = Spy(arm, 5)
    srv = PolicyServer(spy, log_path=tmp_path / "log.jsonl")
    with pytest.raises(RuntimeError):
        srv.step(0.0, H)
    srv.set_scheduler_mode("SWEEP")
    srv.step(0.1, H)
    assert spy.seen == ["SWEEP"]
    rec = json.loads((tmp_path / "log.jsonl").read_text().splitlines()[0])
    assert rec["mode_in"] == "SWEEP" and rec["arm_id"] == arm and len(rec["chunk"]) == 25


def test_b6_inner_policy_stays_blind() -> None:
    inner = Spy("B2", 5)
    b6 = StopOnHold(inner, 5)
    b6.predict(H, "SWEEP")
    assert inner.seen == [None]


def test_harness_reproduces_b4_bound(tmp_path) -> None:
    out = tmp_path / "ev"
    r = subprocess.run([sys.executable, str(SCRIPTS / "run_offline_eval.py"), "--synthetic", "--out",
                        str(out), "--chunk-stride", "9"], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
    tab = pd.read_csv(out / "eval_table.csv").set_index("name")
    assert (tab["passthrough_max_err_m"] < 1e-6).all()
    assert tab.loc["B1", "hold_drift_m"] == 0.0
    test = generate(HoldWorldConfig(render_images=False, seed=302), 120, sessions=3)
    train = generate(HoldWorldConfig(render_images=False, seed=301), 120, sessions=3)
    med = float(np.median(np.concatenate([e.holds["D"].to_numpy() for e in train])))
    d = np.concatenate([e.holds["D"].to_numpy() for e in test])
    assert tab.loc["B4", "timing_error_s"] == pytest.approx(np.mean(np.abs(d - med)), abs=0.05)
    assert json.loads((out / "eval_table.json").read_text())["provenance"]["git_hash"]


def test_policy_result_requires_passthrough() -> None:
    ep = generate(HoldWorldConfig(render_images=False, seed=5, n_stations=(2, 2)), 1)[0]
    ep.action = np.roll(ep.action, 1, axis=0)                      # corrupt the decode path
    s = ChunkSampler([ep], 1.0)
    with pytest.raises(PassthroughError):
        offline_eval("B1", "B1", "-", "-", ZeroMotion(30), EvalContext(s, 0.002), s.index, 0.002)


def test_rate_vs_e_synthetic_follows_theory(tmp_path) -> None:
    tau = round(departure_tau(0.002, 0.02, 0.5), 6)
    r = subprocess.run([sys.executable, str(SCRIPTS / "make_fig_rate_vs_e.py"), "--synthetic",
                        "--out", str(tmp_path), "--tau", str(tau)], capture_output=True, text=True,
                       check=False)
    assert r.returncode == 0, r.stderr
    res = json.loads((tmp_path / "rate_vs_e.json").read_text())["curves"]["uniform"]
    rows = [x for x in res["oracle-synthetic"] if x["n"] > 2000 and x["e_lo"] < 8]
    for x in rows:
        mid = (x["e_lo"] + x["e_hi"]) / 2
        assert x["rate"] == pytest.approx(hazard_closed(Uniform(2, 10), mid, 1.0 - tau), abs=0.05)
