"""M10 acceptance: scripted policies on a mock robot reproduce Fig. 1c (SYNTHETIC)."""
import json
import subprocess
import sys

import numpy as np
import pytest

from usintent.closedloop.trials import ScriptedLeaver, TrialConfig, run_paired, run_trial
from usintent.config import load
from usintent.metrics.closed_loop import force_in_band, fraction_true, trial_success
from usintent.paths import SCRIPTS
from usintent.protocol.scheduler import plan_session
from usintent.theory import Uniform


def test_fig1c_reproduced_within_mc_error(tmp_path) -> None:
    r = subprocess.run([sys.executable, str(SCRIPTS / "make_table_closed_loop.py"), "--synthetic",
                        "--out", str(tmp_path), "--n-trials", "600"], capture_output=True, text=True,
                       check=False)
    assert r.returncode == 0, r.stderr
    rows = json.loads((tmp_path / "closed_loop.json").read_text())["rows"]
    assert len(rows) == 6
    for row in rows:
        half = (row["ci_hi"] - row["ci_lo"]) / 2
        # within Monte-Carlo error (95% cluster-bootstrap CI widened by the dt discretisation)
        assert abs(row["timing_error_s"] - row["predicted_s"]) <= half + 0.02, row
        assert row["never_left"] == 0 and row["n_holds"] == 2400


def test_pairing_same_holds_for_every_arm() -> None:
    proto = {**load("protocol"), "stations_per_episode": {"min": 3, "max": 3}}
    plans = plan_session(proto, "p", 1, "uniform", 0, "op", 5).episodes
    pols = {r: ScriptedLeaver(r, Uniform(2, 10), 5, 0.001, 0) for r in ("median", "hedger")}
    df = run_paired(pols, plans, TrialConfig(0.05, 0.002))
    a, b = df[df.arm == "median"], df[df.arm == "hedger"]
    assert np.array_equal(a[["trial_id", "hold_id", "D"]].to_numpy(), b[["trial_id", "hold_id", "D"]].to_numpy())
    assert np.allclose(a["T"], Uniform(2, 10).median(), atol=0.05 + 1e-9)


def test_robot_does_not_move_before_commit() -> None:
    proto = {**load("protocol"), "stations_per_episode": {"min": 2, "max": 2}}
    plan = plan_session(proto, "p", 2, "uniform", 0, "op", 1).episodes[0]
    df = run_trial(ScriptedLeaver("median", Uniform(2, 10), 5, 0.001, 0), plan, TrialConfig(0.05, 0.002), "t")
    assert (df["drift_m"] == 0).all()


def test_closed_loop_metric_helpers() -> None:
    ok = force_in_band(np.array([1.0, 3.0, 5.0]), (2.0, 4.0))
    assert list(ok) == [False, True, False] and fraction_true(ok) == pytest.approx(1 / 3)
    assert trial_success(np.array([0.9, 0.85]), np.array([True, True]), 0.8)
    assert not trial_success(np.array([0.9, 0.7]), np.array([True, True]), 0.8)
    assert not trial_success(np.array([0.9]), np.array([True, False]), 0.8)
    with pytest.raises(ValueError):
        force_in_band(np.zeros(2), (3.0, 1.0))
