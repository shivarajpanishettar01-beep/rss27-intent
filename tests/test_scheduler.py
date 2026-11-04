"""M5 scheduler logic: reproducible from seed, D histogram matches the configured distribution."""
import numpy as np
import pytest
from scipy import stats

from tests.protocol_fixtures import syn_protocol
from usintent.config import MissingRigFact, load
from usintent.protocol.scheduler import CueClock, distribution_from_config, plan_session
from usintent.protocol.session import NotConfirmed, confirm

PROTO = syn_protocol()   # SYNTHETIC test values only


def test_requires_protocol_values() -> None:
    with pytest.raises(MissingRigFact):
        plan_session(load("protocol"), "s", 1, "uniform", 0, "op", "v", 1)


def test_reproducible_from_seed() -> None:
    a = plan_session(PROTO, "s", 7, "uniform", 0, "op", "syn-vascular", 20)
    b = plan_session(PROTO, "s", 7, "uniform", 0, "op", "syn-vascular", 20)
    c = plan_session(PROTO, "s", 8, "uniform", 0, "op", "syn-vascular", 20)
    da = [h.sampled_d_s for e in a.episodes for h in e.holds]
    assert da == [h.sampled_d_s for e in b.episodes for h in e.holds]
    assert da != [h.sampled_d_s for e in c.episodes for h in e.holds]
    assert all(3 <= len(e.holds) <= 6 for e in a.episodes)


@pytest.mark.parametrize("name", ["uniform", "shifted_exp"])
def test_histogram_matches_configured_distribution(name) -> None:
    plan = plan_session(PROTO, "s", 3, name, 0, "op", "syn-vascular", 600)
    d = np.array([h.sampled_d_s for e in plan.episodes for h in e.holds])
    dist = distribution_from_config(PROTO, name)
    assert len(d) > 2000
    assert stats.kstest(d, lambda x: dist.cdf(x)).pvalue > 0.01
    other = distribution_from_config(PROTO, "shifted_exp" if name == "uniform" else "uniform")
    assert stats.kstest(d, lambda x: other.cdf(x)).pvalue < 1e-6     # the test has power


def test_cue_clock() -> None:
    plan = plan_session(PROTO, "s", 1, "uniform", 0, "op", "syn-vascular", 1).episodes[0]
    clk = CueClock(plan)
    assert clk.tick(100.0) is None                 # no hold started yet
    clk.arrived(10.0, 0)
    d0 = plan.holds[0].sampled_d_s
    assert clk.tick(10.0 + d0 - 1e-3) is None
    cue = clk.tick(10.0 + d0)
    assert cue["stamp"] == pytest.approx(10.0 + d0) and cue["hold_id"] == plan.holds[0].hold_id
    with pytest.raises(ValueError):
        clk.arrived(20.0, 5)


def test_confirmation_required() -> None:
    printed = []
    confirm({"a": 1}, input_fn=lambda _: "yes", print_fn=printed.append)
    assert '"a": 1' in printed[0]
    with pytest.raises(NotConfirmed):
        confirm({"a": 1}, input_fn=lambda _: "y", print_fn=printed.append)


def test_dry_run_script_waits_for_yes(tmp_path, monkeypatch) -> None:
    import subprocess
    import sys

    from usintent.paths import SCRIPTS

    cfg = tmp_path / "configs"
    cmd = [sys.executable, str(SCRIPTS / "run_protocol_dry_run.py"), "--session-id", "x",
           "--seed", "1", "--distribution", "uniform", "--phantom-id", "vascular"]
    r = subprocess.run(cmd, input="no\n", capture_output=True, text=True, check=False)
    # protocol.yaml still has null stations -> refuses before asking (rig/protocol fact missing)
    assert r.returncode != 0 and "MissingRigFact" in r.stderr
    assert not cfg.exists()


def test_randomises_order_backtracking_speed_tilt_lift() -> None:
    plan = plan_session(PROTO, "s", 4, "uniform", 0, "op", "syn-vascular", 300)
    holds = [h for e in plan.episodes for h in e.holds]
    back = np.array([h.backtrack for h in holds])
    assert 0.1 < back.mean() < 0.35                          # back-tracking happens
    for e in plan.episodes:                                  # stations stay within presets
        for a, b in zip(e.holds[:-1], e.holds[1:]):
            assert b.station_index == a.next_station
        assert all(0 <= h.next_station < PROTO["preset_stations"] for h in e.holds)
    assert {h.sweep_speed_mps for h in holds} == set(PROTO["sweep_speeds_mps"])
    tilt = np.array([h.tilt_deg for h in holds])
    twist = np.array([h.twist_deg for h in holds])
    assert np.abs(tilt).max() <= 10.0 and np.abs(twist).max() <= 15.0 and tilt.std() > 3
    lift = np.mean([h.next_mode == "LIFT" for h in holds])
    assert 0.12 < lift < 0.28
    assert plan.info()["phantom_id"] == "syn-vascular"


def test_each_new_protocol_value_is_required() -> None:
    for key in ("preset_stations", "p_backtrack", "p_lift", "sweep_speeds_mps",
                "tilt_twist_range_deg"):
        with pytest.raises(MissingRigFact):
            plan_session(syn_protocol(**{key: None}), "s", 1, "uniform", 0, "op", "v", 1)
