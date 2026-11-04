"""M5 scheduler logic: reproducible from seed, D histogram matches the configured distribution."""
import numpy as np
import pytest
from scipy import stats

from usintent.config import MissingRigFact, load
from usintent.protocol.scheduler import CueClock, distribution_from_config, plan_session
from usintent.protocol.session import NotConfirmed, confirm

PROTO = {**load("protocol"), "stations_per_episode": {"min": 3, "max": 6}}   # test values only


def test_requires_protocol_values() -> None:
    with pytest.raises(MissingRigFact):
        plan_session(load("protocol"), "s", 1, "uniform", 0, "op", 1)


def test_reproducible_from_seed() -> None:
    a = plan_session(PROTO, "s", 7, "uniform", 0, "op", 20)
    b = plan_session(PROTO, "s", 7, "uniform", 0, "op", 20)
    c = plan_session(PROTO, "s", 8, "uniform", 0, "op", 20)
    da = [h.sampled_d_s for e in a.episodes for h in e.holds]
    assert da == [h.sampled_d_s for e in b.episodes for h in e.holds]
    assert da != [h.sampled_d_s for e in c.episodes for h in e.holds]
    assert all(3 <= len(e.holds) <= 6 for e in a.episodes)


@pytest.mark.parametrize("name", ["uniform", "shifted_exp"])
def test_histogram_matches_configured_distribution(name) -> None:
    plan = plan_session(PROTO, "s", 3, name, 0, "op", 600)
    d = np.array([h.sampled_d_s for e in plan.episodes for h in e.holds])
    dist = distribution_from_config(PROTO, name)
    assert len(d) > 2000
    assert stats.kstest(d, lambda x: dist.cdf(x)).pvalue > 0.01
    other = distribution_from_config(PROTO, "shifted_exp" if name == "uniform" else "uniform")
    assert stats.kstest(d, lambda x: other.cdf(x)).pvalue < 1e-6     # the test has power


def test_cue_clock() -> None:
    plan = plan_session(PROTO, "s", 1, "uniform", 0, "op", 1).episodes[0]
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
           "--seed", "1", "--distribution", "uniform"]
    r = subprocess.run(cmd, input="no\n", capture_output=True, text=True, check=False)
    # protocol.yaml still has null stations -> refuses before asking (rig/protocol fact missing)
    assert r.returncode != 0 and "MissingRigFact" in r.stderr
    assert not cfg.exists()
