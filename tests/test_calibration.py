"""Threshold and tau measurement on SYNTHETIC cued data with known jitter and ramps."""
import numpy as np
import pytest

from usintent.data.chunks import ChunkSampler
from usintent.metrics.calibration import measure_tau, tau_summary, zero_motion_threshold
from usintent.sim.holdworld import HoldWorldConfig, departure_tau, generate


def test_threshold_from_jitter_and_tau_per_speed() -> None:
    cfg = HoldWorldConfig(render_images=False, pose_jitter_m=0.0002, seed=5)
    eps = generate(cfg, 80)
    thr = zero_motion_threshold(ChunkSampler(eps, 1.0), quantile=0.999, margin=1.5)
    # jitter: |difference of two N(0, s^2 I2) positions| ~ Rayleigh(s*sqrt(2)); max over 30 steps
    assert 0.0004 < thr["jitter_median_m"] < 0.0012
    assert thr["threshold_m"] > thr["jitter_max_m"] * 0.9
    taus = measure_tau(eps, 0.003)
    summ = tau_summary(taus)
    for v in cfg.speeds_mps:
        expect = departure_tau(0.003, v, cfg.t_acc_s)
        got = summ["per_speed"][str(v)]["median"]
        assert got == pytest.approx(expect, abs=2.5 / cfg.fps)
    assert summ["n_never_crossed"] == 0
    assert summ["tau_s"] == pytest.approx(np.mean([summ["per_speed"][str(v)]["mean"]
                                                   for v in cfg.speeds_mps]))


def test_script_synthetic() -> None:
    import subprocess
    import sys

    from usintent.paths import SCRIPTS

    r = subprocess.run([sys.executable, str(SCRIPTS / "run_measure_threshold_tau.py"), "--synthetic"],
                       capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
    assert "departure_threshold_m" in r.stdout and '"data_source": "synthetic"' in r.stdout
