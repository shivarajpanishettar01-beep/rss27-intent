"""Phase segmentation (SPEC section 4). SYNTHETIC data with known holds."""
import numpy as np
import pytest

from usintent.config import MissingRigFact, load
from usintent.data.segmentation import (
    SegParams,
    hold_f1,
    params_from_config,
    segment_cued,
    segment_threshold,
)
from usintent.sim.holdworld import HoldWorldConfig, generate

P = SegParams(v_hold_mps=0.002, t_hold_min_s=1.0, t_onset_s=0.5, transition_window_s=0.5)


@pytest.fixture(scope="module")
def eps():
    return generate(HoldWorldConfig(render_images=False, seed=21), 60)


def test_threshold_segmentation_recovers_holds(eps) -> None:
    pred = np.concatenate([segment_threshold(e, P).hold_mask for e in eps])
    truth = np.concatenate([e.hold_id >= 0 for e in eps])
    assert hold_f1(pred, truth) > 0.95


def test_force_contact_mode_also_recovers_holds(eps) -> None:
    p = SegParams(0.002, 1.0, 0.5, 0.5, contact_force_n=2.5)
    pred = np.concatenate([segment_threshold(e, p).hold_mask for e in eps])
    truth = np.concatenate([e.hold_id >= 0 for e in eps])
    assert hold_f1(pred, truth) > 0.95
    seg = segment_threshold(eps[0], p)
    assert np.all(seg.phase[eps[0].time < 1.0] == "approach")


def test_cued_segmentation_matches_logs(eps) -> None:
    e = eps[0]
    seg = segment_cued(e, 0.5)
    assert np.array_equal(seg.hold_mask, e.hold_id >= 0)
    assert len(seg.holds) == len(e.holds)
    a, b = seg.holds[0]
    assert np.allclose(seg.elapsed[a:b], e.time[a:b] - e.time[a])
    assert seg.in_transition[a] and not seg.in_transition[(a + b) // 2]


def test_f1_known_values() -> None:
    t = np.array([1, 1, 0, 0], bool)
    assert hold_f1(t, t) == 1.0
    assert hold_f1(np.array([1, 0, 1, 0], bool), t) == pytest.approx(0.5)


def test_config_nulls_refuse() -> None:
    with pytest.raises(MissingRigFact):
        params_from_config(load("data"))


def test_sensitivity_table(eps) -> None:
    from usintent.data.segmentation import sensitivity

    rows = sensitivity(eps[:20], P)
    assert [r["factor"] for r in rows] == [0.5, 0.75, 1.0, 1.25, 1.5]
    base = rows[2]
    assert base["f1_vs_base"] == 1.0
    assert base["departure_fraction"] == pytest.approx(1 / 6, abs=0.03)
    assert all(r["f1_vs_base"] > 0.9 for r in rows)
    # raising v_hold alone can only add hold time (t_hold_min also scales in `rows`)
    from dataclasses import replace

    lo = sensitivity(eps[:20], replace(P, v_hold_mps=0.001), factors=(1.0,))[0]
    hi = sensitivity(eps[:20], replace(P, v_hold_mps=0.004), factors=(1.0,))[0]
    assert lo["hold_time_fraction"] < base["hold_time_fraction"] < hi["hold_time_fraction"]


def test_sensitivity_script_runs_on_synthetic() -> None:
    import subprocess
    import sys

    from usintent.paths import SCRIPTS

    out = subprocess.check_output([sys.executable, str(SCRIPTS / "run_segmentation_sensitivity.py"),
                                   "--synthetic", "5", "--v-hold", "0.002", "--t-hold-min", "1.0",
                                   "--t-onset", "0.5"], text=True)
    assert '"data_source": "synthetic"' in out and "SYNTHETIC" in out


def test_contact_source_must_be_explicit() -> None:
    cfg = load("data")
    seg = {**cfg["segmentation"], "v_hold_mps": 0.002, "t_hold_min_s": 1.0, "t_onset_s": 0.5}
    with pytest.raises(MissingRigFact):                    # contact_source null
        params_from_config({**cfg, "segmentation": seg})
    p = params_from_config({**cfg, "segmentation": {**seg, "contact_source": "motion"}})
    assert p.contact_force_n is None
    with pytest.raises(MissingRigFact):                    # force needs its threshold
        params_from_config({**cfg, "segmentation": {**seg, "contact_source": "force"}})
