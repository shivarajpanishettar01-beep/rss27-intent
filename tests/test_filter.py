"""M9: property tests - the filter never outputs a command exceeding limits."""
import numpy as np
from scipy.spatial.transform import Rotation

from tests.control_fixtures import TEST_SAFETY
from usintent.control.filter import LimitFilter
from usintent.control.safety import load_limits

LIM = load_limits(TEST_SAFETY)
DT = 0.01


def test_random_property() -> None:
    rng = np.random.default_rng(0)
    f = LimitFilter(LIM)
    box = np.asarray(LIM.workspace_box_base_m).reshape(3, 2)
    n_override = 0
    for i in range(20000):
        pos = rng.uniform(box[:, 0], box[:, 1])                       # state inside the safe set
        z = Rotation.random(random_state=int(rng.integers(1 << 31))).apply([0, 0, 1.0])
        disp = rng.uniform(-0.01, LIM.max_normal_displacement_m)
        force = rng.uniform(0, 2 * LIM.max_contact_force_n)
        v_prev = rng.normal(0, 0.02, 3)
        v_prev *= min(1.0, LIM.max_tcp_speed_mps / np.linalg.norm(v_prev))
        raw = rng.normal(0, rng.choice([0.01, 0.1, 10.0]), 3)
        n0 = len(f.events)
        v = f.apply(float(i), raw, v_prev, pos, z, disp, force, DT)
        zu = z / np.linalg.norm(z)
        assert np.linalg.norm(v) <= LIM.max_tcp_speed_mps * (1 + 1e-9)
        nxt = pos + v * DT
        assert np.all(nxt >= box[:, 0] - 1e-9) and np.all(nxt <= box[:, 1] + 1e-9)
        assert disp + (v @ zu) * DT <= LIM.max_normal_displacement_m + 1e-9
        if force >= LIM.max_contact_force_n:
            assert v @ zu <= 1e-9
        new = [e.constraint for e in f.events[n0:]]
        if np.linalg.norm(v - v_prev) > LIM.max_tcp_accel_mps2 * DT * (1 + 1e-6):
            assert "accel_override" in new or "fallback_zero" in new
            n_override += 1
        if not np.allclose(v, raw):
            assert new, "every intervention is logged"
            assert all(len(e.raw_command) == 3 and len(e.filtered_command) == 3 for e in f.events[n0:])
    assert n_override < 20000


def test_passes_through_admissible_command() -> None:
    f = LimitFilter(LIM)
    v = f.apply(0.0, np.array([0.001, 0, 0]), np.array([0.001, 0, 0]), np.zeros(3) + [0, 0, 0.1],
                np.array([0, 0, -1.0]), 0.0, 1.0, DT)
    assert np.allclose(v, [0.001, 0, 0]) and not f.events


def test_non_finite_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        LimitFilter(LIM).apply(0.0, np.array([np.nan, 0, 0]), np.zeros(3), np.zeros(3) + 0.1,
                               np.array([0, 0, 1.0]), 0.0, 0.0, DT)
