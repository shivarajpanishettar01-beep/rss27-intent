"""M9: hybrid controller on a SYNTHETIC mock plant: force step response, m_hat switching,
HOLD forces zero tangential motion. Step responses are logged (synthetic; URSim is a lab step)."""
import json

import numpy as np
import pytest

from tests.control_fixtures import TEST_CONTROL, TEST_SAFETY
from usintent.control.controller import ControlLoop, ControlParams, MockPlant, hybrid_command
from usintent.control.fault import STREAMS
from usintent.control.safety import load_limits

LIM = load_limits(TEST_SAFETY)
P = ControlParams(**TEST_CONTROL)
DT = 0.01


def _run(m_hat_fn, n, plant, chunk):
    loop = ControlLoop(P, LIM)
    log = []
    for k in range(n):
        t = k * DT
        for s in STREAMS:
            loop.latch.stamp(s, t, [0.0])
        disp = plant.penetration()
        cmd = loop.step(t, plant.pose, plant.force(), disp, m_hat_fn(t), chunk, DT)
        plant.step(cmd, DT)
        log.append({"t": t, "force": plant.force(), "z": float(plant.pose[2]),
                    "x": float(plant.pose[0]), "cmd": cmd.tolist()})
    return loop, log


def test_force_step_response(tmp_path) -> None:
    plant = MockPlant(np.array([0.0, 0.0, 0.1, np.pi, 0.0, 0.0]), surface_z_base=0.1,
                      stiffness_n_per_m=1000.0)
    loop, log = _run(lambda t: "HOLD", 600, plant, np.zeros((10, 5)))
    f = np.array([r["force"] for r in log])
    assert f[-1] == pytest.approx(P.force_setpoint_n, abs=0.05)
    assert f.max() <= LIM.max_contact_force_n
    assert not loop.latch.faulted
    assert np.allclose([r["x"] for r in log], 0.0)              # HOLD: no tangential motion
    (tmp_path / "step_response_synthetic.json").write_text(json.dumps(log[::10]))


def test_lift_retracts_and_sweep_tracks() -> None:
    plant = MockPlant(np.array([0.0, 0.0, 0.095, np.pi, 0.0, 0.0]), 0.1, 1000.0)
    chunk = np.zeros((10, 5))
    chunk[0, 0] = 0.0002                                      # 2 cm/s along tool x
    _, log = _run(lambda t: "SWEEP" if t < 1.0 else "LIFT", 200, plant, chunk)
    assert log[99]["x"] > 0.015                                # moved along x while sweeping
    assert log[-1]["z"] > log[99]["z"] and log[-1]["force"] == 0.0


def test_policy_never_commands_force() -> None:
    chunk = np.full((10, 5), 1.0)                              # absurd policy output
    tw = hybrid_command(P, "SWEEP", chunk, force_n=4.0, dt=DT)
    assert tw[2] == pytest.approx(0.0)                         # normal axis owned by force law


def test_control_period_from_config() -> None:
    from usintent.config import MissingRigFact
    from usintent.control.controller import control_period_s

    with pytest.raises(MissingRigFact):
        control_period_s({"rate_hz": None}, LIM)
    assert control_period_s({"rate_hz": 100.0}, LIM) == pytest.approx(0.01)
    with pytest.raises(ValueError):                        # 20 ms period >= 20 ms watchdog
        control_period_s({"rate_hz": 50.0}, LIM)
