"""Teleop gating against a mock robot: motion only after the operator key press, and only to
the planned station (SYNTHETIC protocol values)."""
import pytest

from tests.protocol_fixtures import syn_protocol
from usintent.protocol.scheduler import plan_session
from usintent.protocol.teleop import KEY_MODES, GateError, TeleopGate

KEY_FOR = {v: k for k, v in KEY_MODES.items()}


def _gate(seed=3):
    proto = syn_protocol(stations_per_episode={"min": 5, "max": 5})
    plan = plan_session(proto, "s", seed, "uniform", 0, "op", "syn-vascular", 1).episodes[0]
    return TeleopGate(plan=plan, n_stations=proto["preset_stations"]), plan


class MockRobot:
    """Moves only when the gate releases a command; records every executed motion."""

    def __init__(self, gate):
        self.gate, self.moves = gate, []

    def try_move(self, t, target=None):
        try:
            cmd = self.gate.motion_command(t, target)
        except GateError:
            return False
        self.moves.append(cmd)
        return True


def test_motion_gated_on_key_and_follows_plan() -> None:
    gate, plan = _gate()
    robot = MockRobot(gate)
    gate.arrived_initial(0.0)
    t = 1.0
    for i, h in enumerate(plan.holds):
        assert not robot.try_move(t)                                # no key yet
        wrong = "l" if h.next_mode == "SWEEP" else "s"
        assert gate.key(t, wrong) is None                           # wrong key ignored
        ev = gate.key(t + 0.5, KEY_FOR[h.next_mode])
        assert ev["mode"] == h.next_mode and ev["hold_id"] == h.hold_id
        bad = (h.next_station + 1) % 6
        assert not robot.try_move(t + 0.5, bad)                     # not the planned station
        assert robot.try_move(t + 0.5)
        assert not robot.try_move(t + 0.6)                          # one command per key
        cmd = robot.moves[-1]
        assert cmd["target_station"] == h.next_station and cmd["speed_mps"] == h.sweep_speed_mps
        if i + 1 < len(plan.holds):
            assert cmd["tilt_deg"] == plan.holds[i + 1].tilt_deg
        with pytest.raises(GateError):
            gate.arrived(t + 1.0, (h.next_station + 2) % 6)
        gate.arrived(t + 1.0, h.next_station)
        t += 5.0
    assert gate.state == "DONE"
    assert sum(e["source"] == "wrong_key" for e in gate.events) == len(plan.holds)


def test_bad_inputs() -> None:
    gate, _ = _gate()
    assert gate.key(0.0, "x") is None
    with pytest.raises(GateError):
        gate.arrived(0.0, 1)
