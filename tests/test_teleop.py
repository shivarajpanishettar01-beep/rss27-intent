"""M5 teleop gating against a mock robot: motion only after the operator key press."""
import pytest

from usintent.protocol.teleop import GateError, TeleopGate


class MockRobot:
    """Moves only when given a target by the gate; records every executed motion."""

    def __init__(self, gate: TeleopGate):
        self.gate = gate
        self.moves = []

    def try_move(self, t: float, target: int) -> bool:
        try:
            cmd = self.gate.motion_command(t, target)
        except GateError:
            return False
        self.moves.append(cmd)
        return True


def test_motion_gated_on_key() -> None:
    gate = TeleopGate(n_stations=4)
    robot = MockRobot(gate)
    gate.arrived_initial(0.0)
    assert not robot.try_move(1.0, 1)                     # no key yet
    ev = gate.key(2.5, "s")
    assert ev["mode"] == "SWEEP" and ev["stamp"] == 2.5 and ev["source"] == "operator_key"
    assert robot.try_move(2.5, 1)
    assert not robot.try_move(2.6, 2)                     # one command per key
    assert gate.key(3.0, "s") is None                     # keys ignored while moving
    gate.arrived(4.0, 1)
    assert not robot.try_move(4.1, 2)
    gate.key(6.0, "l")
    assert robot.try_move(6.0, 2)
    assert [m["stamp"] for m in robot.moves] == [2.5, 6.0]
    assert all(m["stamp"] >= e["stamp"] for m, e in zip(
        robot.moves, [x for x in gate.events if x["source"] == "operator_key"]))
    assert [e["source"] for e in gate.events] == ["arrival", "operator_key", "arrival", "operator_key"]


def test_bad_inputs() -> None:
    gate = TeleopGate(n_stations=2)
    assert gate.key(0.0, "x") is None
    gate.key(0.1, "s")
    with pytest.raises(GateError):
        gate.motion_command(0.2, 5)
    with pytest.raises(GateError):
        TeleopGate(n_stations=2).arrived(0.0, 1)
