"""M9: fault latch triggers on stale inputs, NaN, lost F/T, watchdog miss, exceptions."""
import numpy as np
import pytest

from tests.control_fixtures import TEST_CONTROL, TEST_SAFETY
from usintent.control.controller import ControlLoop, ControlParams
from usintent.control.fault import STREAMS, FaultLatch
from usintent.control.safety import load_limits

LIM = load_limits(TEST_SAFETY)
POSE = np.array([0.0, 0.0, 0.1, np.pi, 0.0, 0.0])
CHUNK = np.full((10, 5), 0.0005)


def _feed(latch, t, skip=()):
    for s in STREAMS:
        if s not in skip:
            latch.stamp(s, t, [0.0])


def _loop():
    return ControlLoop(ControlParams(**TEST_CONTROL), LIM)


def test_never_received_is_a_fault() -> None:
    latch = FaultLatch(0.1, 0.02)
    assert not latch.tick(0.0) and "never received" in latch.reason


@pytest.mark.parametrize("lost", ["ft", "bmode"])
def test_lost_stream_latches(lost) -> None:
    loop = _loop()
    t = 0.0
    for _ in range(5):
        _feed(loop.latch, t)
        assert np.any(loop.step(t, POSE, 4.0, 0.0, "SWEEP", CHUNK, 0.01) != 0)
        t += 0.01
    for _ in range(15):                          # stream stops; others continue
        _feed(loop.latch, t, skip=(lost,))
        cmd = loop.step(t, POSE, 4.0, 0.0, "SWEEP", CHUNK, 0.01)
        t += 0.01
    assert loop.latch.faulted and lost in loop.latch.reason and np.all(cmd == 0)


def test_nan_latches_and_stays_latched() -> None:
    loop = _loop()
    _feed(loop.latch, 0.0)
    loop.latch.stamp("ft", 0.005, [np.nan])
    assert np.all(loop.step(0.01, POSE, 4.0, 0.0, "SWEEP", CHUNK, 0.01) == 0)
    for k in range(2, 10):                       # inputs healthy again: still latched
        _feed(loop.latch, k * 0.01)
        assert np.all(loop.step(k * 0.01, POSE, 4.0, 0.0, "SWEEP", CHUNK, 0.01) == 0)
    with pytest.raises(PermissionError):
        loop.latch.human_reset(0.1, "ok")
    loop.latch.human_reset(0.095, "RESET")
    _feed(loop.latch, 0.1)
    assert np.any(loop.step(0.1, POSE, 4.0, 0.0, "SWEEP", CHUNK, 0.01) != 0)


def test_reset_refused_while_cause_persists() -> None:
    latch = FaultLatch(0.1, 0.02)
    _feed(latch, 0.0, skip=("ft",))
    latch.tick(0.0)
    with pytest.raises(PermissionError, match="persist"):
        latch.human_reset(0.01, "RESET")


def test_watchdog_miss_and_exception() -> None:
    loop = _loop()
    _feed(loop.latch, 0.0)
    loop.step(0.0, POSE, 4.0, 0.0, "SWEEP", CHUNK, 0.01)
    _feed(loop.latch, 0.05)
    assert np.all(loop.step(0.05, POSE, 4.0, 0.0, "SWEEP", CHUNK, 0.01) == 0)   # 50 ms > 20 ms
    assert "watchdog" in loop.latch.reason
    loop2 = _loop()
    _feed(loop2.latch, 0.0)
    bad_pose = np.array([np.nan] * 6)
    assert np.all(loop2.step(0.0, bad_pose, 4.0, 0.0, "SWEEP", CHUNK, 0.01) == 0)
    assert loop2.latch.faulted and "exception" in loop2.latch.reason
