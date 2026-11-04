"""A7 wrapper: the when-head's m_hat (never intent) is the inner A5-type model's mode input."""
import numpy as np
import pandas as pd
import pytest

from usintent.policies.base import Policy
from usintent.whenhead.commit import WhenHeadPolicy
from usintent.whenhead.features import FeatureParams
from usintent.whenhead.model import WhenHead


class SpyA5(Policy):
    arm_id = "A5"

    def __init__(self):
        self.seen = []

    def _predict(self, history, mode):
        self.seen.append(mode)
        return np.full((5, 5), 0.001), mode


def _model():
    e = np.repeat(np.arange(0, 10, 0.5), 1)
    bins = pd.DataFrame({"hold_key": [f"h{i}" for i in range(len(e))], "k": 0, "e": e,
                         "kappa": 0.5, "s": 1.0, "f": 0.0, "event": (e >= 6).astype(int),
                         "censored_hold": False, "exit_type": "SWEEP", "D": 6.0})
    return WhenHead([8], image=False, seed=0).fit(bins, epochs=200)


def _hist(e):
    return {"bmode": np.full((1, 32, 32), 100, np.uint8), "wrench": np.zeros((2, 6)),
            "time": np.array([e - 0.1, e]), "e_proprio": np.array([e]),
            "state": np.zeros((2, 12)), "dt": np.array([0.1])}


def test_a7_feeds_m_hat_and_forces_zero_while_holding() -> None:
    spy = SpyA5()
    a7 = WhenHeadPolicy(_model(), spy, FeatureParams(0.1, 60.0, "proxy"))
    with pytest.raises(PermissionError):
        a7.predict(_hist(0.0), "HOLD")
    out = [a7.predict(_hist(e)) for e in np.arange(0, 12, 0.1)]
    modes = [m for _, m in out]
    assert modes[0] == "HOLD" and modes[-1] in ("SWEEP", "LIFT")
    for (chunk, m) in out:
        if m == "HOLD":
            assert np.all(chunk == 0)
    assert spy.seen == modes                                   # inner got the when-head's m_hat
    assert not a7.intent_aware


def test_a7_needs_mode_conditioned_inner() -> None:
    from usintent.baselines.b1_zero import ZeroMotion

    with pytest.raises(ValueError):
        WhenHeadPolicy(_model(), ZeroMotion(5), FeatureParams(0.1, 60.0, "proxy"))
