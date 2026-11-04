"""B6 wrapper: zero output whenever scheduler mode is HOLD. SPEC section 6. M3.

B6 is intent-aware (policies/base.py INTENT_AWARE_ARMS): it receives the scheduler mode; the
wrapped policy is always called WITHOUT the mode, so a blind inner policy stays blind.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from usintent.policies.base import Policy


class StopOnHold(Policy):
    arm_id = "B6"

    def __init__(self, inner: Policy, n_steps: int):
        self.inner = inner
        self.n_steps = n_steps

    def _predict(self, history: Any, mode: str | None) -> tuple[NDArray[np.float64], str]:
        if mode is None:
            raise ValueError("B6 needs the scheduler mode")
        if mode == "HOLD":
            return np.zeros((self.n_steps, 5)), "HOLD"
        chunk, m_hat = self.inner.predict(history)
        return np.asarray(chunk, dtype=np.float64), m_hat
