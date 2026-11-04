"""B1 zero motion. SPEC section 6. M3."""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from usintent.policies.base import Policy


class ZeroMotion(Policy):
    """Always predicts a zero (n, 5) chunk and m_hat = HOLD."""

    arm_id = "B1"

    def __init__(self, n_steps: int):
        self.n_steps = n_steps

    def _predict(self, history: Any, mode: str | None) -> tuple[NDArray[np.float64], str]:
        return np.zeros((self.n_steps, 5)), "HOLD"
