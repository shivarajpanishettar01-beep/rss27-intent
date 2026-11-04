"""B2 constant velocity / Kalman. SPEC section 6. M3.

Velocity = finite difference between the first and last pose of the history window, expressed in
the tool frame of the current pose, extrapolated over the horizon. Needs history_k >= 2.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from usintent.policies.base import Policy


def cv_chunk(state: NDArray[np.float64], time: NDArray[np.float64], n_steps: int, dt: float
             ) -> NDArray[np.float64]:
    """(n, 5) constant-velocity chunk from a (k, 12) state history; zeros if k < 2."""
    if len(state) < 2 or time[-1] <= time[0]:
        return np.zeros((n_steps, 5))
    span = time[-1] - time[0]
    r_now = Rotation.from_rotvec(state[-1, 3:6])
    v_lin = r_now.inv().apply((state[-1, :3] - state[0, :3]) / span)          # tool frame, m/s
    w = (Rotation.from_rotvec(state[0, 3:6]).inv() * r_now).as_rotvec() / span  # rad/s
    steps = (np.arange(1, n_steps + 1) * dt)[:, None]
    out = np.zeros((n_steps, 5))
    out[:, :2] = steps * v_lin[:2]
    out[:, 2:] = steps * w
    return out


class ConstantVelocity(Policy):
    arm_id = "B2"

    def __init__(self, n_steps: int, moving_thr_mps: float = 0.0):
        self.n_steps = n_steps
        self.moving_thr_mps = moving_thr_mps

    def _predict(self, history: Any, mode: str | None) -> tuple[NDArray[np.float64], str]:
        dt = float(history["dt"][0])
        ch = cv_chunk(history["state"], history["time"], self.n_steps, dt)
        moving = np.linalg.norm(ch[-1, :2]) / (self.n_steps * dt) > self.moving_thr_mps
        return ch, "SWEEP" if moving else "HOLD"
