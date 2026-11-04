"""Hybrid force/position controller in the tool frame + control loop (SPEC 12, SAFETY.md). M9.

Normal axis (tool z): admittance law v_z = k_f (F_ref - F), F_ref switched by the contact command
m_hat: HOLD/SWEEP -> force_setpoint_n, RECONTACT -> recontact_setpoint_n, LIFT -> retract at
lift_speed_mps (no force control). The policy never commands force (SAFETY.md principle 6).
Tangential x/y and orientation: track the first waypoint of the policy chunk,
v_xy = chunk[0, :2] / dt + k_p (target - position) style feed-forward; here simply the chunk's
first-step displacement per dt plus k_p times the remaining offset to it. Orientation rates are
passed as chunk[0, 2:5] / dt.
All parameters: configs/control.yaml (nulls refuse). The ControlLoop always routes commands
through LimitFilter and FaultLatch; on any fault or exception it outputs zero.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from usintent.config import require
from usintent.control.fault import FaultLatch
from usintent.control.filter import LimitFilter
from usintent.control.safety import SafetyLimits


@dataclass(frozen=True)
class ControlParams:
    force_setpoint_n: float
    recontact_setpoint_n: float
    admittance_gain_mps_per_n: float
    position_gain_per_s: float
    lift_speed_mps: float

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> ControlParams:
        return cls(*(float(require(cfg, k)) for k in (
            "force_setpoint_n", "recontact_setpoint_n", "admittance_gain_mps_per_n",
            "position_gain_per_s", "lift_speed_mps")))


def control_period_s(control_cfg: dict[str, Any], limits: SafetyLimits) -> float:
    """Control loop period from control.yaml rate_hz (null refuses); must be shorter than the
    watchdog period, or every cycle would trip the watchdog."""
    period = 1.0 / float(require(control_cfg, "rate_hz"))
    if not period < limits.watchdog_period_s:
        raise ValueError(f"control period {period:.4f} s must be < watchdog_period_s "
                         f"{limits.watchdog_period_s} s")
    return period


def hybrid_command(p: ControlParams, m_hat: str, chunk: NDArray[np.float64], force_n: float,
                   dt: float) -> NDArray[np.float64]:
    """Tool-frame twist [vx, vy, vz, wx, wy, wz] (m/s, rad/s)."""
    out = np.zeros(6)
    if m_hat == "LIFT":
        out[2] = -p.lift_speed_mps
        return out
    f_ref = p.recontact_setpoint_n if m_hat == "RECONTACT" else p.force_setpoint_n
    out[2] = p.admittance_gain_mps_per_n * (f_ref - force_n)
    if m_hat != "HOLD":                       # HOLD: tangential motion forced to zero
        out[:2] = chunk[0, :2] / dt * (1.0 + p.position_gain_per_s * dt)
        out[3:] = chunk[0, 2:5] / dt
    return out


class ControlLoop:
    """One control cycle: fault check -> hybrid law -> tool->base -> limit filter -> command."""

    def __init__(self, params: ControlParams, limits: SafetyLimits):
        self.params = params
        self.filter = LimitFilter(limits)
        self.latch = FaultLatch(limits.max_input_age_s, limits.watchdog_period_s)
        self.v_prev: NDArray[np.float64] = np.zeros(3)

    def step(self, now: float, pose: NDArray[np.float64], force_n: float, normal_disp_m: float,
             m_hat: str, chunk: NDArray[np.float64], dt: float) -> NDArray[np.float64]:
        """Returns the base-frame command [vx, vy, vz, wx, wy, wz]; zeros when faulted."""
        try:
            if not self.latch.tick(now):
                self.v_prev = np.zeros(3)
                return np.zeros(6)
            r = Rotation.from_rotvec(pose[3:6])
            tw = hybrid_command(self.params, m_hat, chunk, force_n, dt)
            v_lin = r.apply(tw[:3])
            w = r.apply(tw[3:])
            v_lin = self.filter.apply(now, v_lin, self.v_prev, pose[:3], r.apply([0, 0, 1.0]),
                                      normal_disp_m, force_n, dt)
            if not np.all(np.isfinite(w)):
                raise ValueError("non-finite angular command")
            self.v_prev = v_lin
            return np.concatenate([v_lin, w])
        except Exception as e:  # noqa: BLE001 - any exception latches FAULT and stops (SAFETY 4)
            self.latch.trip(now, f"exception: {type(e).__name__}: {e}")
            self.v_prev = np.zeros(3)
            return np.zeros(6)


@dataclass
class MockPlant:
    """SYNTHETIC mock robot + tissue for tests: the TCP follows the commanded velocity exactly;
    tissue is a linear spring along tool z (F = k * penetration, compressive positive)."""

    pose: NDArray[np.float64]
    surface_z_base: float
    stiffness_n_per_m: float

    def penetration(self) -> float:
        z_tool = Rotation.from_rotvec(self.pose[3:6]).apply([0, 0, 1.0])
        return float(max(0.0, (self.surface_z_base - self.pose[2]) * (-z_tool[2])))

    def force(self) -> float:
        return self.stiffness_n_per_m * self.penetration()

    def step(self, cmd: NDArray[np.float64], dt: float) -> None:
        self.pose = self.pose.copy()
        self.pose[:3] = self.pose[:3] + cmd[:3] * dt
        r = Rotation.from_rotvec(cmd[3:] * dt) * Rotation.from_rotvec(self.pose[3:6])
        self.pose[3:6] = r.as_rotvec()
