"""CBF / limit filter between the policy+controller and the robot (SAFETY.md principle 5). M9.

Input: a raw Cartesian velocity command for the TCP (base frame, m/s, linear part; angular part
passed through unchanged except for zeroing on fault), the current TCP position (base, m), the
previous commanded velocity, the probe normal displacement from the contact reference along tool
z (m, positive = into tissue), the normal contact force (N, compressive positive), the tool z
axis in base, and the step dt (s). Order of constraints:
  1. speed: |v| <= max_tcp_speed (scale)
  2. acceleration: |v - v_prev| <= max_tcp_accel * dt (clip the change)
  3. workspace box (discrete-time barrier): x + v dt stays inside the box
  4. normal displacement barrier: d + (v . z_tool) dt <= max_normal_displacement
  5. force: if F >= max_contact_force, no velocity component into tissue
Constraints 3-5 are hard and may require braking harder than the acceleration limit; that case
is logged as constraint "accel_override". Every intervention is logged with stamp, the raw and
the filtered command.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from usintent.control.safety import SafetyLimits


@dataclass(frozen=True)
class FilterEvent:
    stamp: float
    constraint: str
    raw_command: tuple[float, ...]
    filtered_command: tuple[float, ...]


@dataclass
class LimitFilter:
    limits: SafetyLimits
    events: list[FilterEvent] = field(default_factory=list)

    def _log(self, t: float, name: str, raw: NDArray[np.float64], new: NDArray[np.float64]) -> None:
        self.events.append(FilterEvent(t, name, tuple(map(float, raw)), tuple(map(float, new))))

    def apply(self, t: float, v_raw: NDArray[np.float64], v_prev: NDArray[np.float64],
              pos: NDArray[np.float64], z_tool: NDArray[np.float64], normal_disp_m: float,
              force_n: float, dt: float) -> NDArray[np.float64]:
        lim = self.limits
        if not (np.all(np.isfinite(v_raw)) and np.all(np.isfinite(pos)) and np.isfinite(force_n)):
            raise ValueError("non-finite filter input")      # caller latches a FAULT
        v = np.asarray(v_raw, dtype=np.float64).copy()
        sp = float(np.linalg.norm(v))
        if sp > lim.max_tcp_speed_mps:
            new = v * (lim.max_tcp_speed_mps / sp)
            self._log(t, "speed", v, new)
            v = new
        dv = v - v_prev
        dmax = lim.max_tcp_accel_mps2 * dt
        if np.linalg.norm(dv) > dmax:
            new = v_prev + dv * (dmax / np.linalg.norm(dv))
            self._log(t, "accel", v, new)
            v = new
        before_hard = v.copy()
        box = np.asarray(lim.workspace_box_base_m).reshape(3, 2)
        lo, hi = (box[:, 0] - pos) / dt, (box[:, 1] - pos) / dt
        z = np.asarray(z_tool, dtype=np.float64) / np.linalg.norm(z_tool)
        vin_max = (lim.max_normal_displacement_m - normal_disp_m) / dt
        force_hit = force_n >= lim.max_contact_force_n
        if force_hit:
            vin_max = min(vin_max, 0.0)

        def ok(u: NDArray[np.float64]) -> bool:
            return bool(np.all(u >= lo - 1e-12) and np.all(u <= hi + 1e-12)
                        and u @ z <= vin_max + 1e-12
                        and np.linalg.norm(u) <= lim.max_tcp_speed_mps * (1 + 1e-12))

        for _ in range(20):                     # alternate projections onto the hard sets
            if ok(v):
                break
            new = np.clip(v, lo, hi)
            if not np.array_equal(new, v):
                self._log(t, "workspace", v, new)
                v = new
            vin = float(v @ z)
            if vin > vin_max:
                new = v - (vin - vin_max) * z
                self._log(t, "force" if force_hit else "normal_displacement", v, new)
                v = new
            sp = float(np.linalg.norm(v))
            if sp > lim.max_tcp_speed_mps:
                new = v * (lim.max_tcp_speed_mps / sp)
                self._log(t, "speed", v, new)
                v = new
        if not ok(v):
            new = np.zeros_like(v)              # zero is admissible inside the safe set
            self._log(t, "fallback_zero", v, new)
            v = new
        if not np.array_equal(v, before_hard) and np.linalg.norm(v - v_prev) > dmax * (1 + 1e-9):
            self._log(t, "accel_override", before_hard, v)
        return v
