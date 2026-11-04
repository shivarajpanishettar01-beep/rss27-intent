"""Watchdog and latched FAULT state (SAFETY.md principle 4). M9.

Faults: exception in the loop, any input older than max_input_age_s (stale), NaN/inf in any
input, F/T stream lost, B-mode stream lost (never received or stale), watchdog miss (loop
period > watchdog_period_s). On fault the command is zero velocity and the latch stays set until
``human_reset("RESET")`` is called by an operator; a reset is refused while a cause persists.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

STREAMS = ("ft", "bmode", "pose", "policy")


@dataclass
class FaultLatch:
    max_input_age_s: float
    watchdog_period_s: float
    faulted: bool = False
    reason: str = ""
    history: list[tuple[float, str]] = field(default_factory=list)
    _last: dict[str, float] = field(default_factory=dict)
    _last_tick: float | None = None

    def stamp(self, stream: str, t: float, values: object = None) -> None:
        """Record a new message on ``stream``; NaN/inf in ``values`` latches a fault."""
        if stream not in STREAMS:
            raise KeyError(stream)
        if values is not None and not np.all(np.isfinite(np.asarray(values, dtype=np.float64))):
            self.trip(t, f"non-finite value on {stream}")
            return
        self._last[stream] = t

    def trip(self, t: float, reason: str) -> None:
        if not self.faulted:
            self.faulted, self.reason = True, reason
        self.history.append((t, reason))

    def causes(self, now: float) -> list[str]:
        out = []
        for s in STREAMS:
            if s not in self._last:
                out.append(f"{s} stream never received")
            elif now - self._last[s] > self.max_input_age_s:
                out.append(f"{s} stale ({now - self._last[s]:.3f} s > {self.max_input_age_s} s)")
        return out

    def tick(self, now: float) -> bool:
        """Call once per control cycle; returns True if motion is allowed."""
        if self._last_tick is not None and now - self._last_tick > self.watchdog_period_s:
            self.trip(now, f"watchdog miss ({now - self._last_tick:.3f} s)")
        self._last_tick = now
        for c in self.causes(now):
            self.trip(now, c)
        return not self.faulted

    def human_reset(self, now: float, confirm: str) -> None:
        if confirm != "RESET":
            raise PermissionError("reset needs the operator to type RESET")
        remaining = self.causes(now)
        if remaining:
            raise PermissionError(f"cannot reset, causes persist: {remaining}")
        self.faulted, self.reason = False, ""
        self._last_tick = None
        self.history.append((now, "human reset"))
