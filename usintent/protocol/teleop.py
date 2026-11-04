"""Teleop gating logic: the robot moves only after the operator's key press. SPEC 12, M5.

State machine per episode: HOLDING (at a station) --key--> MOVING --arrival--> HOLDING.
A key press while HOLDING emits a ``mode_key`` event (key time, mode, station, hold id) and
releases exactly one motion command to the next preset station. Motion commands are refused in
any other state. This module only decides *whether* a target may be sent; executing motion on
the real robot is subject to docs/SAFETY.md (M9 controller / launch-file gates).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

KEY_MODES = {"s": "SWEEP", "l": "LIFT"}


class GateError(RuntimeError):
    """A motion command was requested while motion is not released."""


@dataclass
class TeleopGate:
    n_stations: int
    station: int = 0
    hold_id: int = 0
    state: str = "HOLDING"
    events: list[dict[str, Any]] = field(default_factory=list)
    commands: list[dict[str, Any]] = field(default_factory=list)
    _released: bool = False

    def key(self, t: float, key: str) -> dict[str, Any] | None:
        """Operator key press. Returns the mode_key event, or None if ignored."""
        if self.state != "HOLDING" or key not in KEY_MODES:
            return None
        ev = {"stamp": t, "mode": KEY_MODES[key], "source": "operator_key",
              "station_index": self.station, "hold_id": self.hold_id,
              "sampled_hold_s": float("nan")}
        self.events.append(ev)
        self._released = True
        self.state = "MOVING"
        return ev

    def motion_command(self, t: float, target_station: int) -> dict[str, Any]:
        """Request motion to ``target_station``; raises GateError unless released by a key."""
        if not self._released or self.state != "MOVING":
            raise GateError("motion requested before the operator key press")
        if not 0 <= target_station < self.n_stations:
            raise GateError(f"station {target_station} outside 0..{self.n_stations - 1}")
        self._released = False          # one command per key press
        cmd = {"stamp": t, "target_station": target_station}
        self.commands.append(cmd)
        return cmd

    def arrived_initial(self, t: float = 0.0) -> dict[str, Any]:
        """Arrival event for the first station (episode start, after the approach)."""
        ev = {"stamp": t, "mode": "HOLD", "source": "arrival", "station_index": self.station,
              "hold_id": self.hold_id, "sampled_hold_s": float("nan")}
        self.events.append(ev)
        return ev

    def arrived(self, t: float, station: int) -> dict[str, Any]:
        """Robot reached a station: the next hold starts. Returns the arrival event."""
        if self.state != "MOVING":
            raise GateError("arrival reported while not moving")
        self.station = station
        self.hold_id += 1
        self.state = "HOLDING"
        ev = {"stamp": t, "mode": "HOLD", "source": "arrival", "station_index": station,
              "hold_id": self.hold_id, "sampled_hold_s": float("nan")}
        self.events.append(ev)
        return ev
