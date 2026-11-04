"""Teleop gating logic: the robot moves only after the operator's key press. SPEC 12, M5.

State machine per episode: HOLDING (at a station) --key--> MOVING --arrival--> HOLDING.
The gate follows the scheduler's episode plan (usintent.protocol.scheduler.EpisodePlan): at hold
i the only accepted key is the one for the cued exit mode ('s' = SWEEP, 'l' = LIFT); it emits a
``mode_key`` event (key time, mode, station, hold id) and releases exactly one motion command to
the planned next station, carrying the planned sweep speed and the tilt/twist for that station.
A wrong key is logged and ignored. Motion commands are refused in any other state. This module
only decides *whether* and *where* motion may go; executing it on the real robot is subject to
docs/SAFETY.md (M9 controller and launch-file gates).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from usintent.protocol.scheduler import EpisodePlan

KEY_MODES = {"s": "SWEEP", "l": "LIFT"}


class GateError(RuntimeError):
    """A motion command was requested while motion is not released, or to the wrong target."""


@dataclass
class TeleopGate:
    plan: EpisodePlan
    n_stations: int                      # number of preset stations (protocol.yaml)
    i: int = 0                           # index of the current hold in the plan
    state: str = "HOLDING"
    events: list[dict[str, Any]] = field(default_factory=list)
    commands: list[dict[str, Any]] = field(default_factory=list)
    _released: bool = False

    @property
    def hold(self) -> Any:
        return self.plan.holds[self.i]

    def arrived_initial(self, t: float = 0.0) -> dict[str, Any]:
        """Arrival event for the first station (episode start, after the approach)."""
        h = self.hold
        ev = {"stamp": t, "mode": "HOLD", "source": "arrival", "station_index": h.station_index,
              "hold_id": h.hold_id, "sampled_hold_s": float("nan")}
        self.events.append(ev)
        return ev

    def key(self, t: float, key: str) -> dict[str, Any] | None:
        """Operator key press. Returns the mode_key event, or None if ignored."""
        if self.state != "HOLDING" or key not in KEY_MODES:
            return None
        h = self.hold
        if KEY_MODES[key] != h.next_mode:
            self.events.append({"stamp": t, "mode": KEY_MODES[key], "source": "wrong_key",
                                "station_index": h.station_index, "hold_id": h.hold_id,
                                "sampled_hold_s": float("nan")})
            return None
        ev = {"stamp": t, "mode": h.next_mode, "source": "operator_key",
              "station_index": h.station_index, "hold_id": h.hold_id,
              "sampled_hold_s": float("nan")}
        self.events.append(ev)
        self._released = True
        self.state = "MOVING"
        return ev

    def motion_command(self, t: float, target_station: int | None = None) -> dict[str, Any]:
        """The one released motion: to the planned next station (raises GateError otherwise)."""
        if not self._released or self.state != "MOVING":
            raise GateError("motion requested before the operator key press")
        h = self.hold
        target = h.next_station if target_station is None else target_station
        if target != h.next_station or not 0 <= target < self.n_stations:
            raise GateError(f"target {target} is not the planned next station {h.next_station}")
        self._released = False          # one command per key press
        nxt = self.plan.holds[self.i + 1] if self.i + 1 < len(self.plan.holds) else None
        cmd = {"stamp": t, "hold_id": h.hold_id, "target_station": target, "mode": h.next_mode,
               "speed_mps": h.sweep_speed_mps,
               "tilt_deg": nxt.tilt_deg if nxt is not None else 0.0,
               "twist_deg": nxt.twist_deg if nxt is not None else 0.0}
        self.commands.append(cmd)
        return cmd

    def arrived(self, t: float, station: int) -> dict[str, Any]:
        """Robot reached a station: the next hold starts. Returns the arrival event."""
        if self.state != "MOVING":
            raise GateError("arrival reported while not moving")
        if station != self.hold.next_station:
            raise GateError(f"arrived at {station}, expected {self.hold.next_station}")
        self.i += 1
        self.state = "HOLDING" if self.i < len(self.plan.holds) else "DONE"
        ev = {"stamp": t, "mode": "HOLD", "source": "arrival", "station_index": station,
              "hold_id": self.hold.hold_id if self.state == "HOLDING" else -1,
              "sampled_hold_s": float("nan")}
        self.events.append(ev)
        return ev
