"""Scheduler logic: station plan and hold durations per session block, cue timing. SPEC 12, M5.

Deterministic given the seed (logged in SessionInfo). The scheduler samples the *cue offset*
(``sampled_D``, hold start -> cue); the logged D (hold start -> key press) additionally contains
the operator's reaction time (see reports/M1.md, decision "D vs cue").
Every protocol value comes from configs/protocol.yaml; nulls raise MissingRigFact.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from usintent.config import require
from usintent.theory.distributions import HoldDistribution, ShiftedExp, Uniform


def distribution_from_config(proto: dict[str, Any], name: str) -> HoldDistribution:
    """Hold distribution by name ("uniform" | "shifted_exp") from protocol.yaml."""
    if name == "uniform":
        return Uniform(float(require(proto, "hold_distributions.uniform.a")),
                       float(require(proto, "hold_distributions.uniform.b")))
    if name == "shifted_exp":
        return ShiftedExp(float(require(proto, "hold_distributions.shifted_exp.shift")),
                          float(require(proto, "hold_distributions.shifted_exp.scale")))
    raise ValueError(f"unknown hold distribution {name!r}")


@dataclass(frozen=True)
class PlannedHold:
    hold_id: int
    station_index: int
    sampled_d_s: float          # hold start -> cue (s)
    next_mode: str              # SWEEP | LIFT, requested at the cue


@dataclass
class EpisodePlan:
    episode_index: int
    holds: list[PlannedHold]


@dataclass
class SessionPlan:
    session_id: str
    seed: int
    distribution: str
    block_id: int
    operator_id: str
    episodes: list[EpisodePlan] = field(default_factory=list)

    def info(self) -> dict[str, Any]:
        """Fields of the SessionInfo message."""
        return {"session_id": self.session_id, "seed": self.seed,
                "hold_distribution": self.distribution, "block_id": self.block_id,
                "operator_id": self.operator_id}


def plan_session(proto: dict[str, Any], session_id: str, seed: int, distribution: str,
                 block_id: int, operator_id: str, n_episodes: int, p_lift: float = 0.0
                 ) -> SessionPlan:
    """Sample the station count, order and cue offsets for every hold of a session block.

    Station order is 0, 1, 2, ... (stations visited in sequence); ``p_lift`` is the probability
    that the scheduler asks for LIFT instead of SWEEP at a cue (0 unless the protocol says so).
    """
    smin = int(require(proto, "stations_per_episode.min"))
    smax = int(require(proto, "stations_per_episode.max"))
    dist = distribution_from_config(proto, distribution)
    rng = np.random.default_rng(seed)
    plan = SessionPlan(session_id, seed, distribution, block_id, operator_id)
    hid = 0
    for e in range(n_episodes):
        n = int(rng.integers(smin, smax + 1))
        d = dist.sample(n, rng)
        lift = rng.random(n) < p_lift
        holds = [PlannedHold(hid + j, j, float(d[j]), "LIFT" if lift[j] else "SWEEP")
                 for j in range(n)]
        hid += n
        plan.episodes.append(EpisodePlan(e, holds))
    return plan


class CueClock:
    """Runtime cue timing: on arrival at a station start the hold; emit the cue once
    ``sampled_d_s`` has elapsed. Pure logic; the ROS node calls ``tick(now)``."""

    def __init__(self, plan: EpisodePlan):
        self.plan = plan
        self._next = 0
        self._start: float | None = None
        self.cues: list[dict[str, Any]] = []

    @property
    def current(self) -> PlannedHold | None:
        return self.plan.holds[self._next] if self._next < len(self.plan.holds) else None

    def arrived(self, t: float, station_index: int) -> None:
        cur = self.current
        if cur is None:
            raise RuntimeError("no hold left in this episode")
        if station_index != cur.station_index:
            raise ValueError(f"arrived at station {station_index}, plan expects {cur.station_index}")
        self._start = t

    def tick(self, now: float) -> dict[str, Any] | None:
        """Return a cue event dict when the current hold's cue time is reached, else None."""
        cur = self.current
        if cur is None or self._start is None or now < self._start + cur.sampled_d_s:
            return None
        cue = {"stamp": self._start + cur.sampled_d_s, "mode": cur.next_mode,
               "source": "scheduler_cue", "station_index": cur.station_index,
               "hold_id": cur.hold_id, "sampled_hold_s": cur.sampled_d_s}
        self.cues.append(cue)
        self._next += 1
        self._start = None
        return cue
