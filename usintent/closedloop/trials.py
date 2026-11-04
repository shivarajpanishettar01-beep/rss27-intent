"""Closed-loop trial runner (M10): fixed station sequences, scheduler-drawn holds, paired arms.

A trial is one episode plan (stations + hold durations) drawn once from the scheduler with a
seed; every arm runs the *same* trials (pairing). The robot is a mock executor that applies the
first step of each commanded chunk, so a policy sees the consequences of its own motion (its
proprioceptive elapsed time e resets when it moves). T for a hold = time from hold start to the
first commanded chunk whose max tangential displacement exceeds ``thr_m``; D = the operator's
intended hold duration. Image and force metrics need a world that renders frames / wrench; the
mock here does not, so those columns are filled only when the caller supplies them (rig logs).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from usintent.metrics.open_loop import max_displacement
from usintent.policies.base import Policy
from usintent.protocol.scheduler import EpisodePlan
from usintent.theory.distributions import HoldDistribution


class ScriptedLeaver(Policy):
    """Blind scripted policy that stays still until its own leave time, then commands a fixed
    departure chunk. Leave time per hold: ``rule`` =
      "median"   -> med(D)                              (Eq. 2 bound)
      "imitator" -> T ~ p_D, drawn at hold start         (calibrated imitator)
      "hedger"   -> T ~ Exp(mean = E[D])                 (memoryless hedger)
    Hold start is detected from proprioception (``e_proprio`` == 0)."""

    def __init__(self, rule: str, dist: HoldDistribution, n_steps: int, step_m: float, seed: int):
        self.arm_id = f"scripted-{rule}"
        self.rule, self.dist = rule, dist
        self.chunk = np.zeros((n_steps, 5))
        self.chunk[:, 0] = step_m * np.arange(1, n_steps + 1)
        self.rng = np.random.default_rng(seed)
        self.leave_at = np.inf

    def _draw(self) -> float:
        if self.rule == "median":
            return self.dist.median()
        if self.rule == "imitator":
            return float(self.dist.sample(1, self.rng)[0])
        if self.rule == "hedger":
            return float(self.rng.exponential(self.dist.mean()))
        raise ValueError(self.rule)

    def _predict(self, history: Any, mode: str | None) -> tuple[NDArray[np.float64], str]:
        e = float(history["e_proprio"][-1])
        if e == 0.0:
            self.leave_at = self._draw()
        if np.isnan(e) or e >= self.leave_at:
            return self.chunk.copy(), "SWEEP"
        return np.zeros_like(self.chunk), "HOLD"


@dataclass(frozen=True)
class TrialConfig:
    dt_s: float                 # policy period
    thr_m: float                # departure threshold
    max_hold_s: float = 40.0    # stop waiting for a commit after this long (T = NaN)


def run_trial(policy: Policy, plan: EpisodePlan, cfg: TrialConfig, trial_id: str,
              hold_durations: NDArray[np.float64] | None = None) -> pd.DataFrame:
    """Run one trial on the mock robot. D per hold = ``hold_durations`` or the plan's sampled_D.

    Returns one row per hold: trial_id, hold_id, station_index, D, T, err, drift_m.
    """
    rows = []
    d_all = (np.array([h.sampled_d_s for h in plan.holds]) if hold_durations is None
             else np.asarray(hold_durations, dtype=np.float64))
    pose = np.zeros(12)
    for h, d in zip(plan.holds, d_all):
        t_first, drift = np.nan, []
        k = 0
        while k * cfg.dt_s <= cfg.max_hold_s:
            e = k * cfg.dt_s
            hist = {"state": np.repeat(pose[None], 2, axis=0), "time": np.array([e - cfg.dt_s, e]),
                    "dt": np.array([cfg.dt_s]), "e_proprio": np.array([e])}
            mode = ("HOLD" if e < d else "SWEEP") if policy.intent_aware else None
            chunk, _ = policy.predict(hist, mode)
            md = float(max_displacement(np.asarray(chunk, dtype=np.float64)[None])[0])
            if md > cfg.thr_m:
                t_first = e
                pose = pose.copy()
                pose[0] += float(chunk[-1, 0])            # executes the departure to the next station
                break
            drift.append(md)
            k += 1
        rows.append({"trial_id": trial_id, "hold_id": h.hold_id, "station_index": h.station_index,
                     "D": float(d), "T": t_first, "err": abs(t_first - d),
                     "drift_m": float(np.mean(drift)) if drift else 0.0})
    return pd.DataFrame(rows)


def run_paired(policies: dict[str, Policy], plans: list[EpisodePlan], cfg: TrialConfig) -> pd.DataFrame:
    """Every arm on the same trials (pairing across arms). Adds an ``arm`` column."""
    out = []
    for name, pol in policies.items():
        for i, plan in enumerate(plans):
            df = run_trial(pol, plan, cfg, f"trial{i}")
            df.insert(0, "arm", name)
            out.append(df)
    return pd.concat(out, ignore_index=True)
