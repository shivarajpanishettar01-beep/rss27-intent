"""Phase segmentation: cued (from logs) and Open-H (thresholds from configs/data.yaml). SPEC 4.

Phases per step: ``approach`` (before first contact), ``contact_onset`` (first T_onset s after
contact), ``hold``, ``sweep`` (the rest during/after contact). ``in_transition`` flags steps
within +-w s of any phase change; it is a flag, not a phase, so hold chunks that contain the
departure keep their ``hold`` label.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from usintent.config import require
from usintent.data.loaders import Episode

PHASES = ("approach", "contact_onset", "hold", "sweep")


@dataclass(frozen=True)
class SegParams:
    """Threshold-segmentation parameters (all from configs/data.yaml for real data)."""

    v_hold_mps: float          # translational speed below which the probe is holding (m/s)
    t_hold_min_s: float        # minimum hold duration (s)
    t_onset_s: float           # length of the contact-onset phase (s)
    transition_window_s: float  # +-w around phase changes (s)
    contact_force_n: float | None = None  # |Fz| above this = contact; None -> motion proxy

    def scaled(self, factor: float) -> SegParams:
        """All thresholds multiplied by ``factor`` (for the +-50 % sensitivity analysis)."""
        return SegParams(self.v_hold_mps * factor, self.t_hold_min_s * factor,
                         self.t_onset_s * factor, self.transition_window_s,
                         None if self.contact_force_n is None else self.contact_force_n * factor)


def params_from_config(cfg: dict[str, Any]) -> SegParams:
    """Read SegParams from the loaded data config; nulls raise MissingRigFact."""
    source = str(require(cfg, "segmentation.contact_source"))
    if source not in ("force", "motion"):
        raise ValueError('segmentation.contact_source must be "force" or "motion"')
    cf = float(require(cfg, "segmentation.contact_force_n")) if source == "force" else None
    return SegParams(
        v_hold_mps=float(require(cfg, "segmentation.v_hold_mps")),
        t_hold_min_s=float(require(cfg, "segmentation.t_hold_min_s")),
        t_onset_s=float(require(cfg, "segmentation.t_onset_s")),
        transition_window_s=float(require(cfg, "segmentation.transition_window_s")),
        contact_force_n=cf,
    )


@dataclass
class Segmentation:
    """Per-step phase labels plus hold intervals [start, end) in steps."""

    phase: NDArray[np.str_]
    in_transition: NDArray[np.bool_]
    holds: list[tuple[int, int]]          # [start_step, end_step); end = first step not in hold
    elapsed: NDArray[np.float64]          # e (s) since hold start; NaN outside holds
    hold_index: NDArray[np.int64]         # index into ``holds``; -1 outside holds
    source: str = "cued"                  # "cued" (from logs) or "threshold" (Open-H style)

    @property
    def hold_mask(self) -> NDArray[np.bool_]:
        return np.asarray(self.hold_index >= 0)


def translational_speed(ep: Episode) -> NDArray[np.float64]:
    """|d p / dt| of the TCP position in base (m/s), central differences."""
    if len(ep) < 2:
        return np.zeros(len(ep))
    v = np.gradient(ep.state[:, :3], ep.time, axis=0)
    return np.asarray(np.linalg.norm(v, axis=1), dtype=np.float64)


def _runs(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    """[start, end) index runs where mask is True."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def _finish(ep: Episode, holds: list[tuple[int, int]], contact_step: int, t_onset: float,
            w: float, source: str) -> Segmentation:
    n = len(ep)
    phase = np.full(n, "sweep", dtype="<U13")
    phase[:contact_step] = "approach"
    onset_end = np.searchsorted(ep.time, ep.time[min(contact_step, n - 1)] + t_onset)
    phase[contact_step:onset_end] = "contact_onset"
    hold_index = np.full(n, -1, dtype=np.int64)
    elapsed = np.full(n, np.nan)
    for j, (a, b) in enumerate(holds):
        phase[a:b] = "hold"
        hold_index[a:b] = j
        elapsed[a:b] = ep.time[a:b] - ep.time[a]
    change_t = ep.time[1:][phase[1:] != phase[:-1]]
    in_tr = np.zeros(n, dtype=bool)
    for tc in change_t:
        in_tr |= np.abs(ep.time - tc) <= w
    return Segmentation(phase, in_tr, holds, elapsed, hold_index, source)


def segment_cued(ep: Episode, transition_window_s: float) -> Segmentation:
    """Phases from logged modes and key times (cued data): a hold is hold start .. key press."""
    holds = []
    for hid in ep.holds["hold_id"]:
        idx = np.flatnonzero(ep.hold_id == hid)
        if len(idx):
            holds.append((int(idx[0]), int(idx[-1]) + 1))
    contact_step = holds[0][0] if holds else len(ep)
    return _finish(ep, holds, contact_step, 0.0, transition_window_s, "cued")


def segment_threshold(ep: Episode, p: SegParams) -> Segmentation:
    """Open-H style segmentation without mode labels (SPEC section 4)."""
    speed = translational_speed(ep)
    slow = speed < p.v_hold_mps
    if p.contact_force_n is not None:
        contact = np.abs(ep.wrench[:, 2]) > p.contact_force_n
        contact_step = int(np.argmax(contact)) if contact.any() else len(ep)
    else:
        contact = np.ones(len(ep), dtype=bool)
        contact_step = -1   # motion proxy: first detected hold start (set below)
    holds = []
    for a, b in _runs(slow & contact):
        long_enough = ep.time[b - 1] - ep.time[a] + ep.dt >= p.t_hold_min_s
        at_start = a == 0 and p.contact_force_n is None   # resting before the approach
        if long_enough and not at_start:
            holds.append((a, b))
    if contact_step < 0:
        contact_step = holds[0][0] if holds else len(ep)
    holds = [(a, b) for a, b in holds if a >= contact_step]
    return _finish(ep, holds, contact_step, p.t_onset_s, p.transition_window_s, "threshold")


def hold_f1(pred: NDArray[np.bool_], truth: NDArray[np.bool_]) -> float:
    """Step-level F1 of hold labels."""
    tp = float(np.sum(pred & truth))
    fp = float(np.sum(pred & ~truth))
    fn = float(np.sum(~pred & truth))
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def sensitivity(episodes: list[Episode], base: SegParams,
                factors: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5),
                horizon_s: float = 1.0) -> list[dict[str, float]]:
    """Segmentation results under scaled thresholds (SPEC: +-50 %).

    Per factor: number of holds, fraction of time in hold, mean hold duration (s), departure
    fraction of hold chunks (measured pi-bar), and step-level hold F1 against factor 1.0.
    """
    ref = np.concatenate([segment_threshold(e, base).hold_mask for e in episodes])
    out = []
    for f in factors:
        p = base.scaled(f)
        segs = [segment_threshold(e, p) for e in episodes]
        mask = np.concatenate([s.hold_mask for s in segs])
        durs, dep, tot = [], 0, 0
        for e, s in zip(episodes, segs):
            for a, b in s.holds:
                durs.append(e.time[b - 1] - e.time[a] + e.dt)
                if b < len(e):                        # hold ended inside the episode
                    starts = np.arange(a, b)
                    dep += int(np.sum(e.time[b] < e.time[starts] + horizon_s))
                    tot += len(starts)
        out.append({"factor": f, "n_holds": float(len(durs)),
                    "hold_time_fraction": float(mask.mean()),
                    "mean_hold_s": float(np.mean(durs)) if durs else float("nan"),
                    "departure_fraction": dep / tot if tot else float("nan"),
                    "f1_vs_base": hold_f1(mask, ref)})
    return out


def causal_elapsed(ep: Episode, v_hold_mps: float) -> NDArray[np.float64]:
    """Elapsed hold time e (s) from proprioception only, causally: time since the TCP speed
    (backward difference) last fell below ``v_hold_mps``; NaN while moving. SPEC section 9."""
    n = len(ep)
    e = np.full(n, np.nan)
    if n == 0:
        return e
    v = np.zeros(n)
    v[1:] = np.linalg.norm(np.diff(ep.state[:, :3], axis=0), axis=1) / np.diff(ep.time)
    since = None
    for k in range(n):
        if v[k] < v_hold_mps:
            since = ep.time[k] if since is None else since
            e[k] = ep.time[k] - since
        else:
            since = None
    return e
