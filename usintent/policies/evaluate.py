"""Open-loop evaluation harness shared by baselines (M3) and policies (M8). SPEC 5, 10.

Rules enforced here:
- CLAUDE.md rule 4: :func:`evaluate_open_loop` runs the ground-truth passthrough check first and
  refuses to score if it fails.
- Intent gating: the scheduler mode is passed only to intent-aware arms (policies/base.py);
  histories never contain it.
- Elapsed hold time for blind models comes from proprioception only (``e_proprio``).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from usintent.data.chunks import ChunkSampler, policy_target
from usintent.data.passthrough import passthrough_check
from usintent.data.segmentation import causal_elapsed
from usintent.metrics import open_loop as ol
from usintent.policies.base import Policy


class EvalContext:
    """Per-episode proprioceptive extras (e_proprio) for building policy histories."""

    def __init__(self, sampler: ChunkSampler, v_hold_mps: float | None):
        self.sampler = sampler
        self.e_proprio: dict[int, NDArray[np.float64]] = {}
        if v_hold_mps is not None:
            self.e_proprio = {e.episode_index: causal_elapsed(e, v_hold_mps)
                              for e in sampler.episodes}
        self._eps = {e.episode_index: e for e in sampler.episodes}

    def history(self, episode_index: int, start: int) -> dict[str, Any]:
        h: dict[str, Any] = self.sampler.history(episode_index, start)
        if self.e_proprio:
            h["e_proprio"] = self.e_proprio[episode_index][start:start + 1]
        return h

    def scheduler_mode(self, episode_index: int, start: int) -> str:
        return str(self._eps[episode_index].mode[start])


def predict_chunks(policy: Policy, ctx: EvalContext, rows: pd.DataFrame
                   ) -> tuple[NDArray[np.float64], list[str]]:
    """Run ``policy`` at every chunk start in ``rows`` (in time order per episode).

    Returns (m, n, 5) chunks and the contact commands m_hat.
    """
    chunks, m_hat = [], []
    for ep_idx, start in zip(rows["episode_index"].to_numpy(), rows["start"].to_numpy()):
        h = ctx.history(int(ep_idx), int(start))
        mode = ctx.scheduler_mode(int(ep_idx), int(start)) if policy.intent_aware else None
        c, m = policy.predict(h, mode)
        chunks.append(np.asarray(c, dtype=np.float64))
        m_hat.append(m)
    return np.stack(chunks), m_hat


def hold_timing(policy: Policy, ctx: EvalContext, thr_m: float, horizon_after_key_s: float = 12.0
                ) -> pd.DataFrame:
    """Timing per cued hold: T = time from hold start to the first step whose predicted chunk is
    departure-like (SPEC section 5); D from the sidecar. Columns: hold_id, D, T, err.

    Recorded steps are used until the key press. After it, the recorded robot moves on, so a
    policy that would have committed *late* cannot be timed from the recording. The hold is
    therefore extended counterfactually: the last in-hold observation (pose, wrench, frame) is
    repeated with advancing time and e_proprio, up to ``horizon_after_key_s``. This is exact for
    policies that depend on pose, force and e (B1-B4, B6) and an approximation for image policies
    (their true late behaviour is measured in closed loop, M10).
    """
    rows = []
    for ep in ctx.sampler.episodes:
        n = ctx.sampler.n_steps(ep)
        for _, h in ep.holds.iterrows():
            if h["censored"]:
                continue
            steps = np.flatnonzero(ep.hold_id == h["hold_id"])
            steps = steps[steps + n < len(ep)]
            t_first = np.nan
            for k in steps:
                c, _ = _predict_at(policy, ctx, ep.episode_index, int(k), ep, 0.0)
                if ol.max_displacement(c[None])[0] > thr_m:
                    t_first = float(ep.time[k] - h["start_time"])
                    break
            if np.isnan(t_first) and len(steps):
                last = int(steps[-1])
                for j in range(1, round(horizon_after_key_s * ep.fps) + 1):
                    c, _ = _predict_at(policy, ctx, ep.episode_index, last, ep, j * ep.dt)
                    if ol.max_displacement(c[None])[0] > thr_m:
                        t_first = float(ep.time[last] + j * ep.dt - h["start_time"])
                        break
            rows.append({"episode_index": ep.episode_index, "hold_id": int(h["hold_id"]),
                         "D": float(h["D"]), "T": t_first})
    df = pd.DataFrame(rows)
    df["err"] = np.abs(df["T"] - df["D"])
    return df


def _predict_at(policy: Policy, ctx: EvalContext, ep_idx: int, k: int, ep: Any, shift_s: float
                ) -> tuple[NDArray[np.float64], str]:
    """Predict at step k; with shift_s > 0, the observation at k is replayed shift_s later."""
    hist = ctx.history(ep_idx, k)
    if shift_s > 0:
        hist = dict(hist)
        hist["state"] = np.repeat(hist["state"][-1:], len(hist["state"]), axis=0)
        hist["wrench"] = np.repeat(hist["wrench"][-1:], len(hist["wrench"]), axis=0)
        hist["time"] = hist["time"][-1] + shift_s - (len(hist["time"]) - 1 - np.arange(
            len(hist["time"]))) * ep.dt
        if "bmode" in hist:
            hist["bmode"] = np.repeat(hist["bmode"][-1:], len(hist["bmode"]), axis=0)
        if "e_proprio" in hist:
            hist["e_proprio"] = hist["e_proprio"] + shift_s
    # in the extension the scheduler mode stays what it was at the last recorded hold step
    mode = str(ep.mode[k]) if policy.intent_aware else None
    c, m = policy.predict(hist, mode)
    return np.asarray(c, dtype=np.float64), m


def evaluate_open_loop(policy: Policy, ctx: EvalContext, rows: pd.DataFrame, thr_m: float,
                       cv_chunks: NDArray[np.float64] | None = None) -> dict[str, Any]:
    """Passthrough first (raises on failure), then the SPEC section 5 open-loop metrics."""
    pt = passthrough_check(ctx.sampler, max_chunks=5000)
    pred, _ = predict_chunks(policy, ctx, rows)
    gt = policy_target(ctx.sampler.chunks6(rows))
    in_hold = (rows["phase"] == "hold").to_numpy()
    dep = rows["contains_departure"].to_numpy(dtype=bool)
    err_t = ol.chunk_error_trans(pred, gt)
    out: dict[str, Any] = {
        "passthrough": pt,
        "n_chunks": len(rows),
        "chunk_error_trans_m": float(err_t.mean()),
        "chunk_error_rot_rad": float(ol.chunk_error_rot(pred, gt).mean()),
        "hold_drift_m": ol.hold_drift(pred, in_hold, dep),
        "departure_like_rate": ol.departure_like_rate(pred, in_hold, thr_m,
                                                      rows["exit_type"].to_numpy().astype(str)),
        "gt_departure_fraction": float(dep[in_hold].mean()) if in_hold.any() else float("nan"),
    }
    if cv_chunks is not None:
        out["ratio_to_cv"] = ol.ratio_to_cv(err_t, ol.chunk_error_trans(cv_chunks, gt),
                                            rows["phase"].to_numpy().astype(str))
    return out
