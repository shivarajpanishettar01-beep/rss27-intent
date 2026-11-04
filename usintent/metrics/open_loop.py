"""Chunk error, ratio to B2, hold drift, departure-like rate, direction, rate vs e. SPEC section 5.

Inputs are 5-D chunks ``(m, n, 5)`` = [x, y (m), rx, ry, rz (rad)] cumulative displacement from
the chunk-start pose, in the chunk-start tool frame (usintent.data.chunks). "Displacement" in
the hold metrics means the tangential translation |[x, y]| (m); orientation is reported by the
error metrics only.
"""
from __future__ import annotations

from typing import TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import NDArray

Chunks: TypeAlias = NDArray[np.float64]


def _check(pred: Chunks, gt: Chunks | None = None) -> None:
    if pred.ndim != 3 or pred.shape[-1] != 5:
        raise ValueError(f"chunks must be (m, n, 5), got {pred.shape}")
    if gt is not None and gt.shape != pred.shape:
        raise ValueError(f"shape mismatch {pred.shape} vs {gt.shape}")


def chunk_error_trans(pred: Chunks, gt: Chunks) -> NDArray[np.float64]:
    """Per chunk: mean over steps of the L2 translation error |dxy| (m). Shape (m,)."""
    _check(pred, gt)
    return np.asarray(np.linalg.norm(pred[..., :2] - gt[..., :2], axis=-1).mean(axis=1))


def chunk_error_rot(pred: Chunks, gt: Chunks) -> NDArray[np.float64]:
    """Per chunk: mean over steps of the L2 rotation-vector error (rad). Shape (m,)."""
    _check(pred, gt)
    return np.asarray(np.linalg.norm(pred[..., 2:] - gt[..., 2:], axis=-1).mean(axis=1))


def max_displacement(pred: Chunks) -> NDArray[np.float64]:
    """Per chunk: max over steps of |[x, y]| (m). Shape (m,)."""
    _check(pred)
    return np.asarray(np.linalg.norm(pred[..., :2], axis=-1).max(axis=1))


def ratio_to_cv(err_policy: NDArray[np.float64], err_cv: NDArray[np.float64],
                phase: NDArray[np.str_], near_zero_m: float = 1e-4) -> pd.DataFrame:
    """Ratio of means (policy / B2) per phase, on the same chunks.

    Returns one row per phase: n, mean_policy, mean_cv, ratio, unstable (B2 mean error below
    ``near_zero_m``, typical of holds where B2 is nearly exact, so the ratio is unstable).
    """
    if not (err_policy.shape == err_cv.shape == phase.shape):
        raise ValueError("inputs must share shape (m,)")
    df = pd.DataFrame({"phase": phase, "p": err_policy, "cv": err_cv})
    g = df.groupby("phase", sort=True)
    out = pd.DataFrame({"n": g.size(), "mean_policy": g["p"].mean(), "mean_cv": g["cv"].mean()})
    out["ratio"] = out["mean_policy"] / out["mean_cv"].where(out["mean_cv"] > 0)
    out["unstable"] = out["mean_cv"] < near_zero_m
    return out.reset_index()


def hold_drift(pred: Chunks, in_hold: NDArray[np.bool_], contains_departure: NDArray[np.bool_]
               ) -> float:
    """Mean predicted displacement (m) over chunks that start in a hold and contain no departure.

    Displacement of one chunk = mean over its steps of |[x, y]|.
    """
    _check(pred)
    sel = in_hold & ~contains_departure
    if not np.any(sel):
        return float("nan")
    return float(np.linalg.norm(pred[sel, :, :2], axis=-1).mean())


def departure_like(pred: Chunks, thr_m: float) -> NDArray[np.bool_]:
    """Per chunk: predicted max displacement exceeds ``thr_m`` (m)."""
    if thr_m <= 0:
        raise ValueError("threshold must be positive")
    return np.asarray(max_displacement(pred) > thr_m)


def departure_like_rate(pred: Chunks, in_hold: NDArray[np.bool_], thr_m: float,
                        exit_type: NDArray[np.str_] | None = None) -> dict[str, float]:
    """Fraction of hold chunks predicted departure-like; overall ("all") and per exit type."""
    flags = departure_like(pred, thr_m)
    out = {"all": float(flags[in_hold].mean()) if np.any(in_hold) else float("nan")}
    if exit_type is not None:
        for et in np.unique(exit_type[in_hold]):
            m = in_hold & (exit_type == et)
            out[str(et)] = float(flags[m].mean())
    return out


def departure_direction(pred: Chunks, exit_dir_xy: NDArray[np.float64], min_disp_m: float = 0.0
                        ) -> NDArray[np.float64]:
    """Per chunk: cosine between the predicted final [x, y] displacement and the ground-truth
    exit direction (unit or non-unit (m, 2) vectors in the chunk-start tool frame). NaN where the
    prediction is shorter than ``min_disp_m`` or the direction is zero. Exclude back-tracking
    holds before calling (SPEC section 5)."""
    _check(pred)
    p = pred[:, -1, :2]
    pn = np.linalg.norm(p, axis=1)
    dn = np.linalg.norm(exit_dir_xy, axis=1)
    ok = (pn > max(min_disp_m, 0.0)) & (pn > 0) & (dn > 0)
    cos = np.full(len(p), np.nan)
    cos[ok] = np.sum(p[ok] * exit_dir_xy[ok], axis=1) / (pn[ok] * dn[ok])
    return cos


def rate_vs_e(flags: NDArray[np.bool_], e: NDArray[np.float64], bin_edges: NDArray[np.float64]
              ) -> pd.DataFrame:
    """Departure-like rate binned by elapsed hold time e (s). Rows: e_lo, e_hi, n, rate."""
    if flags.shape != e.shape:
        raise ValueError("flags and e must share shape")
    idx = np.digitize(e, bin_edges) - 1
    rows = []
    for b in range(len(bin_edges) - 1):
        m = idx == b
        rows.append({"e_lo": float(bin_edges[b]), "e_hi": float(bin_edges[b + 1]),
                     "n": int(m.sum()), "rate": float(flags[m].mean()) if m.any() else np.nan})
    return pd.DataFrame(rows)


def first_departure_time(times: NDArray[np.float64], flags: NDArray[np.bool_]) -> float:
    """Time (s) of the first step whose prediction is departure-like; NaN if none."""
    hit = np.flatnonzero(flags)
    return float(times[hit[0]]) if len(hit) else float("nan")
