"""Timing error |T-D|, drift, vessel in view, force in band, interventions, success. SPEC 5. M10.

The per-hold timing error is defined here (M3) because B4 is checked against the Eq. 2 bound;
the remaining closed-loop metrics are filled in M10.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def timing_error(t_motion: NDArray[np.float64], d_hold: NDArray[np.float64]) -> NDArray[np.float64]:
    """|T - D| per hold (s). T = time of first commanded motion above thr after hold start,
    D = hold duration (start to key press), both relative to hold start. NaN T (no motion during
    the evaluated window) propagates as NaN."""
    t = np.asarray(t_motion, dtype=np.float64)
    d = np.asarray(d_hold, dtype=np.float64)
    if t.shape != d.shape:
        raise ValueError("shape mismatch")
    return np.asarray(np.abs(t - d))


def hold_drift_closed(disp_before_commit_m: NDArray[np.float64]) -> float:
    """Mean |commanded tangential displacement| (m) while the hold is still active (before T)."""
    d = np.asarray(disp_before_commit_m, dtype=np.float64)
    return float(np.mean(d)) if d.size else float("nan")


def fraction_true(flags: NDArray[np.bool_]) -> float:
    """Fraction of time steps for which a condition holds (vessel in view, force in band)."""
    f = np.asarray(flags, dtype=bool)
    return float(f.mean()) if f.size else float("nan")


def force_in_band(force_n: NDArray[np.float64], band_n: tuple[float, float]) -> NDArray[np.bool_]:
    """Per step: band[0] <= F <= band[1] (band from metrics.yaml force_band_n)."""
    lo, hi = band_n
    if not lo < hi:
        raise ValueError("force band must be [min, max] with min < max")
    f = np.asarray(force_n, dtype=np.float64)
    return np.asarray((f >= lo) & (f <= hi))


def trial_success(view_frac_per_hold: NDArray[np.float64], force_ok_all: NDArray[np.bool_],
                  p_view: float) -> bool:
    """All stations imaged with the vessel in view >= p_view of each hold, and the force never
    left the band during the trial (SPEC section 5)."""
    v = np.asarray(view_frac_per_hold, dtype=np.float64)
    return bool(v.size > 0 and np.all(v >= p_view) and np.all(np.asarray(force_ok_all, dtype=bool)))
