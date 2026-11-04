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
