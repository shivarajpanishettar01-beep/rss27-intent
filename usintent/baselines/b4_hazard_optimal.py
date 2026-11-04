"""B4 hazard-optimal blind predictor (leaves at med(D)). SPEC section 6. M3.

Knows p_D (or the training median) and the proprioceptive elapsed hold time e
(``history["e_proprio"]``, computed causally from robot state only). Predicts zero motion while
e < med(D), then the mean training departure chunk of the most frequent exit type. Its timing
error attains Eq. 2, E|D - med(D)|.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

from usintent.policies.base import Policy


class HazardOptimalBlind(Policy):
    arm_id = "B4"

    def __init__(self, median_d_s: float, departure_chunk: NDArray[np.float64], exit_type: str):
        self.median_d_s = median_d_s
        self.departure_chunk = np.asarray(departure_chunk, dtype=np.float64)
        self.exit_type = exit_type

    @classmethod
    def fit(cls, median_d_s: float, departure_chunks: NDArray[np.float64],
            exit_types: list[str]) -> HazardOptimalBlind:
        """departure_chunks: (m, n, 5) training chunks starting at the step of each key press."""
        top = Counter(exit_types).most_common(1)[0][0]
        sel = np.array([t == top for t in exit_types])
        return cls(median_d_s, departure_chunks[sel].mean(axis=0), top)

    def _predict(self, history: Any, mode: str | None) -> tuple[NDArray[np.float64], str]:
        e = float(history["e_proprio"][-1])
        # NaN e = probe is moving (not in a hold); only hold behaviour is specified, and B4 is
        # scored on holds only. While holding: zero motion until e reaches the median.
        if not np.isnan(e) and e < self.median_d_s:
            return np.zeros_like(self.departure_chunk), "HOLD"
        return self.departure_chunk.copy(), self.exit_type
