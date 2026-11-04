"""Commit rule: HOLD until 1 - S(e) > 0.5. SPEC section 9. M7.

S(e_k) = prod_{j <= k} (1 - h_j) after bin k. While 1 - S <= threshold (0.5, configs/whenhead.yaml
``commit_threshold``) the contact command stays HOLD and the action head is forced to zero motion;
at the first bin start e_k where 1 - S(e_k) > threshold the head switches to the predicted exit
type, so the commit time is T = e_k (relative to hold start).
For timing evaluation on recordings, bins after the key press do not exist (the operator left);
like ``policies.evaluate.hold_timing``, the hold is extended counterfactually by repeating the
features of the last recorded bin with advancing e, so late commits are timed too.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from usintent.whenhead.model import EXIT_CLASSES, WhenHead


def commit_index(hazards: NDArray[np.float64], threshold: float = 0.5) -> int:
    """First bin index k with 1 - prod_{j<=k}(1 - h_j) > threshold; -1 if never."""
    f = 1.0 - np.cumprod(1.0 - np.asarray(hazards, dtype=np.float64))
    hit = np.flatnonzero(f > threshold)
    return int(hit[0]) if len(hit) else -1


class CommitRule:
    """Online use: feed one bin of features at a time; returns (m_hat, force_zero_motion)."""

    def __init__(self, model: WhenHead, threshold: float = 0.5):
        self.model = model
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self.survival = 1.0
        self.committed: str | None = None

    def step(self, features: dict[str, float]) -> tuple[str, bool]:
        if self.committed is not None:
            return self.committed, False
        row = pd.DataFrame([features])
        self.survival *= 1.0 - float(self.model.hazards(row)[0])
        if 1.0 - self.survival > self.threshold:
            self.committed = EXIT_CLASSES[int(np.argmax(self.model.exit_proba(row)[0]))]
            return self.committed, False
        return "HOLD", True


def commit_times(model: WhenHead, bins: pd.DataFrame, bin_width_s: float, threshold: float = 0.5,
                 max_extend_s: float = 20.0) -> pd.DataFrame:
    """Per uncensored hold: commit time T (s from hold start), D, |T - D|, predicted exit type.

    Bins after the last recorded one repeat its features with e advancing by ``bin_width_s``.
    """
    rows: list[dict[str, Any]] = []
    for key, grp in bins.groupby("hold_key", sort=False):
        if bool(grp["censored_hold"].iloc[0]):
            continue
        grp = grp.sort_values("k")
        last = grp.iloc[[-1]]
        n_ext = round(max_extend_s / bin_width_s)
        ext = pd.concat([last] * n_ext, ignore_index=True)
        ext["e"] = float(last["e"].iloc[0]) + bin_width_s * np.arange(1, n_ext + 1)
        allb = pd.concat([grp, ext], ignore_index=True)
        h = model.hazards(allb)
        k = commit_index(h, threshold)
        t = float(allb["e"].iloc[k]) if k >= 0 else np.nan
        d = float(grp["D"].iloc[0])
        exit_pred = EXIT_CLASSES[int(np.argmax(model.exit_proba(allb.iloc[[max(k, 0)]])[0]))]
        rows.append({"hold_key": key, "D": d, "T": t, "err": abs(t - d),
                     "exit_pred": exit_pred, "exit_type": grp["exit_type"].iloc[0]})
    return pd.DataFrame(rows)
