"""Rig calibration from recorded cued episodes: departure threshold and tau (paper Sec. III, VI).

- Departure threshold ``thr``: "set from the zero-motion baseline and the measured controller
  jitter". Recorded chunks that start in a hold and contain no departure are, by construction,
  zero-motion commands executed by the controller; their max tangential displacement is pure
  jitter. thr = ``margin`` x the ``quantile`` of that distribution.
- tau: time from the operator key press until the recorded tangential displacement (relative
  to the pose at the key press, in the tool frame) first exceeds thr. Reported overall and per
  sweep-speed level (tau depends on speed; pi-bar_tau is linear in tau, so the mean tau over
  levels gives the exact mixture prediction).
The script prints these; the human copies them into configs/metrics.yaml (code never writes
configs).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from usintent.data.chunks import ChunkSampler, chunks_from_poses, policy_target


def zero_motion_threshold(sampler: ChunkSampler, quantile: float = 0.999, margin: float = 1.5
                          ) -> dict[str, float]:
    """thr (m) from the jitter of recorded hold chunks without a departure."""
    idx = sampler.index
    rows = idx[(idx["phase"] == "hold") & ~idx["contains_departure"]]
    if len(rows) == 0:
        raise ValueError("no hold chunks without a departure")
    disp = np.linalg.norm(policy_target(sampler.chunks6(rows))[..., :2], axis=-1).max(axis=1)
    q = float(np.quantile(disp, quantile))
    return {"threshold_m": margin * q, "jitter_quantile_m": q, "quantile": quantile,
            "margin": margin, "jitter_median_m": float(np.median(disp)),
            "jitter_max_m": float(disp.max()), "n_chunks": float(len(rows))}


def measure_tau(episodes: list[Any], thr_m: float, max_s: float = 3.0) -> pd.DataFrame:
    """Per uncensored hold: tau (s) from key press to the first threshold crossing."""
    out = []
    for ep in episodes:
        for _, h in ep.holds.iterrows():
            if h["censored"] or not np.isfinite(h["key_time"]):
                continue
            k0 = int(np.searchsorted(ep.time, h["key_time"]))
            n = int(min(round(max_s * ep.fps), len(ep) - 1 - k0))
            if k0 >= len(ep) - 1 or n < 1:
                continue
            ch = chunks_from_poses(ep.state[:, :6], np.array([k0]), n)[0]
            d = np.linalg.norm(ch[:, :2], axis=1)
            hit = np.flatnonzero(d > thr_m)
            tau = float(ep.time[k0 + 1 + hit[0]] - h["key_time"]) if len(hit) else np.nan
            out.append({"episode_index": ep.episode_index, "hold_id": int(h["hold_id"]),
                        "exit_type": h["exit_type"],
                        "sweep_speed_mps": float(h.get("sweep_speed_mps", np.nan)), "tau_s": tau})
    return pd.DataFrame(out)


def tau_summary(taus: pd.DataFrame) -> dict[str, Any]:
    """Overall and per-speed tau statistics; ``tau_s`` = mean over speed levels (or overall)."""
    t = taus["tau_s"].dropna()
    per_speed = {str(k): {"mean": float(g.mean()), "median": float(g.median()), "n": len(g)}
                 for k, g in taus.dropna(subset=["tau_s"]).groupby("sweep_speed_mps")["tau_s"]}
    levels: NDArray[np.float64] = np.array([v["mean"] for v in per_speed.values()])
    return {"tau_s": float(levels.mean()) if len(levels) else float(t.mean()),
            "tau_mean_all_holds_s": float(t.mean()), "tau_median_s": float(t.median()),
            "n_holds": len(t), "n_never_crossed": int(taus["tau_s"].isna().sum()),
            "per_speed": per_speed}
