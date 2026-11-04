"""kappa (confidence), s (coupling), f (force variance), e. SPEC section 9. M7 (partly used in M4).

All features are causal (use only the current and past steps).
- ``coupling_indicator``: fraction of image columns whose near-field mean intensity exceeds a
  threshold (uint8 levels); uncoupled columns go dark. Threshold: configs/whenhead.yaml
  ``coupling_threshold`` (null until measured).
- ``force_variance``: variance of the normal force (N^2) over the last ``window_s`` seconds.
- ``e``: elapsed hold time from proprioception (usintent.data.segmentation.causal_elapsed).
- ``kappa``: mean confidence-map value in the vessel ROI (usintent.whenhead.confidence).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def coupling_indicator(frame: NDArray[np.uint8], threshold: float,
                       near_field_frac: float = 0.15) -> float:
    """Fraction of columns whose mean over the top ``near_field_frac`` rows exceeds threshold."""
    rows = max(1, round(frame.shape[0] * near_field_frac))
    near = frame[:rows].astype(np.float64).mean(axis=0)
    return float(np.mean(near > threshold))


def force_variance(fz: NDArray[np.float64], time: NDArray[np.float64], window_s: float = 1.0
                   ) -> NDArray[np.float64]:
    """Causal rolling variance of fz (N^2) over [t - window_s, t]; 0 for a single sample."""
    out = np.zeros(len(fz))
    c1 = np.concatenate([[0.0], np.cumsum(fz)])
    c2 = np.concatenate([[0.0], np.cumsum(fz**2)])
    lo = np.searchsorted(time, time - window_s, side="left")
    for k in range(len(fz)):
        n = k + 1 - lo[k]
        m = (c1[k + 1] - c1[lo[k]]) / n
        out[k] = max(0.0, (c2[k + 1] - c2[lo[k]]) / n - m * m)
    return out


def depth_profile_features(frame: NDArray[np.uint8], bands: int = 4) -> NDArray[np.float64]:
    """Mean intensity (uint8 levels / 255) in ``bands`` equal depth bands (attenuation proxy)."""
    f = frame.astype(np.float64) / 255.0
    edges = np.linspace(0, frame.shape[0], bands + 1).round().astype(int)
    return np.array([f[edges[i]:edges[i + 1]].mean() for i in range(bands)])


def pooled_embedding(frame: NDArray[np.uint8], grid: int = 4) -> NDArray[np.float64]:
    """Parameter-free frozen image embedding: mean intensity on a grid x grid lattice (/255)."""
    f = frame.astype(np.float64) / 255.0
    re = np.linspace(0, frame.shape[0], grid + 1).round().astype(int)
    ce = np.linspace(0, frame.shape[1], grid + 1).round().astype(int)
    return np.array([f[re[i]:re[i + 1], ce[j]:ce[j + 1]].mean()
                     for i in range(grid) for j in range(grid)])


# ---------------------------------------------------------------------------------------------
# Per-bin feature table for the survival head (SPEC section 9)
# ---------------------------------------------------------------------------------------------



@dataclass(frozen=True)
class FeatureParams:
    bin_width_s: float            # = policy period (configs/whenhead.yaml bin_width_s)
    coupling_threshold: float     # uint8 level (configs/whenhead.yaml coupling_threshold)
    confidence_impl: str = "proxy"   # "karamalis" | "proxy"
    force_window_s: float = 1.0


ROI_DILATE_PX = 2


def vessel_roi(frame: NDArray[np.uint8]) -> NDArray[np.bool_]:
    """Vessel ROI: the segmented lumen dilated by ``ROI_DILATE_PX`` so that the wall and the
    surrounding tissue are included (the anechoic lumen alone carries no signal); the whole
    frame if no vessel is found."""
    from scipy import ndimage

    from usintent.metrics.vessel import segment_vessel

    v = segment_vessel(frame)
    if not v.in_view:
        return np.ones(frame.shape, bool)
    return np.asarray(ndimage.binary_dilation(v.mask, iterations=ROI_DILATE_PX))


def kappa(frame: NDArray[np.uint8], impl: str, roi: NDArray[np.bool_] | None = None) -> float:
    """Mean confidence in the vessel ROI (``roi`` given, or segmented from this frame)."""
    from usintent.whenhead.confidence import karamalis_confidence, proxy_confidence, roi_mean

    conf = karamalis_confidence(frame) if impl == "karamalis" else proxy_confidence(frame)
    return roi_mean(conf, vessel_roi(frame) if roi is None else roi)


def hold_bins(episodes: list[object], p: FeatureParams) -> pd.DataFrame:
    """One row per (hold, bin k): features at the bin start (causal) and the survival target.

    Columns: hold_key, session_id, k, e (s), kappa, s, f, event (1 only in the bin that contains
    the key press), censored_hold, exit_type, D (NaN if censored), start_step.
    The vessel ROI is segmented once, on the hold's first frame (the probe is still).
    Holds come from the logs (cued/synthetic); the hold is observed until its key press or until
    the episode ends (right-censored, no event term).
    """
    rows = []
    for ep in episodes:
        fvar = force_variance(ep.wrench[:, 2], ep.time, p.force_window_s)  # type: ignore[attr-defined]
        stride = max(1, round(p.bin_width_s * ep.fps))  # type: ignore[attr-defined]
        for _, h in ep.holds.iterrows():  # type: ignore[attr-defined]
            steps = np.flatnonzero(ep.hold_id == h["hold_id"])  # type: ignore[attr-defined]
            if not len(steps):
                continue
            a = int(steps[0])
            censored = bool(h["censored"]) or not np.isfinite(h["key_time"])
            d = np.nan if censored else float(h["key_time"] - h["start_time"])
            n_obs = len(steps)
            roi = vessel_roi(ep.bmode[a])  # type: ignore[attr-defined]  # hold-start frame: causal
            for k, s in enumerate(range(a, a + n_obs, stride)):
                e = float(ep.time[s] - h["start_time"])  # type: ignore[attr-defined]
                event = (not censored) and e <= d < e + p.bin_width_s
                frame = ep.bmode[s]  # type: ignore[attr-defined]
                rows.append({"hold_key": f"{ep.episode_index}/{h['hold_id']}",  # type: ignore[attr-defined]
                             "session_id": ep.session_id, "k": k, "e": e,  # type: ignore[attr-defined]
                             "kappa": kappa(frame, p.confidence_impl, roi),
                             "s": coupling_indicator(frame, p.coupling_threshold),
                             "f": float(fvar[s]), "event": int(event), "censored_hold": censored,
                             "exit_type": h["exit_type"], "D": d, "start_step": s})
                if event:
                    break
    return pd.DataFrame(rows)
