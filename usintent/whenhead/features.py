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

import numpy as np
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
