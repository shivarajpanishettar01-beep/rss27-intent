"""Simple vessel segmentation for closed-loop image metrics. SPEC section 5. M3.

Method: Gaussian smoothing, dark threshold (fraction of the frame's coupled-tissue median),
connected components, and the largest dark blob that does not touch the image border (uncoupled
columns and the far field touch the border). Pixel units: the probe-to-image calibration is a
rig fact that is still unknown, so centring error is reported in pixels (lateral axis = columns).
Must be validated against human labels on a subset of real frames before use (SPEC section 5).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage


@dataclass(frozen=True)
class VesselResult:
    mask: NDArray[np.bool_]            # (rows, cols)
    in_view: bool
    centroid_rc: tuple[float, float]   # (row, col) pixels; NaN if not in view
    area_px: int


def segment_vessel(frame: NDArray[np.uint8], dark_frac: float = 0.4, smooth_px: float = 1.5,
                   min_area_px: int = 12) -> VesselResult:
    """Segment the vessel in one B-mode frame (rows = depth, cols = lateral)."""
    img = ndimage.gaussian_filter(frame.astype(np.float64), smooth_px)
    bright = img[img > np.percentile(img, 25)]
    ref = float(np.median(bright)) if bright.size else 0.0
    dark = img < dark_frac * ref
    labels, n = ndimage.label(dark)
    best, best_area = 0, 0
    rows, cols = frame.shape
    for lab in range(1, n + 1):
        m = labels == lab
        rr, cc = np.nonzero(m)
        if rr.min() == 0 or cc.min() == 0 or rr.max() == rows - 1 or cc.max() == cols - 1:
            continue
        if m.sum() > best_area:
            best, best_area = lab, int(m.sum())
    if best == 0 or best_area < min_area_px:
        return VesselResult(np.zeros_like(dark), False, (float("nan"), float("nan")), 0)
    mask = labels == best
    r, c = ndimage.center_of_mass(mask)
    return VesselResult(mask, True, (float(r), float(c)), best_area)


def centring_error_px(result: VesselResult, cols: int) -> float:
    """|vessel centroid column - image centre column| (pixels); NaN if not in view."""
    if not result.in_view:
        return float("nan")
    return abs(result.centroid_rc[1] - (cols - 1) / 2.0)


def iou(a: NDArray[np.bool_], b: NDArray[np.bool_]) -> float:
    """Intersection over union of two masks (for validation against labels)."""
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0
