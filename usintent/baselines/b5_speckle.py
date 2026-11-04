"""B5 speckle-decorrelation motion estimate. SPEC section 6. M3.

Out-of-plane (elevational, tool x) displacement between consecutive frames from the normalized
cross-correlation rho of speckle patches, inverted through the calibration curve
rho(d) = rho0 * exp(-d^2 / (2 sigma^2)):  d = sigma * sqrt(-2 ln(rho / rho0)).
``sigma`` (m) and ``rho0`` (zero-motion correlation, < 1 because of electronic noise) are rig
calibrations (configs/rig.yaml: calibration.*, null until measured); on synthetic data the
generator's ground truth is used. Decorrelation gives |d| only: the sign is unobservable, and
B5 reports motion along +tool x. In-plane motion: integer-pixel phase correlation; converted to
metres with ``pixel_spacing_m`` (probe-to-image calibration, null until measured) when given.
The chunk is a constant-velocity extrapolation of the last frame-pair motion.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from usintent.policies.base import Policy


def patch_ncc(a: NDArray[np.float64], b: NDArray[np.float64], patch: int,
              min_std: float) -> NDArray[np.float64]:
    """NCC of co-located square patches of two frames; patches with std < min_std are skipped."""
    rows, cols = a.shape
    out = []
    for r in range(0, rows - patch + 1, patch):
        for c in range(0, cols - patch + 1, patch):
            pa = a[r:r + patch, c:c + patch].ravel()
            pb = b[r:r + patch, c:c + patch].ravel()
            sa, sb = pa.std(), pb.std()
            if sa < min_std or sb < min_std:
                continue
            out.append(float(np.mean((pa - pa.mean()) * (pb - pb.mean())) / (sa * sb)))
    return np.asarray(out)


def elevational_displacement(a: NDArray[np.uint8], b: NDArray[np.uint8], sigma_m: float,
                             rho0: float, patch: int = 16, min_std: float = 8.0) -> float:
    """|d| (m) between two frames from the median patch NCC."""
    rho = patch_ncc(a.astype(np.float64), b.astype(np.float64), patch, min_std)
    if rho.size == 0:
        return float("nan")
    ratio = float(np.clip(np.median(rho) / rho0, 1e-6, 1.0))
    return float(sigma_m * np.sqrt(-2.0 * np.log(ratio)))


def inplane_shift_px(a: NDArray[np.uint8], b: NDArray[np.uint8]) -> tuple[int, int]:
    """Integer (row, col) shift of b relative to a by phase correlation."""
    fa = np.fft.fft2(a.astype(np.float64) - a.mean())
    fb = np.fft.fft2(b.astype(np.float64) - b.mean())
    cross = fb * np.conj(fa)
    corr = np.fft.ifft2(cross / np.maximum(np.abs(cross), 1e-12)).real
    r, c = np.unravel_index(int(np.argmax(corr)), corr.shape)
    rows, cols = a.shape
    return (int(r - rows) if r > rows // 2 else int(r), int(c - cols) if c > cols // 2 else int(c))


class SpeckleDecorrelation(Policy):
    arm_id = "B5"

    def __init__(self, n_steps: int, sigma_m: float, rho0: float,
                 pixel_spacing_m: float | None = None, moving_thr_mps: float = 0.0):
        self.n_steps = n_steps
        self.sigma_m = sigma_m
        self.rho0 = rho0
        self.pixel_spacing_m = pixel_spacing_m
        self.moving_thr_mps = moving_thr_mps

    def velocity(self, history: Any) -> NDArray[np.float64]:
        """(vx, vy) tool-frame velocity estimate (m/s) from the last two frames."""
        frames, times = history["bmode"], history["time"]
        if len(frames) < 2 or times[-1] <= times[-2]:
            return np.zeros(2)
        dt = float(times[-1] - times[-2])
        d = elevational_displacement(frames[-2], frames[-1], self.sigma_m, self.rho0)
        vx = 0.0 if np.isnan(d) else d / dt
        vy = 0.0
        if self.pixel_spacing_m is not None:
            vy = inplane_shift_px(frames[-2], frames[-1])[1] * self.pixel_spacing_m / dt
        return np.array([vx, vy])

    def _predict(self, history: Any, mode: str | None) -> tuple[NDArray[np.float64], str]:
        v = self.velocity(history)
        dt = float(history["dt"][0])
        chunk = np.zeros((self.n_steps, 5))
        chunk[:, :2] = (np.arange(1, self.n_steps + 1) * dt)[:, None] * v
        return chunk, "SWEEP" if np.linalg.norm(v) > self.moving_thr_mps else "HOLD"
