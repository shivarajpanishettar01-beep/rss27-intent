"""Confidence maps: Karamalis random-walk + fast proxy, with runtime report. SPEC section 9. M7.

``karamalis_confidence`` follows Karamalis et al., "Ultrasound confidence maps using random
walks", Medical Image Analysis 2012, as I understand the method (I could not consult a
reference implementation from this session; it is checked here only for its qualitative
properties): a random walk on the pixel graph of a (downsampled) B-mode frame, with the
transducer row (top) as source label 1 and the bottom row as sink label 0. Edge weights penalise
intensity jumps of the attenuation-weighted image c = I * exp(-alpha * depth):
  vertical edges   w = exp(-beta * |c_i - c_j|)
  horizontal and diagonal edges   w = exp(-beta * (|c_i - c_j| + gamma))
The confidence is the probability that a walker from a pixel reaches the transducer row first:
the solution of the Dirichlet problem L_U x_U = -B x_M (sparse direct solve).
Defaults alpha = 2.0, beta = 90, gamma = 0.05 (intensities in [0, 1], depth in [0, 1]) are
algorithm parameters, not rig facts.

``proxy_confidence``: depth-attenuation-compensated intensity under the transducer, as a ratio to
the near-field mean (SPEC section 9, fallback if Karamalis is too slow). It is a ratio (about 1
for tissue whose attenuation is fully compensated), not a probability; it is not clipped because
clipping saturated it in bright tissue. Agreement with Karamalis is reported as a correlation.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.sparse.linalg import spsolve


def _downsample(frame: NDArray[np.uint8], shape: tuple[int, int] | None) -> NDArray[np.float64]:
    f = frame.astype(np.float64) / 255.0
    if shape is None:
        return f
    r, c = min(shape[0], f.shape[0]), min(shape[1], f.shape[1])    # never upsample
    if (r, c) == f.shape:
        return f
    ri = (np.arange(r) * f.shape[0] / r).astype(int)
    ci = (np.arange(c) * f.shape[1] / c).astype(int)
    rb = np.add.reduceat(f, ri, axis=0) / np.diff(np.append(ri, f.shape[0]))[:, None]
    return np.asarray(np.add.reduceat(rb, ci, axis=1) / np.diff(np.append(ci, f.shape[1]))[None, :])


def karamalis_confidence(frame: NDArray[np.uint8], alpha: float = 2.0, beta: float = 90.0,
                         gamma: float = 0.05, shape: tuple[int, int] | None = (32, 32)
                         ) -> NDArray[np.float64]:
    """Random-walk confidence map in [0, 1] at ``shape`` resolution (rows = depth)."""
    img = _downsample(frame, shape)
    rows, cols = img.shape
    depth = (np.arange(rows) / max(rows - 1, 1))[:, None]
    c = img * np.exp(-alpha * depth)
    idx = np.arange(rows * cols).reshape(rows, cols)
    ii, jj, ww = [], [], []

    def add(a: NDArray[np.int64], b: NDArray[np.int64], w: NDArray[np.float64]) -> None:
        ii.extend([a.ravel(), b.ravel()])
        jj.extend([b.ravel(), a.ravel()])
        ww.extend([w.ravel(), w.ravel()])

    add(idx[:-1, :], idx[1:, :], np.exp(-beta * np.abs(c[:-1, :] - c[1:, :])))            # vertical
    add(idx[:, :-1], idx[:, 1:], np.exp(-beta * (np.abs(c[:, :-1] - c[:, 1:]) + gamma)))  # horizontal
    add(idx[:-1, :-1], idx[1:, 1:], np.exp(-beta * (np.abs(c[:-1, :-1] - c[1:, 1:]) + gamma)))
    add(idx[:-1, 1:], idx[1:, :-1], np.exp(-beta * (np.abs(c[:-1, 1:] - c[1:, :-1]) + gamma)))
    w = sp.csr_matrix((np.concatenate(ww) + 1e-10, (np.concatenate(ii), np.concatenate(jj))),
                      shape=(rows * cols, rows * cols))
    lap = sp.diags(np.asarray(w.sum(axis=1)).ravel()) - w
    seeds = np.concatenate([idx[0], idx[-1]])
    vals = np.concatenate([np.ones(cols), np.zeros(cols)])
    free = np.setdiff1d(np.arange(rows * cols), seeds)
    lap = lap.tocsr()
    x = np.zeros(rows * cols)
    x[seeds] = vals
    rhs = -lap[free][:, seeds] @ vals
    x[free] = spsolve(lap[free][:, free].tocsc(), rhs)
    return np.clip(x.reshape(rows, cols), 0.0, 1.0)


def proxy_confidence(frame: NDArray[np.uint8], alpha: float = 1.0,
                     near_field_frac: float = 0.15) -> NDArray[np.float64]:
    """Attenuation-compensated intensity ratio I * exp(alpha * depth) / near-field mean."""
    f = frame.astype(np.float64)
    rows = f.shape[0]
    depth = (np.arange(rows) / max(rows - 1, 1))[:, None]
    near = f[: max(1, round(rows * near_field_frac))].mean()
    if near <= 0:
        return np.zeros_like(f)
    return np.asarray(f * np.exp(alpha * depth) / near, dtype=np.float64)


def roi_mean(conf: NDArray[np.float64], roi: NDArray[np.bool_]) -> float:
    """Mean confidence in an ROI given at frame resolution (resampled to the map size)."""
    if roi.shape != conf.shape:
        ri = (np.arange(conf.shape[0]) * roi.shape[0] / conf.shape[0]).astype(int)
        ci = (np.arange(conf.shape[1]) * roi.shape[1] / conf.shape[1]).astype(int)
        roi = roi[np.ix_(ri, ci)]
    return float(conf[roi].mean()) if roi.any() else float(conf.mean())


@dataclass(frozen=True)
class RuntimeReport:
    impl: str
    shape: tuple[int, int]
    ms_per_frame_median: float
    ms_per_frame_p95: float
    n_frames: int


def measure_runtime(frames: NDArray[np.uint8], impl: str = "karamalis",
                    shape: tuple[int, int] = (32, 32)) -> RuntimeReport:
    """Wall-clock time per frame of one confidence implementation on this machine."""
    ts = []
    for fr in frames:
        t = time.perf_counter()
        if impl == "karamalis":
            karamalis_confidence(fr, shape=shape)
        else:
            proxy_confidence(fr)
        ts.append((time.perf_counter() - t) * 1e3)
    return RuntimeReport(impl, shape, float(np.median(ts)), float(np.quantile(ts, 0.95)), len(ts))
