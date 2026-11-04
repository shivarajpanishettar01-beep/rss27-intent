"""B3 IMM filter (HOLD, CV, LIFT) + detection lag. SPEC section 6. M3.

Interacting-multiple-model filter on TCP position, expressed in the tool frame of the pose at
filter reset (orientation assumed slowly varying). State x = [px, py, pz, vx, vy, vz] (m, m/s);
measurement = position. Modes:
  HOLD: v = 0;  CV: constant tangential velocity (vx, vy), vz = 0;  LIFT: constant velocity
  along the probe axis (vz), vx = vy = 0.
Markov transition matrix ``trans[i, j] = P(mode j at k+1 | mode i at k)`` per step: a
constructor argument; :func:`estimate_transition_matrix` fits it from training-split labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from usintent.data.loaders import Episode
from usintent.policies.base import Policy

IMM_MODES = ("HOLD", "CV", "LIFT")
_LABEL_TO_MODE = {"HOLD": 0, "SWEEP": 1, "LIFT": 2, "RECONTACT": 2}


def estimate_transition_matrix(episodes: list[Episode], prior_count: float = 1.0
                               ) -> NDArray[np.float64]:
    """Per-step transition matrix from operator mode labels (training split only)."""
    counts = np.full((3, 3), prior_count)
    for ep in episodes:
        m = np.array([_LABEL_TO_MODE[c] for c in ep.contact_cmd])
        np.add.at(counts, (m[:-1], m[1:]), 1.0)
    return np.asarray(counts / counts.sum(axis=1, keepdims=True))


@dataclass(frozen=True)
class IMMParams:
    trans: NDArray[np.float64]      # (3, 3) per-step Markov matrix
    dt: float                       # s
    meas_std_m: float = 1e-4        # position measurement noise (m)
    hold_pos_std_m: float = 1e-5    # per-step position diffusion in HOLD (m)
    vel_std_mps: float = 2e-3       # per-step velocity diffusion in CV / LIFT (m/s)


def _models(p: IMMParams) -> list[tuple[NDArray[np.float64], NDArray[np.float64]]]:
    dt = p.dt
    eye3 = np.eye(3)
    out = []
    for active in ([], [0, 1], [2]):      # velocity components each mode keeps
        f = np.zeros((6, 6))
        f[:3, :3] = eye3
        q = np.zeros((6, 6))
        q[:3, :3] = eye3 * p.hold_pos_std_m**2
        for a in active:
            f[a, 3 + a] = dt
            f[3 + a, 3 + a] = 1.0
            q[3 + a, 3 + a] = p.vel_std_mps**2
            q[a, a] += (p.vel_std_mps * dt) ** 2
        out.append((f, q))
    return out


class IMMFilter:
    """Stateful IMM; call :meth:`update` with each new position (tool frame of reset pose)."""

    def __init__(self, p: IMMParams):
        self.p = p
        self.models = _models(p)
        self.h = np.hstack([np.eye(3), np.zeros((3, 3))])
        self.r = np.eye(3) * p.meas_std_m**2
        self.reset(np.zeros(3))

    def reset(self, pos: NDArray[np.float64]) -> None:
        x = np.concatenate([pos, np.zeros(3)])
        p0 = np.diag([self.p.meas_std_m**2] * 3 + [1e-4] * 3)
        self.x = [x.copy() for _ in range(3)]
        self.cov = [p0.copy() for _ in range(3)]
        self.mu = np.array([0.8, 0.1, 0.1])

    def update(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        trans = self.p.trans
        c = trans.T @ self.mu                                   # predicted mode probabilities
        mix = (trans * self.mu[:, None]) / np.maximum(c[None, :], 1e-300)   # mix[i, j]
        new_x, new_p, lik = [], [], np.zeros(3)
        for j, (f, q) in enumerate(self.models):
            x0 = sum(mix[i, j] * self.x[i] for i in range(3))
            p0 = sum(mix[i, j] * (self.cov[i] + np.outer(self.x[i] - x0, self.x[i] - x0))
                     for i in range(3))
            xp = f @ x0
            pp = f @ p0 @ f.T + q
            s = self.h @ pp @ self.h.T + self.r
            innov = z - self.h @ xp
            k = pp @ self.h.T @ np.linalg.inv(s)
            new_x.append(xp + k @ innov)
            new_p.append((np.eye(6) - k @ self.h) @ pp)
            _, logdet = np.linalg.slogdet(2 * np.pi * s)
            lik[j] = -0.5 * (innov @ np.linalg.solve(s, innov) + logdet)
        logw = lik + np.log(np.maximum(c, 1e-300))
        w = np.exp(logw - logw.max())
        self.mu = w / w.sum()
        self.x, self.cov = new_x, new_p
        return self.mu.copy()

    def velocity(self) -> NDArray[np.float64]:
        best = int(np.argmax(self.mu))
        return np.asarray(self.x[best][3:], dtype=np.float64)


class IMMBaseline(Policy):
    """B3 as a policy: chunk from the most probable mode, extrapolated at constant velocity."""

    arm_id = "B3"

    def __init__(self, p: IMMParams, n_steps: int):
        self.p = p
        self.n_steps = n_steps
        self.filter = IMMFilter(p)
        self._last_t = np.inf
        self._r0: Rotation | None = None

    def _to_frame(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        assert self._r0 is not None
        return np.asarray(self._r0.inv().apply(state[:3]), dtype=np.float64)

    def _predict(self, history: Any, mode: str | None) -> tuple[NDArray[np.float64], str]:
        times, states = history["time"], history["state"]
        if times[-1] < self._last_t - 1e-12 or self._r0 is None:   # new episode or time reset
            self._r0 = Rotation.from_rotvec(states[-1, 3:6])
            self.filter.reset(self._to_frame(states[-1]))
            self._last_t = float(times[-1])
        for t, s in zip(times, states):
            if t > self._last_t + 1e-12:
                self.filter.update(self._to_frame(s))
                self._last_t = float(t)
        best = int(np.argmax(self.filter.mu))
        chunk = np.zeros((self.n_steps, 5))
        if best == 1:
            v = self.filter.velocity()
            steps = (np.arange(1, self.n_steps + 1) * self.p.dt)[:, None]
            chunk[:, :2] = steps * v[:2]
        return chunk, ("HOLD", "SWEEP", "LIFT")[best]


def mode_probabilities(ep: Episode, p: IMMParams) -> NDArray[np.float64]:
    """(N, 3) IMM mode probabilities over one episode (filter reset at step 0)."""
    r0 = Rotation.from_rotvec(ep.state[0, 3:6])
    f = IMMFilter(p)
    f.reset(r0.inv().apply(ep.state[0, :3]))
    out = np.zeros((len(ep), 3))
    out[0] = f.mu
    for k in range(1, len(ep)):
        out[k] = f.update(r0.inv().apply(ep.state[k, :3]))
    return out


def detection_lags(episodes: list[Episode], p: IMMParams) -> pd.DataFrame:
    """Per hold: time (s) from key press until P(new mode) > 0.5 (new = CV for SWEEP exits,
    LIFT for LIFT exits) and until P(HOLD) < 0.5 (any departure). NaN if never detected."""
    rows = []
    for ep in episodes:
        mu = mode_probabilities(ep, p)
        starts = np.sort(ep.holds["start_time"].to_numpy())
        for _, h in ep.holds.iterrows():
            if h["censored"]:
                continue
            nxt = starts[starts > h["start_time"]]
            window_end = nxt[0] if len(nxt) else np.inf      # stop at the next hold
            after = (ep.time >= h["key_time"]) & (ep.time < window_end)
            target = 1 if h["exit_type"] == "SWEEP" else 2
            hit_new = np.flatnonzero(after & (mu[:, target] > 0.5))
            hit_any = np.flatnonzero(after & (mu[:, 0] < 0.5))
            key = float(h["key_time"])
            rows.append({"episode_index": ep.episode_index, "hold_id": int(h["hold_id"]),
                         "exit_type": h["exit_type"],
                         "lag_new_mode_s": _lag(ep.time, hit_new, key),
                         "lag_any_departure_s": _lag(ep.time, hit_any, key)})
    return pd.DataFrame(rows)


def _lag(time: NDArray[np.float64], hit: NDArray[np.int64], key: float) -> float:
    return float(time[hit[0]] - key) if len(hit) else float("nan")
