"""Pose/action geometry and the chunk sampler with per-chunk metadata. SPEC section 4.

Conventions (docs/INTERFACES.md, usintent.data.loaders):
- pose = [x, y, z (m, base frame), rx, ry, rz (rotation vector of R_base_tool, rad)].
- step action a_k = [R_k^T (p_{k+1} - p_k), rotvec(R_k^T R_{k+1})] (tool frame at step k).
- chunk at step i with n steps: for j = 1..n, the pose of step i+j relative to pose i, expressed
  in the tool frame at i: translation R_i^T (p_{i+j} - p_i) and rotation rotvec(R_i^T R_{i+j}).
  ``chunk6`` keeps all 6 components; the 5-D policy target drops dz: [x, y, rx, ry, rz].
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from usintent.data.loaders import Episode
from usintent.data.segmentation import Segmentation, segment_cued

POLICY_DIMS = (0, 1, 3, 4, 5)   # x, y, rx, ry, rz of a 6-D displacement


def actions_from_poses(pose: NDArray[np.float64]) -> NDArray[np.float64]:
    """(N, 6) poses (base) -> (N, 6) per-step tool-frame actions; last row is zero."""
    rot = Rotation.from_rotvec(pose[:, 3:6])
    out = np.zeros_like(pose)
    if len(pose) < 2:
        return out
    r0, r1 = rot[:-1], rot[1:]
    out[:-1, :3] = r0.inv().apply(pose[1:, :3] - pose[:-1, :3])
    out[:-1, 3:] = (r0.inv() * r1).as_rotvec()
    return out


def compose_actions(actions: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compose n step actions (each in its own start frame) into cumulative 6-D displacements.

    Returns (n, 6): row j is the pose after j+1 steps relative to the start, in the start frame.
    """
    n = len(actions)
    out = np.zeros((n, 6))
    rot = Rotation.identity()
    pos = np.zeros(3)
    for j in range(n):
        pos = pos + rot.apply(actions[j, :3])
        rot = rot * Rotation.from_rotvec(actions[j, 3:])
        out[j, :3] = pos
        out[j, 3:] = rot.as_rotvec()
    return out


def chunk_from_poses(pose: NDArray[np.float64], i: int, n: int) -> NDArray[np.float64]:
    """Reference chunk (n, 6) at step i computed directly from recorded poses."""
    ri = Rotation.from_rotvec(pose[i, 3:6])
    seg = pose[i + 1 : i + 1 + n]
    out = np.zeros((n, 6))
    out[:, :3] = ri.inv().apply(seg[:, :3] - pose[i, :3])
    out[:, 3:] = (ri.inv() * Rotation.from_rotvec(seg[:, 3:6])).as_rotvec()
    return out


def integrate_actions(pose0: NDArray[np.float64], actions: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rebuild (N, 6) base-frame poses from the first pose and N per-step tool-frame actions.

    Pose k+1 = (p_k + R_k t_k, R_k exp(w_k)). Row 0 equals ``pose0``; the last action is unused.
    """
    n = len(actions)
    out = np.zeros((n, 6))
    out[0] = pose0
    if n < 2:
        return out
    if np.all(actions[:, 3:] == 0.0):   # pure translation: closed form (common in synthetic data)
        r0 = Rotation.from_rotvec(pose0[3:6])
        out[1:, :3] = pose0[:3] + r0.apply(np.cumsum(actions[:-1, :3], axis=0))
        out[1:, 3:] = pose0[3:6]
        return out
    rot = Rotation.from_rotvec(pose0[3:6])
    pos = np.asarray(pose0[:3], dtype=np.float64).copy()
    for k in range(n - 1):
        pos = pos + rot.apply(actions[k, :3])
        rot = rot * Rotation.from_rotvec(actions[k, 3:])
        out[k + 1, :3] = pos
        out[k + 1, 3:] = rot.as_rotvec()
    return out


def chunks_from_poses(pose: NDArray[np.float64], starts: NDArray[np.int64], n: int) -> NDArray[np.float64]:
    """Vectorized :func:`chunk_from_poses` for many start steps: returns (len(starts), n, 6)."""
    starts = np.asarray(starts, dtype=np.int64)
    if np.any(starts + n >= len(pose)) or np.any(starts < 0):
        raise IndexError("chunk exceeds episode")
    idx = starts[:, None] + np.arange(1, n + 1)[None, :]            # (m, n)
    if np.all(pose[:, 3:6] == pose[0, 3:6]):     # constant orientation: fast exact path
        rmat = Rotation.from_rotvec(pose[0, 3:6]).as_matrix()
        out_c = np.zeros((len(starts), n, 6))
        out_c[:, :, :3] = (pose[idx, :3] - pose[starts, None, :3]) @ rmat
        return out_c
    ri = Rotation.from_rotvec(np.repeat(pose[starts, 3:6], n, axis=0))
    rj = Rotation.from_rotvec(pose[idx.ravel(), 3:6])
    dp = pose[idx.ravel(), :3] - np.repeat(pose[starts, :3], n, axis=0)
    out = np.empty((len(starts) * n, 6))
    out[:, :3] = ri.inv().apply(dp)
    out[:, 3:] = (ri.inv() * rj).as_rotvec()
    return out.reshape(len(starts), n, 6)


# ---------------------------------------------------------------------------------------------
# Chunk sampler (SPEC section 4): chunk starts uniform in time, with per-chunk metadata.
# ---------------------------------------------------------------------------------------------



def policy_target(chunk6: NDArray[np.float64]) -> NDArray[np.float64]:
    """(..., n, 6) composed displacement -> (..., n, 5) policy target [x, y, rx, ry, rz]."""
    return np.asarray(chunk6[..., list(POLICY_DIMS)], dtype=np.float64)


def _exit_type_from_motion(ep: Episode, end: int, n: int) -> str:
    """Exit type of a segmented hold without labels: LIFT if probe-axis motion dominates."""
    j = min(end + n, len(ep) - 1)
    ch = chunk_from_poses(ep.state[:, :6], max(end - 1, 0), max(j - end + 1, 1))
    tang = float(np.linalg.norm(ch[-1, :2]))
    axial = float(abs(ch[-1, 2]))
    return "LIFT" if axial > tang else "SWEEP"


@dataclass
class ChunkSampler:
    """All valid chunk starts of a set of episodes with metadata; sampling uniform in time.

    ``segmentations`` maps episode_index -> Segmentation; default: cued segmentation from logs.
    Chunks are built from the dataset actions (integrated once per episode), i.e. through the
    same path the training code uses, and expressed in the tool frame at the chunk start.
    """

    episodes: list[Episode]
    horizon_s: float
    history_k: int = 1
    segmentations: dict[int, Segmentation] | None = None
    transition_window_s: float = 0.5

    def __post_init__(self) -> None:
        self._eps = {e.episode_index: e for e in self.episodes}
        if len(self._eps) != len(self.episodes):
            raise ValueError("duplicate episode_index")
        if self.segmentations is None:
            self.segmentations = {e.episode_index: segment_cued(e, self.transition_window_s)
                                  for e in self.episodes}
        self._poses = {e.episode_index: integrate_actions(e.state[0, :6], e.action)
                       for e in self.episodes}
        self.index = self._build_index()

    def n_steps(self, ep: Episode) -> int:
        return round(self.horizon_s * ep.fps)

    def _build_index(self) -> pd.DataFrame:
        assert self.segmentations is not None
        frames = []
        for ep in self.episodes:
            n = self.n_steps(ep)
            seg = self.segmentations[ep.episode_index]
            starts = np.arange(0, len(ep) - n)
            df = pd.DataFrame({
                "episode_index": ep.episode_index, "start": starts, "t": ep.time[starts],
                "phase": seg.phase[starts], "in_transition": seg.in_transition[starts],
                "e": seg.elapsed[starts], "seg_hold": seg.hold_index[starts],
                "session_id": ep.session_id, "day": ep.day,
            })
            end_t = np.full(len(starts), np.nan)
            exit_type = np.full(len(starts), "", dtype="<U6")
            backtrack = np.zeros(len(starts), dtype=bool)
            hold_id = np.full(len(starts), -1, dtype=np.int64)
            labelled = seg.source == "cued"
            for j, (a, b) in enumerate(seg.holds):
                m = df["seg_hold"].to_numpy() == j
                if labelled:   # cued: end = key time, exit type from the sidecar
                    row = ep.holds[ep.holds["hold_id"] == ep.hold_id[a]].iloc[0]
                    end_t[m] = row["key_time"]
                    exit_type[m] = row["exit_type"]
                    backtrack[m] = bool(row["backtrack"])
                    hold_id[m] = int(row["hold_id"])
                else:          # segmented: end = first moving step
                    end_t[m] = ep.time[b] if b < len(ep) else np.nan
                    exit_type[m] = _exit_type_from_motion(ep, b, n)
            df["hold_end_t"] = end_t
            df["exit_type"] = exit_type
            df["backtrack"] = backtrack
            df["hold_id"] = hold_id
            # NaN end (censored) compares False: no departure observed in the chunk
            df["contains_departure"] = (df["phase"] == "hold") & (end_t < df["t"] + self.horizon_s)
            frames.append(df)
        return pd.concat(frames, ignore_index=True)

    def chunk6(self, episode_index: int, start: int) -> NDArray[np.float64]:
        """(n, 6) composed displacement for one chunk, built from dataset actions."""
        ep = self._eps[episode_index]
        return chunks_from_poses(self._poses[episode_index], np.array([start]), self.n_steps(ep))[0]

    def chunks6(self, rows: pd.DataFrame) -> NDArray[np.float64]:
        """Stack of (len(rows), n, 6) chunks for index rows (any order)."""
        out = []
        for ep_idx, grp in rows.groupby("episode_index", sort=False):
            ep = self._eps[int(ep_idx)]
            out.append((grp.index.to_numpy(), chunks_from_poses(
                self._poses[int(ep_idx)], grp["start"].to_numpy(), self.n_steps(ep))))
        order = np.concatenate([o for o, _ in out])
        stacked = np.concatenate([c for _, c in out])
        pos = {k: i for i, k in enumerate(order)}
        return stacked[[pos[k] for k in rows.index]]

    def history(self, episode_index: int, start: int) -> dict[str, NDArray[np.float64]]:
        """Observation history of ``history_k`` steps ending at ``start`` (clamped at 0).

        Never contains the scheduler mode: intent reaches a policy only through the explicit
        ``mode`` argument of ``Policy.predict`` (intent gating, policies/base.py).
        """
        ep = self._eps[episode_index]
        idx = np.clip(np.arange(start - self.history_k + 1, start + 1), 0, None)
        out: dict[str, NDArray[np.float64]] = {
            "state": ep.state[idx], "wrench": ep.wrench[idx], "time": ep.time[idx],
            "dt": np.array([ep.dt]),
        }
        if ep.bmode is not None:
            out["bmode"] = ep.bmode[idx]
        return out

    def sample(self, n: int, rng: np.random.Generator,
               weights: NDArray[np.float64] | None = None) -> pd.DataFrame:
        """Draw n chunk-start rows; uniform in time unless ``weights`` (len(index)) is given."""
        p = None if weights is None else weights / weights.sum()
        pick = rng.choice(len(self.index), size=n, replace=True, p=p)
        return self.index.iloc[pick]
