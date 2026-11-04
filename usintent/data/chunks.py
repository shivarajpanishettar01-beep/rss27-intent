"""Pose/action geometry and the chunk sampler with per-chunk metadata. SPEC section 4.

Conventions (docs/INTERFACES.md, usintent.data.loaders):
- pose = [x, y, z (m, base frame), rx, ry, rz (rotation vector of R_base_tool, rad)].
- step action a_k = [R_k^T (p_{k+1} - p_k), rotvec(R_k^T R_{k+1})] (tool frame at step k).
- chunk at step i with n steps: for j = 1..n, the pose of step i+j relative to pose i, expressed
  in the tool frame at i: translation R_i^T (p_{i+j} - p_i) and rotation rotvec(R_i^T R_{i+j}).
  ``chunk6`` keeps all 6 components; the 5-D policy target drops dz: [x, y, rx, ry, rz].
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

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
