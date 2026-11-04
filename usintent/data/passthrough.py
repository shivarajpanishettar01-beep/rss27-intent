"""Ground-truth passthrough check (must be zero error before any scoring). SPEC 4, CLAUDE rule 4.

Path exercised (identical to the one used for policy outputs):
  dataset actions --compose--> 6-D chunk --project--> 5-D target --normalize--> network space
  --denormalize--> 5-D chunk --tool->base--> commanded waypoints (base frame)
The reference is built only from recorded *poses* (observation.state), independently of the
actions: base displacement projected onto the chunk-start tool x/y plane, and the recorded
future orientation. Any indexing, frame or normalization bug gives a non-zero error.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from usintent.data.chunks import ChunkSampler, policy_target

PASSTHROUGH_TOL = 1e-6


class PassthroughError(AssertionError):
    """Ground truth did not survive the decode/transform path."""


@dataclass(frozen=True)
class Normalizer:
    """Per-dimension affine normalization of 5-D chunks: z = (x - mean) / std."""

    mean: NDArray[np.float64]   # (5,)
    std: NDArray[np.float64]    # (5,)

    @classmethod
    def fit(cls, chunks5: NDArray[np.float64], min_std: float = 1e-8) -> Normalizer:
        flat = chunks5.reshape(-1, chunks5.shape[-1])
        return cls(flat.mean(axis=0), np.maximum(flat.std(axis=0), min_std))

    def encode(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray((x - self.mean) / self.std, dtype=np.float64)

    def decode(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(z * self.std + self.mean, dtype=np.float64)


def chunk5_to_base(chunk5: NDArray[np.float64], pose_start: NDArray[np.float64]
                   ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Map a 5-D tool-frame chunk to base-frame commands.

    Returns (dp_base (n, 3) m: displacement within the start tool x/y plane, rotvec_base (n, 3)
    rad: absolute target orientation R_start * exp(r)).
    """
    r0 = Rotation.from_rotvec(pose_start[3:6])
    xy = np.zeros((len(chunk5), 3))
    xy[:, :2] = chunk5[:, :2]
    dp = r0.apply(xy)
    rot = (r0 * Rotation.from_rotvec(chunk5[:, 2:5])).as_rotvec()
    return dp, rot


def reference_from_poses(pose: NDArray[np.float64], start: int, n: int
                         ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Reference built from recorded poses only (same outputs as :func:`chunk5_to_base`)."""
    r0 = Rotation.from_rotvec(pose[start, 3:6])
    m = r0.as_matrix()
    plane = m[:, :2] @ m[:, :2].T          # projector onto tool x/y plane (base coords)
    dp = (pose[start + 1:start + 1 + n, :3] - pose[start, :3]) @ plane.T
    return dp, pose[start + 1:start + 1 + n, 3:6]


def _rot_err(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    return float(np.max((Rotation.from_rotvec(a).inv() * Rotation.from_rotvec(b)).magnitude()))


def passthrough_check(sampler: ChunkSampler, normalizer: Normalizer | None = None,
                      max_chunks: int | None = None, seed: int = 0,
                      tol: float = PASSTHROUGH_TOL) -> dict[str, float]:
    """Run recorded actions through the full path; raise PassthroughError if error > tol.

    Returns {"max_trans_err_m", "max_rot_err_rad", "n_chunks"}.
    """
    rows = sampler.index
    if max_chunks is not None and len(rows) > max_chunks:
        rows = rows.iloc[np.sort(np.random.default_rng(seed).choice(len(rows), max_chunks, False))]
    ch5 = policy_target(sampler.chunks6(rows))
    norm = normalizer or Normalizer.fit(ch5)
    decoded = norm.decode(norm.encode(ch5))
    eps = {e.episode_index: e for e in sampler.episodes}
    t_err = r_err = 0.0
    for k, (_, row) in enumerate(rows.iterrows()):
        ep = eps[int(row["episode_index"])]
        start, n = int(row["start"]), sampler.n_steps(ep)
        dp, rot = chunk5_to_base(decoded[k], ep.state[start, :6])
        dp_ref, rot_ref = reference_from_poses(ep.state[:, :6], start, n)
        t_err = max(t_err, float(np.max(np.abs(dp - dp_ref))))
        r_err = max(r_err, _rot_err(rot, rot_ref))
    out = {"max_trans_err_m": t_err, "max_rot_err_rad": r_err, "n_chunks": float(len(rows))}
    if t_err > tol or r_err > tol:
        raise PassthroughError(f"passthrough failed: {out}")
    return out
