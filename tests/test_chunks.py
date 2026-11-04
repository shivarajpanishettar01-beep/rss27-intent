"""Chunk sampler and geometry (SPEC section 4). SYNTHETIC data."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from usintent.data.chunks import (
    ChunkSampler,
    actions_from_poses,
    chunk_from_poses,
    chunks_from_poses,
    compose_actions,
    integrate_actions,
    policy_target,
)
from usintent.sim.holdworld import HoldWorldConfig, generate
from usintent.theory import Uniform, pibar


def _random_poses(n=80, seed=0):
    rng = np.random.default_rng(seed)
    pose = np.zeros((n, 6))
    pose[:, :3] = np.cumsum(rng.normal(0, 1e-3, (n, 3)), axis=0)
    rots = Rotation.from_rotvec(np.cumsum(rng.normal(0, 0.02, (n, 3)), axis=0))
    pose[:, 3:] = rots.as_rotvec()
    return pose


def test_compose_and_integrate_are_exact_with_rotation() -> None:
    pose = _random_poses()
    act = actions_from_poses(pose)
    assert np.max(np.abs(integrate_actions(pose[0], act) - pose)[:, :3]) < 1e-12
    ref = chunk_from_poses(pose, 10, 20)
    comp = compose_actions(act[10:30])
    assert np.max(np.abs(comp - ref)) < 1e-10
    vec = chunks_from_poses(pose, np.array([10, 33]), 20)
    assert np.max(np.abs(vec[0] - ref)) < 1e-12
    assert np.max(np.abs(vec[1] - chunk_from_poses(pose, 33, 20))) < 1e-12


def test_known_translation_chunk() -> None:
    pose = np.zeros((10, 6))
    pose[:, 0] = np.arange(10) * 0.001
    pose[:, 3:] = [0, 0, np.pi / 2]           # tool x = base y, tool y = -base x
    ch = chunks_from_poses(pose, np.array([0]), 3)[0]
    assert np.allclose(ch[:, 0], 0) and np.allclose(ch[:, 1], [-0.001, -0.002, -0.003])
    assert policy_target(ch).shape == (3, 5)


def test_chunk_starts_uniform_in_time_and_metadata() -> None:
    eps = generate(HoldWorldConfig(render_images=False, seed=1), 300)
    s = ChunkSampler(eps, horizon_s=1.0)
    idx = s.index
    hold = idx[idx["phase"] == "hold"]
    assert hold["contains_departure"].mean() == pytest.approx(pibar(Uniform(2, 10), 1.0), abs=0.015)
    assert np.all(hold["e"] >= 0) and idx.loc[idx["phase"] != "hold", "e"].isna().all()
    assert set(hold["exit_type"]) <= {"SWEEP", "LIFT"}
    # e equals t - hold start from the sidecar
    ep = eps[0]
    rows = hold[hold["episode_index"] == 0]
    starts = ep.holds.set_index("hold_id").loc[rows["hold_id"], "start_time"].to_numpy()
    assert np.allclose(rows["e"].to_numpy(), rows["t"].to_numpy() - starts, atol=ep.dt)
    # sampling without weights is uniform over chunk starts (= uniform in time at fixed fps)
    draw = s.sample(200_000, np.random.default_rng(0))
    frac = (draw["phase"] == "hold").mean()
    assert frac == pytest.approx((idx["phase"] == "hold").mean(), abs=0.005)


def test_history_is_clamped_and_causal() -> None:
    eps = generate(HoldWorldConfig(image_shape=(8, 8), seed=2), 1)
    s = ChunkSampler(eps, horizon_s=1.0, history_k=3)
    h = s.history(0, 0)
    assert h["bmode"].shape == (3, 8, 8) and np.all(h["time"] == 0.0)
    h = s.history(0, 10)
    assert np.allclose(h["time"], eps[0].time[8:11])
