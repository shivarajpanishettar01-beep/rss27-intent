"""Ground-truth passthrough (SPEC section 4, CLAUDE.md rule 4). SYNTHETIC data."""
from dataclasses import replace

import numpy as np
import pytest

from usintent.data.chunks import ChunkSampler
from usintent.data.loaders import read_lerobot, write_lerobot
from usintent.data.passthrough import Normalizer, PassthroughError, passthrough_check
from usintent.sim.holdworld import HoldWorldConfig, generate

CFG = HoldWorldConfig(render_images=False, tool_yaw_rad=0.6, seed=4)


def test_passthrough_zero_error_in_memory() -> None:
    eps = generate(CFG, 4)
    out = passthrough_check(ChunkSampler(eps, 1.0))
    assert out["max_trans_err_m"] < 1e-9 and out["max_rot_err_rad"] < 1e-9


def test_passthrough_after_lerobot_round_trip(tmp_path) -> None:
    eps = generate(replace(CFG, render_images=True, image_shape=(8, 8), n_stations=(2, 3)), 2)
    back = read_lerobot(write_lerobot(eps, tmp_path / "ds"))
    out = passthrough_check(ChunkSampler(back, 1.0))
    assert out["max_trans_err_m"] < 1e-6


def test_passthrough_catches_off_by_one() -> None:
    ep = generate(CFG, 1)[0]
    ep.action = np.roll(ep.action, 1, axis=0)
    with pytest.raises(PassthroughError):
        passthrough_check(ChunkSampler([ep], 1.0))


def test_passthrough_catches_wrong_frame() -> None:
    ep = generate(CFG, 1)[0]
    ep.state[:, 5] += 0.3          # orientation logged in a different frame than the actions
    with pytest.raises(PassthroughError):
        passthrough_check(ChunkSampler([ep], 1.0))


def test_passthrough_catches_normalizer_mismatch() -> None:
    eps = generate(CFG, 2)
    s = ChunkSampler(eps, 1.0)

    class Broken(Normalizer):
        def decode(self, z):
            return z * self.std          # forgets the mean

    norm = Normalizer(np.full(5, 0.01), np.ones(5))
    with pytest.raises(PassthroughError):
        passthrough_check(s, Broken(norm.mean, norm.std))
