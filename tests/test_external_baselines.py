"""External comparison families: ACT (trained in our loop) and SmolVLA / pi0.5 (lerobot-train).

ACT tests run the real LeRobot ACT on synthetic hold-world data (small model sizes). VLA tests
cover everything that needs no pretrained weights: config validation, the LeRobot v3 export,
the passthrough through LeRobot's own loader (rule 4), the decode path and the train command.
"""
import json

import numpy as np
import pytest

from usintent.config import load
from usintent.data.chunks import ChunkSampler, chunks_from_poses, policy_target
from usintent.data.passthrough import PassthroughError
from usintent.policies import act_adapter
from usintent.policies import lerobot_vla_adapter as vla
from usintent.policies.base import INTENT_AWARE_ARMS
from usintent.sim.holdworld import HoldWorldConfig, generate

SMALL = {"dim_model": 32, "n_heads": 2, "dim_feedforward": 64, "n_encoder_layers": 1,
         "n_decoder_layers": 1, "latent_dim": 8}


@pytest.fixture(scope="module")
def eps():
    return generate(HoldWorldConfig(image_shape=(32, 32), fps=10.0, n_stations=(2, 3), seed=3), 3)


@pytest.fixture(scope="module")
def sampler(eps):
    return ChunkSampler(eps, 1.0, history_k=load("data")["history_frames_k"])


# ------------------------------------------------------------------------------------- configs


def test_external_arm_configs_and_gating() -> None:
    for arm in ["ACT-1", "SmolVLA-1", "PI05-1"]:
        cfg = load(f"arms/{arm}")
        assert not cfg["intent_aware"] and "mode_embedding" not in cfg["inputs"]
        assert arm not in INTENT_AWARE_ARMS
    assert load("arms/ACT-5")["intent_aware"] and "ACT-5" in INTENT_AWARE_ARMS
    # every external arm sees the same observation set as A1 / DP-1 (+ mode for ACT-5)
    base = set(load("arms/DP-1")["inputs"])
    for arm in ["ACT-1", "SmolVLA-1", "PI05-1"]:
        assert set(load(f"arms/{arm}")["inputs"]) == base
    assert set(load("arms/ACT-5")["inputs"]) == base | {"mode_embedding"}


def test_vla_configs_need_rename_map_and_schedule() -> None:
    from usintent.config import MissingRigFact, require

    for arm in ["SmolVLA-1", "PI05-1"]:
        cfg = load(f"arms/{arm}")
        with pytest.raises(MissingRigFact):
            require(cfg, "rename_map")
        with pytest.raises(MissingRigFact):
            require(cfg, "schedule.steps")
    assert load("arms/SmolVLA-1")["checkpoint"] == "lerobot/smolvla_base"
    assert load("arms/PI05-1")["checkpoint"] == "lerobot/pi05_base"


def test_act_rejects_history_and_mismatched_mode() -> None:
    cfg = dict(load("arms/ACT-1"))
    with pytest.raises(ValueError, match="one observation step"):
        act_adapter.arm_spec_from_config({**cfg, "inputs": ["bmode", "bmode_history"]}, 10, (32, 32))
    with pytest.raises(ValueError, match="intent_aware"):
        act_adapter.arm_spec_from_config({**cfg, "inputs": ["bmode", "mode_embedding"]}, 10, (32, 32))
    with pytest.raises(ValueError, match="not an ACT arm"):
        act_adapter.arm_spec_from_config(load("arms/DP-1"), 10, (32, 32))


def test_vla_rejects_intent_inputs() -> None:
    cfg = dict(load("arms/SmolVLA-1"))
    with pytest.raises(ValueError):
        vla.arm_spec_from_config({**cfg, "intent_aware": True}, 10, False)
    with pytest.raises(ValueError):
        vla.arm_spec_from_config(load("arms/ACT-1"), 10, False)


# ----------------------------------------------------------------------------------------- ACT


@pytest.mark.parametrize("arm", ["ACT-1", "ACT-5"])
def test_act_dry_run_and_determinism(sampler, arm) -> None:
    spec = act_adapter.arm_spec_from_config(load(f"arms/{arm}"), 10, (32, 32), **SMALL)
    pol = act_adapter.ACTPolicy(spec, seed=0)
    losses = pol.train_steps(sampler, 10, 4, 1e-4, np.random.default_rng(0))
    assert len(losses) == 10 and all(np.isfinite(losses))
    h = sampler.history(0, 40)
    mode = "HOLD" if pol.intent_aware else None
    chunk, m = pol.predict(h, mode)
    assert chunk.shape == (10, 5) and m in ("HOLD", "SWEEP", "LIFT", "RECONTACT")
    again, _ = pol.predict(h, mode)
    assert np.array_equal(chunk, again)                   # zero VAE latent: deterministic


def test_act_intent_gating_and_mode_changes_output(sampler) -> None:
    blind = act_adapter.ACTPolicy(
        act_adapter.arm_spec_from_config(load("arms/ACT-1"), 10, (32, 32), **SMALL), seed=0)
    aware = act_adapter.ACTPolicy(
        act_adapter.arm_spec_from_config(load("arms/ACT-5"), 10, (32, 32), **SMALL), seed=0)
    blind.train_steps(sampler, 2, 2, 1e-4, np.random.default_rng(0))
    aware.train_steps(sampler, 2, 2, 1e-4, np.random.default_rng(0))
    h = sampler.history(0, 40)
    with pytest.raises(PermissionError):
        blind.predict(h, "HOLD")
    with pytest.raises(ValueError):
        aware.predict(h, None)
    a, _ = aware.predict(h, "HOLD")
    b, _ = aware.predict(h, "SWEEP")
    assert not np.allclose(a, b)                          # the mode token reaches the decoder


def test_act_targets_match_dp_targets(sampler) -> None:
    """ACT and DP train on the identical target (shared code path)."""
    from usintent.policies.dp_adapter import DPPolicy
    from usintent.policies.dp_adapter import arm_spec_from_config as dp_spec

    rows = sampler.index.iloc[:8]
    act = act_adapter.ACTPolicy(
        act_adapter.arm_spec_from_config(load("arms/ACT-1"), 10, (32, 32), **SMALL), seed=0)
    dp = DPPolicy(dp_spec(load("arms/DP-1"), 10, (32, 32), sampler.history_k), seed=0)
    assert np.array_equal(act.targets(sampler, rows), dp.targets(sampler, rows))


def test_act_checkpoint_roundtrip(sampler, tmp_path, monkeypatch) -> None:
    """model.pt written as run_train_arm.py does it reloads through evaluate.load_policy."""
    import torch

    from usintent.policies.evaluate import load_policy

    spec = act_adapter.arm_spec_from_config(load("arms/ACT-1"), 10, (32, 32), **SMALL)
    pol = act_adapter.ACTPolicy(spec, seed=0)
    pol.train_steps(sampler, 2, 2, 1e-4, np.random.default_rng(0))
    p = tmp_path / "model.pt"
    torch.save({"state_dict": pol.model.state_dict(), "norm_lo": pol.norm.lo,
                "norm_hi": pol.norm.hi, "spec": spec.__dict__}, p)
    back = load_policy("ACT-1", p)
    h = sampler.history(0, 40)
    assert np.allclose(back.predict(h)[0], pol.predict(h)[0])


# ----------------------------------------------------------------------------------------- VLA


def test_vla_decode_passthrough_is_exact(eps) -> None:
    out = vla.decode_passthrough(eps, 10)
    assert out["max_err"] < 1e-6 and out["mode_mismatches"] == 0 and out["n_chunks"] > 0


def test_vla_decode_passthrough_catches_off_by_one(eps) -> None:
    """A one-step shift of the action rows (the classic silent failure) must fail rule 4."""
    from dataclasses import replace

    shifted = [replace(ep, action=np.roll(ep.action, 1, axis=0)) for ep in eps]
    with pytest.raises(PassthroughError):
        vla.decode_passthrough(shifted, 10)


def test_vla_export_passthrough_through_lerobot_loader(eps, tmp_path) -> None:
    """Exported copy read back with LeRobot's loader and delta_timestamps (as lerobot-train
    does) reproduces our chunks and modes; images round-trip exactly."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    spec = vla.arm_spec_from_config(load("arms/SmolVLA-1"), 10, use_camera_topic=False)
    root = vla.export_vla_dataset(eps, tmp_path / "vla", spec)
    out = vla.vla_passthrough(root, eps, 10, max_starts=60)
    assert out["max_err"] < 1e-5 and out["mode_mismatches"] == 0 and out["padded_chunks"] == 0
    ds = LeRobotDataset("local/usintent_vla", root=root)
    item = ds[5]
    img = (item[vla.IMG_KEY].numpy() * 255).round().astype(np.uint8)          # (3, H, W)
    assert np.array_equal(img[0], eps[0].bmode[5]) and item["task"] == vla.TASK
    assert item[vla.STATE_KEY].shape == (2,) and float(item[vla.STATE_KEY].abs().max()) == 0.0
    meta = json.loads((root / "meta" / "usintent_vla.json").read_text())
    assert meta["arm_id"] == "SmolVLA-1" and meta["n_steps"] == 10


def test_vla_decode_matches_pose_chunk(eps) -> None:
    ep = eps[0]
    a = vla.action_vector(ep).astype(np.float64)
    chunk, m = vla.decode_chunk(a[40:50])
    ref = policy_target(chunks_from_poses(ep.state[:, :6], np.array([40]), 10))[0]
    assert np.allclose(chunk, ref, atol=1e-6) and m == ep.contact_cmd[41]


def test_vla_train_command(tmp_path) -> None:
    spec = vla.arm_spec_from_config(load("arms/PI05-1"), 20, use_camera_topic=True)
    cmd = vla.train_command(spec, tmp_path / "d", "lerobot/pi05_base", tmp_path / "o", 3,
                            1000, 32, 2.5e-5, {"observation.images.bmode": "observation.images.base_0_rgb"},
                            {"policy.dtype": "bfloat16"})
    assert cmd[0] == "lerobot-train"
    for flag in ("--policy.path=lerobot/pi05_base", "--seed=3", "--steps=1000", "--batch_size=32",
                 "--policy.chunk_size=20", "--policy.n_action_steps=20",
                 "--policy.push_to_hub=false", "--policy.dtype=bfloat16",
                 "--policy.scheduler_decay_steps=1000"):
        assert flag in cmd
    rm = next(c for c in cmd if c.startswith("--rename_map="))
    assert json.loads(rm.split("=", 1)[1]) == {"observation.images.bmode": "observation.images.base_0_rgb"}


def test_vla_observation_format(sampler) -> None:
    spec = vla.arm_spec_from_config(load("arms/SmolVLA-1"), 10, use_camera_topic=False)
    obs = vla.observation(spec, sampler.history(0, 40))
    assert obs[vla.IMG_KEY].shape == (32, 32, 3) and obs[vla.IMG_KEY].dtype == np.uint8
    assert vla.CAM_KEY not in obs and obs[vla.STATE_KEY].shape == (2,)


def test_vla_policy_needs_a_checkpoint(tmp_path) -> None:
    spec = vla.arm_spec_from_config(load("arms/SmolVLA-1"), 10, use_camera_topic=False)
    with pytest.raises(FileNotFoundError):
        vla.LeRobotVLAPolicy(spec, tmp_path / "missing", {}, device="cpu")
    with pytest.raises(FileNotFoundError):
        vla.checkpoint_dir(tmp_path)
