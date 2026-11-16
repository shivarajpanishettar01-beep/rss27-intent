"""US-VLA arm (USVLA-1): action representation, rule-4 passthroughs, export, client.

Always run: config checks, absolute-pose actions and decoding (incl. rotations), passthrough
failure on an off-by-one, export, commands, client logic with a fake connection.
Run when the US-VLA pieces are present (skipped otherwise, e.g. in CI):
- ``USINTENT_USVLA_LEROBOT`` = a site dir holding the LeRobot commit US-VLA pins (0cf8648):
  ``scripts/usvla_run.py build-dataset`` builds the LeRobot v2.1 copy and reads it back as
  openpi does.
- ``USINTENT_USVLA_ROOT`` = a US-VLA checkout and ``openpi_client`` importable: our client talks
  to US-VLA's real WebsocketPolicyServer wrapping ``SeededPolicy`` around a stand-in policy.
"""
import json
import os
import runpy
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from usintent.config import load
from usintent.data.chunks import chunks_from_poses, policy_target
from usintent.data.passthrough import PassthroughError
from usintent.policies import usvla_adapter as uv
from usintent.policies.base import INTENT_AWARE_ARMS
from usintent.sim.holdworld import HoldWorldConfig, generate

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "usvla_run.py"


@pytest.fixture(scope="module")
def eps():
    return generate(HoldWorldConfig(image_shape=(32, 32), fps=10.0, n_stations=(2, 3), seed=5), 3)


def _spec(n=10, cam=False):
    return uv.arm_spec_from_config(load("arms/USVLA-1"), n, cam)


def test_config() -> None:
    cfg = load("arms/USVLA-1")
    assert cfg["family"] == "usvla" and not cfg["intent_aware"] and "USVLA-1" not in INTENT_AWARE_ARMS
    assert set(cfg["inputs"]) == set(load("arms/DP-1")["inputs"]) | {"proprio"}
    assert cfg["train_config"] == "pi05_ur_usfm_fusion"
    with pytest.raises(ValueError, match="TCP pose"):
        uv.arm_spec_from_config({**cfg, "inputs": ["bmode", "camera"]}, 10, False)
    with pytest.raises(ValueError):
        uv.arm_spec_from_config({**cfg, "intent_aware": True}, 10, False)
    with pytest.raises(ValueError):
        uv.arm_spec_from_config(load("arms/DP-1"), 10, False)


def test_actions_are_next_pose(eps) -> None:
    ep = eps[0]
    a = uv.usvla_actions(ep)
    assert a.shape == (len(ep), 6) and a.dtype == np.float32
    assert np.allclose(a[:-1], ep.state[1:, :6]) and np.allclose(a[-1], ep.state[-1, :6])


def test_decode_with_rotation() -> None:
    """Absolute poses with changing orientation decode to the ground-truth tool-frame chunk."""
    rng = np.random.default_rng(0)
    rv = Rotation.from_euler("xyz", np.cumsum(rng.normal(0, 0.05, (30, 3)), 0)).as_rotvec()
    pose = np.hstack([np.cumsum(rng.normal(0, 1e-3, (30, 3)), 0), rv])
    chunk, m = uv.decode_chunk(pose[6:16], pose[5])
    ref = policy_target(chunks_from_poses(pose, np.array([5]), 10))[0]
    assert np.allclose(chunk, ref, atol=1e-12) and m == "SWEEP"


def test_decode_passthrough(eps) -> None:
    out = uv.decode_passthrough(eps, 10)
    assert out["max_err"] < 1e-5 and out["n_chunks"] > 0


def test_passthrough_catches_unshifted_actions(eps, monkeypatch) -> None:
    """Storing the current pose instead of the next one (off-by-one) must fail rule 4."""
    monkeypatch.setattr(uv, "usvla_actions", lambda ep: ep.state[:, :6].astype(np.float32))
    with pytest.raises(PassthroughError):
        uv.decode_passthrough(eps, 10)


def test_export(eps, tmp_path) -> None:
    out = uv.export_usvla_episodes(eps, tmp_path / "x", _spec(), "usintent/test")
    man = json.loads((out / "manifest.json").read_text())
    assert man["n_steps"] == 10 and man["fps"] == 10 and not man["has_wrist"]
    assert man["repo_id"] == "usintent/test" and len(man["episodes"]) == 3
    z = np.load(out / man["episodes"][0])
    assert set(z.files) == {"state", "actions", "ultrasound"}
    assert np.array_equal(z["ultrasound"], eps[0].bmode)
    with pytest.raises(FileExistsError):
        uv.export_usvla_episodes(eps, out, _spec(), "usintent/test")


def test_params_and_commands(tmp_path) -> None:
    p = uv.train_params(_spec(20), "usintent/r", 2, 1000, 8, 5e-5)
    assert p["action_horizon"] == 20 and p["seed"] == 2 and p["config_name"] == "pi05_ur_usfm_fusion"
    cmds = uv.run_commands(SCRIPT, tmp_path / "e", tmp_path / "run.json")
    assert cmds["train"][:3] == ["uv", "run", "python"] and "train" in cmds["train"]


class _FakeClient:
    def __init__(self, meta, horizon, pose):
        self.meta, self.h, self.pose, self.sent = meta, horizon, pose, []

    def get_server_metadata(self):
        return self.meta

    def infer(self, obs):
        self.sent.append(obs)
        return {"actions": np.repeat(self.pose[None] + [0.01, 0, 0, 0, 0, 0], self.h, 0)}


def test_client_logic(eps) -> None:
    from usintent.data.chunks import ChunkSampler

    s = ChunkSampler(eps, 1.0, history_k=load("data")["history_frames_k"])
    h = s.history(0, 40)
    pose = h["state"][-1][:6]
    meta = {"usintent": {"arm_id": "USVLA-1", "seed": 1, "run": "r"}}
    pol = uv.USVLAPolicy(_spec(), "x", 0, {"arm_id": "USVLA-1", "seed": 1, "run": "r"}, seed=1,
                         integration_steps=4, client=_FakeClient(meta, 10, pose))
    chunk, m = pol.predict(h)
    assert chunk.shape == (10, 5) and m == "SWEEP" and np.allclose(np.abs(chunk[:, :2]).sum(1), 0.01)
    sent = pol.client.sent[0]
    assert sent[uv.NOISE_KEY] == 1 and sent[uv.STEPS_KEY] == 4 and sent["prompt"] == uv.TASK
    assert sent["images"]["ultrasound_camera"].shape == (32, 32, 3)
    assert not sent["images"]["side_camera"].any()
    with pytest.raises(PermissionError):
        pol.predict(h, "HOLD")
    with pytest.raises(RuntimeError, match="expected"):
        uv.USVLAPolicy(_spec(), "x", 0, {"seed": 2}, client=_FakeClient(meta, 10, pose))
    bad = uv.USVLAPolicy(_spec(20), "x", 0, {}, client=_FakeClient(meta, 10, pose))
    with pytest.raises(RuntimeError, match="steps"):
        bad.predict(h)


# ------------------------------------------------------------- cross-environment checks


@pytest.mark.skipif(not os.environ.get("USINTENT_USVLA_LEROBOT"),
                    reason="needs the LeRobot commit US-VLA pins (USINTENT_USVLA_LEROBOT)")
def test_build_dataset_with_pinned_lerobot(eps, tmp_path) -> None:
    export = uv.export_usvla_episodes(eps, tmp_path / "x", _spec(), "usintent/usvla_test")
    env = {**os.environ, "PYTHONPATH": os.environ["USINTENT_USVLA_LEROBOT"],
           "HF_LEROBOT_HOME": str(tmp_path / "lerobot_home")}
    for _ in range(2):                        # second call: dataset exists -> check only
        r = subprocess.run([sys.executable, "-P", str(SCRIPT), "build-dataset", "--export", str(export)],
                           env=env, cwd=tmp_path, capture_output=True, text=True, check=False)
        assert r.returncode == 0, r.stderr[-2000:]
        out = json.loads(r.stdout.strip().splitlines()[-1])
        assert out["max_err"] == 0 and out["mismatches"] == 0 and out["n_checked"] > 0
    info = json.loads((tmp_path / "lerobot_home" / "usintent/usvla_test" / "meta" / "info.json").read_text())
    assert info["codebase_version"] == "v2.1"
    assert set(info["features"]) >= {"base_rgb", "wrist_rgb", "ultrasound_rgb", "state", "actions"}


@pytest.mark.skipif(not os.environ.get("USINTENT_USVLA_ROOT"),
                    reason="needs a US-VLA checkout (USINTENT_USVLA_ROOT) and openpi_client")
def test_client_against_usvla_server(eps) -> None:
    pytest.importorskip("openpi_client")
    root = Path(os.environ["USINTENT_USVLA_ROOT"])
    server_mod = runpy.run_path(str(root / "src/openpi/serving/websocket_policy_server.py"))
    run_mod = runpy.run_path(str(SCRIPT))
    calls = []

    class StandIn:              # openpi Policy surface used by SeededPolicy: infer(obs, noise=)
        def __init__(self):
            self._sample_kwargs = {}

        def infer(self, obs, *, noise=None):
            calls.append((dict(self._sample_kwargs), None if noise is None else noise.copy()))
            st = np.asarray(obs["state"], np.float64)
            return {"actions": np.repeat(st[None] + [0, 0.02, 0, 0, 0, 0], 10, 0)}

    meta = {"usintent": {"arm_id": "USVLA-1", "seed": 0, "run": "r"}}
    srv = server_mod["WebsocketPolicyServer"](
        policy=run_mod["SeededPolicy"](StandIn(), 10, 32, meta), host="127.0.0.1", port=18765,
        metadata=meta)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    time.sleep(1.0)
    from usintent.data.chunks import ChunkSampler

    s = ChunkSampler(eps, 1.0, history_k=load("data")["history_frames_k"])
    pol = uv.USVLAPolicy(_spec(), "127.0.0.1", 18765, {"arm_id": "USVLA-1", "seed": 0, "run": "r"},
                         seed=0, integration_steps=16)
    chunk, m = pol.predict(s.history(0, 40))
    again, _ = pol.predict(s.history(0, 40))
    assert chunk.shape == (10, 5) and m == "SWEEP" and np.allclose(np.abs(chunk[:, :2]).sum(1), 0.02)
    assert calls[0][0] == {"num_steps": 16} and calls[0][1].shape == (10, 32)
    assert np.array_equal(calls[0][1], calls[1][1]) and np.array_equal(chunk, again)
    with pytest.raises(RuntimeError):
        uv.USVLAPolicy(_spec(), "127.0.0.1", 18765, {"seed": 3})
