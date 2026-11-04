"""GR00T-H adapter: configs from the arm YAMLs, LeRobot-v2 export, GR00T-path passthrough.
SYNTHETIC data. Training/inference need the Isaac-GR00T environment (lab)."""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from usintent.config import load
from usintent.data.loaders import MODES
from usintent.data.passthrough import PassthroughError
from usintent.policies.gr00t_adapter import (
    action_vector,
    arm_spec_from_config,
    decode_action,
    export_gr00t_dataset,
    finetune_command,
    gr00t_passthrough,
    modality_json,
)
from usintent.sim.holdworld import HoldWorldConfig, generate


def _spec(arm, **kw):
    return arm_spec_from_config(load(f"arms/{arm}"), 10, 3, camera_present=kw.pop("cam", False), **kw)


def test_specs_follow_table_1() -> None:
    assert _spec("A1").state_keys == (("bias", 1),) and _spec("A1").video_delta == (0,)
    assert _spec("A2").video_delta == (-2, -1, 0) and _spec("A2").state_keys == (("bias", 1),)
    assert _spec("A3").state_keys == (("tcp_pose", 6),)
    assert ("mode", len(MODES)) in _spec("A5").state_keys
    assert _spec("A6").state_keys == (("tcp_pose", 6), ("time_since_motion", 1))
    assert ("mode", len(MODES)) in _spec("A7").state_keys      # fed the when-head's m_hat
    for arm in ("A1", "A2", "A3", "A6"):                         # blind arms: no mode input
        assert "mode" not in dict(_spec(arm).state_keys)
    assert _spec("A1", cam=True).use_camera and not _spec("A1").use_camera
    mj = modality_json(_spec("A6"))
    assert mj["state"]["time_since_motion"] == {"start": 6, "end": 7}
    assert mj["action"]["contact_cmd"] == {"start": 6, "end": 10}


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    eps = generate(HoldWorldConfig(image_shape=(32, 32), fps=10.0, n_stations=(2, 3),
                                   tool_yaw_rad=0.5, seed=3), 3)
    out = export_gr00t_dataset(eps, tmp_path_factory.mktemp("g") / "ds", _spec("A5"), 0.002)
    return eps, out


def test_export_layout_and_lossless_video(exported) -> None:
    import av

    eps, out = exported
    info = json.loads((out / "meta/info.json").read_text())
    assert info["codebase_version"] == "v2.1" and info["total_episodes"] == 3
    assert info["total_frames"] == sum(len(e) for e in eps)
    for f in ("modality.json", "tasks.jsonl", "episodes.jsonl", "stats.json"):
        assert (out / "meta" / f).is_file()
    assert (out / "modality_config.py").is_file()
    df = pd.read_parquet(out / "data/chunk-000/episode_000000.parquet")
    assert len(df) == len(eps[0]) and len(df["action"].iloc[0]) == 10
    assert np.allclose(np.stack(df["action"]), action_vector(eps[0]), atol=1e-6)
    st = np.stack(df["observation.state"])
    assert np.allclose(st[:, :4].sum(1), 1.0)                    # one-hot operator mode
    mp4 = out / "videos/chunk-000/observation.images.bmode/episode_000000.mp4"
    frames = [f.to_ndarray(format="gray") for f in av.open(str(mp4)).decode(video=0)]
    assert len(frames) == len(eps[0])
    assert all(np.array_equal(a, b) for a, b in zip(frames, eps[0].bmode))   # lossless
    stats = json.loads((out / "meta/stats.json").read_text())
    assert set(stats["action"]) == {"mean", "std", "min", "max", "q01", "q99"}


def test_gr00t_passthrough_zero_error_and_catches_bugs(exported) -> None:
    eps, _ = exported
    res = gr00t_passthrough(eps, 10)
    assert res["max_err"] < 1e-6 and res["mode_mismatches"] == 0
    bad = eps[0]
    bad.action = np.roll(bad.action, 1, axis=0)
    with pytest.raises(PassthroughError):
        gr00t_passthrough([bad], 10)


def test_decode_action_prefixed_keys() -> None:
    a = {"action.tool_step": np.zeros((1, 4, 6)), "action.contact_cmd": np.eye(4)[None, [2, 0, 0, 0]]}
    chunk, m = decode_action(a)
    assert chunk.shape == (4, 5) and m == "LIFT"


def test_finetune_command_sets_seed_and_no_state_dropout(exported) -> None:
    _, out = exported
    cmd = finetune_command(_spec("A5"), out, "nvidia/GR00T-H-N1.7", out / "run", 3, 100, 8, 1e-4,
                           out / "gr00t")
    assert cmd[cmd.index("--seed") + 1] == "3"
    assert cmd[cmd.index("--state-dropout-prob") + 1] == "0.0"
    assert cmd[cmd.index("--embodiment-tag") + 1] == "NEW_EMBODIMENT"


def test_train_arm_dry_run_exports() -> None:
    from usintent.paths import SCRIPTS

    r = subprocess.run([sys.executable, str(SCRIPTS / "run_train_arm.py"), "--arm", "A2",
                        "--seed", "1", "--dry-run"], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
    assert '"gr00t_passthrough"' in r.stdout and "gr00t_finetune_seeded.py" in r.stdout


@pytest.mark.lab
def test_modality_config_runs_against_gr00t_source(exported) -> None:
    """Needs an Isaac-GR00T checkout: GR00T_ROOT=/path/to/Isaac-GR00T pytest -m lab."""
    root = os.environ.get("GR00T_ROOT")
    assert root, "set GR00T_ROOT to an Isaac-GR00T checkout"
    _, out = exported
    code = ("import runpy; runpy.run_path('modality_config.py'); "
            "from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS; "
            "c = MODALITY_CONFIGS['new_embodiment']; "
            "print(c['action'].modality_keys, len(c['action'].delta_indices), c['state'].modality_keys)")
    r = subprocess.run([sys.executable, "-c", code], cwd=out, capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": root}, check=False)
    assert r.returncode == 0, r.stderr
    assert "['tool_step', 'contact_cmd'] 10" in r.stdout
