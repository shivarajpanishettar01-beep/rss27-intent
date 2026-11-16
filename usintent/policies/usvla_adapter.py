"""US-VLA (Zhang, Wu et al., ACM MM 2026) as an external, ultrasound-specific comparison arm.

US-VLA is pi0.5 + a USFM ultrasound encoder + an ultrasound-aware expert-fusion module, released
at https://github.com/VMVLab/US-VLA (openpi fork, JAX, its own uv environment). It is used as
released: fine-tuned with its own ``pi05_ur_usfm_fusion`` train config and served by its own
policy server. Everything that runs in the US-VLA environment is in ``scripts/usvla_run.py``.

Read from the US-VLA source (commit dde576a, 2026-08-17):
- Data (README "Data", training/config.py LeRobotURDataConfig): a LeRobot dataset with per-frame
  ``state`` (6-D TCP pose), ``actions`` (6-D absolute TCP pose target), images ``base_rgb``
  (side camera), ``wrist_rgb``, ``ultrasound_rgb`` and the task as prompt
  (``prompt_from_task=True``). No delta-action transform: the model predicts absolute poses.
- openpi's loader (training/data_loader.py create_torch_dataset) builds ``action_horizon``
  actions from frame t with ``delta_timestamps = [k / fps for k in range(action_horizon)]``,
  from the LeRobot commit pinned in its pyproject (0cf8648, dataset format v2.1,
  ``lerobot.common.datasets``). That LeRobot cannot share an environment with lerobot 0.6.1,
  so the training copy is built in two steps (``export_usvla_episodes`` here, then
  ``usvla_run.py build-dataset`` there; both check the action stream, CLAUDE.md rule 4).
- policies/ur_policy.py: all three image masks are True (no missing-camera path); the output
  transform returns ``actions[:, :6]``. ``Pi0Config.action_horizon`` defaults to 50; inference
  noise is (1, action_horizon, action_dim = 32) standard normal, replaceable through
  ``Policy.infer(obs, noise=...)``; ``sample_actions(num_steps=10)`` sets the flow steps.
- The ablation configs listed in the README (``pi05_ur_us``, ``pi05_ur_usfm``, ...) are **not**
  in the released ``config.py``; only ``pi05_ur_usfm_fusion`` is.

Our conventions: state = our TCP pose ``Episode.state[:, :6]`` (base frame, m and rotation
vector, the UR convention); action at frame k = pose at k+1 (last frame repeats). A predicted
absolute chunk is decoded relative to the current pose into our tool-frame chunk with the same
``chunks_from_poses`` used for ground truth. US-VLA has no contact output: it is executed as
``SWEEP`` (contact kept by our force controller, tangential motion as predicted), the exact
execution semantics of a pose-only policy under usintent.control.controller.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from usintent.data.chunks import chunks_from_poses, policy_target
from usintent.data.loaders import Episode
from usintent.policies.base import Policy
from usintent.policies.gr00t_adapter import TASK

TRAIN_CONFIG = "pi05_ur_usfm_fusion"   # the released US-VLA full-model config
EXECUTED_MODE = "SWEEP"                # pose-only policy: contact held by the force controller
NOISE_KEY, STEPS_KEY = "__usintent_noise_seed", "__usintent_num_steps"   # read by usvla_run.py serve


@dataclass(frozen=True)
class USVLAArmSpec:
    arm_id: str
    n_steps: int            # chunk length = US-VLA action_horizon
    use_camera: bool        # our probe camera -> US-VLA wrist_rgb; side camera is always blank
    train_config: str = TRAIN_CONFIG


def arm_spec_from_config(arm_cfg: dict[str, Any], n_steps: int, use_camera_topic: bool) -> USVLAArmSpec:
    if arm_cfg["family"] != "usvla":
        raise ValueError(f"{arm_cfg['arm_id']} is not a US-VLA arm")
    inputs = set(arm_cfg["inputs"])
    if arm_cfg["intent_aware"] or "mode_embedding" in inputs or "bmode_history" in inputs:
        raise ValueError(f"{arm_cfg['arm_id']}: US-VLA arms are intent-blind single-frame arms")
    if "proprio" not in inputs:
        raise ValueError(f"{arm_cfg['arm_id']}: US-VLA predicts absolute poses and needs the TCP pose")
    return USVLAArmSpec(str(arm_cfg["arm_id"]), n_steps,
                        use_camera="camera" in inputs and use_camera_topic,
                        train_config=str(arm_cfg.get("train_config") or TRAIN_CONFIG))


def usvla_actions(ep: Episode) -> NDArray[np.float32]:
    """(N, 6) absolute TCP pose target per frame: the pose at k+1 (last frame repeats)."""
    pose = ep.state[:, :6]
    return np.vstack([pose[1:], pose[-1:]]).astype(np.float32)


def decode_chunk(pred_abs: NDArray[np.float64], current_pose: NDArray[np.float64]
                 ) -> tuple[NDArray[np.float64], str]:
    """(n, 6) predicted absolute poses + (6,) current pose -> our (n, 5) tool-frame chunk."""
    pred = np.asarray(pred_abs, dtype=np.float64)[:, :6]
    poses = np.vstack([np.asarray(current_pose, dtype=np.float64)[None, :6], pred])
    return policy_target(chunks_from_poses(poses, np.array([0]), len(pred)))[0], EXECUTED_MODE


def decode_passthrough(episodes: list[Episode], n_steps: int, tol: float = 1e-5,
                       max_starts: int = 2000, seed: int = 0) -> dict[str, float]:
    """Rule 4: recorded US-VLA actions of frames s..s+n-1 (exactly what openpi's loader gives
    for frame s), decoded against the recorded pose at s, reproduce our ground-truth chunk.
    ``tol`` (m / rad) covers float32 storage of absolute poses."""
    from usintent.data.passthrough import PassthroughError

    rng = np.random.default_rng(seed)
    per_ep = max(1, max_starts // max(1, len(episodes)))
    worst, n = 0.0, 0
    for ep in episodes:
        act, pose = usvla_actions(ep).astype(np.float64), ep.state[:, :6]
        starts = np.arange(0, len(ep) - n_steps - 1)
        if len(starts) > per_ep:
            starts = np.sort(rng.choice(starts, per_ep, False))
        ref = policy_target(chunks_from_poses(pose, starts, n_steps))
        for i, s in enumerate(starts):
            chunk, _ = decode_chunk(act[s:s + n_steps], pose[s].astype(np.float32))
            worst = max(worst, float(np.max(np.abs(chunk - ref[i]))))
            n += 1
    out = {"max_err": worst, "n_chunks": float(n)}
    if worst > tol or n == 0:
        raise PassthroughError(f"US-VLA decode passthrough failed: {out}")
    return out


def export_usvla_episodes(episodes: list[Episode], out: Path, spec: USVLAArmSpec,
                          repo_id: str) -> Path:
    """Step 1 of the training copy (our environment): one ``.npz`` per episode plus
    ``manifest.json``, read by ``usvla_run.py build-dataset`` in the US-VLA environment."""
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    if not episodes or any(e.bmode is None for e in episodes):
        raise ValueError("US-VLA export needs B-mode frames")
    fps = episodes[0].fps
    if any(e.fps != fps for e in episodes) or abs(fps - round(fps)) > 1e-9:
        raise ValueError("LeRobot needs one integer fps for all episodes")
    if spec.use_camera and any(e.camera is None for e in episodes):
        raise ValueError("arm uses the camera but an episode has none")
    out.mkdir(parents=True)
    files = []
    for ep in episodes:
        arrays: dict[str, NDArray[Any]] = {"state": ep.state[:, :6].astype(np.float32),
                                           "actions": usvla_actions(ep),
                                           "ultrasound": np.asarray(ep.bmode, np.uint8)}
        if spec.use_camera:
            arrays["wrist"] = np.asarray(ep.camera, np.uint8)
        name = f"episode_{int(ep.episode_index):06d}.npz"
        np.savez_compressed(out / name, **arrays)  # type: ignore[arg-type]
        files.append(name)
    side_shape = (list(episodes[0].camera.shape[1:3]) if spec.use_camera     # type: ignore[union-attr]
                  else list(episodes[0].bmode.shape[1:3]))                    # type: ignore[union-attr]
    (out / "manifest.json").write_text(json.dumps(
        {"arm_id": spec.arm_id, "repo_id": repo_id, "fps": round(fps), "n_steps": spec.n_steps,
         "task": TASK, "has_wrist": spec.use_camera, "blank_shape": side_shape,
         "train_config": spec.train_config, "episodes": files,
         "episode_indices": [int(e.episode_index) for e in episodes]}, indent=2))
    return out


def train_params(spec: USVLAArmSpec, repo_id: str, seed: int, steps: int, batch_size: int,
                 lr: float, wandb: bool = False) -> dict[str, Any]:
    """Overrides applied by ``usvla_run.py`` to the released train config (same dict for
    ``train`` and ``serve``, so the served model is built exactly as it was trained)."""
    return {"arm_id": spec.arm_id, "config_name": spec.train_config, "repo_id": repo_id,
            "exp_name": f"usintent_{spec.arm_id}_seed{seed}", "seed": seed, "steps": steps,
            "batch_size": batch_size, "lr": lr, "action_horizon": spec.n_steps, "wandb": wandb}


def run_commands(script: Path, export: Path, run_json: Path) -> dict[str, list[str]]:
    """Commands to run **inside the US-VLA checkout** (cwd = its root, ``uv run``)."""
    base = ["uv", "run", "python", str(script)]
    return {"build_dataset": base + ["build-dataset", "--export", str(export)],
            "train": base + ["train", "--run-json", str(run_json)],
            "serve": base + ["serve", "--run-json", str(run_json), "--port", "<port>"]}


def observation(spec: USVLAArmSpec, history: dict[str, Any]) -> dict[str, Any]:
    """Latest frame in US-VLA's inference format (its README / ur_policy.make_ur_example)."""
    us = np.repeat(np.asarray(history["bmode"][-1], np.uint8)[:, :, None], 3, axis=2)
    wrist = (np.asarray(history["camera"][-1], np.uint8) if spec.use_camera
             else np.zeros_like(us))
    return {"state": np.asarray(history["state"][-1][:6], np.float32),
            "images": {"side_camera": np.zeros_like(wrist), "wrist_camera": wrist,
                       "ultrasound_camera": us},
            "prompt": TASK}


class USVLAPolicy(Policy):
    """Client of a running ``usvla_run.py serve`` (openpi's WebsocketPolicyServer).

    The server's metadata must name this arm, seed and run, so a chunk can never be scored
    against the wrong checkpoint. ``deterministic_noise`` sends a fixed noise seed per call;
    ``integration_steps`` sets openpi's ``num_steps``.
    """

    def __init__(self, spec: USVLAArmSpec, host: str, port: int, expected: dict[str, Any],
                 seed: int = 0, deterministic_noise: bool = True,
                 integration_steps: int | None = None, client: Any = None):
        self.arm_id, self.spec, self.seed = spec.arm_id, spec, seed
        self.deterministic_noise, self.integration_steps = deterministic_noise, integration_steps
        if client is None:
            from openpi_client.websocket_client_policy import WebsocketClientPolicy

            client = WebsocketClientPolicy(host=host, port=port)
        self.client = client
        meta = dict(client.get_server_metadata().get("usintent", {}))
        for k, v in expected.items():
            if meta.get(k) != v:
                raise RuntimeError(f"US-VLA server serves {meta}, expected {k}={v!r}")

    def _predict(self, history: Any, mode: str | None) -> tuple[NDArray[np.float64], str]:
        obs = observation(self.spec, history)
        if self.deterministic_noise:
            obs[NOISE_KEY] = int(self.seed)
        if self.integration_steps is not None:
            obs[STEPS_KEY] = int(self.integration_steps)
        act = np.asarray(self.client.infer(obs)["actions"], dtype=np.float64)
        if act.shape[0] != self.spec.n_steps:
            raise RuntimeError(f"server returned {act.shape[0]} steps, expected {self.spec.n_steps}")
        return decode_chunk(act, np.asarray(history["state"][-1][:6], np.float64))
