"""Pretrained LeRobot VLAs (SmolVLA, pi0.5) as external comparison families.

These arms are fine-tuned with LeRobot's own trainer (``lerobot-train``) from the authors'
released base checkpoints, so the baseline is the published recipe, not our re-implementation.
Our code only (1) exports a LeRobot v3.0 training copy whose per-frame action reproduces our
chunk target, (2) checks that through LeRobot's own loader (CLAUDE.md rule 4), (3) writes the
seeded training command, and (4) wraps the trained checkpoint behind our Policy interface.

Source read from the installed lerobot==0.6.1:
- Base checkpoint ids: ``lerobot/smolvla_base`` (policies/smolvla/modeling_smolvla.py docstring,
  fine-tune example ``lerobot-train --policy.path=lerobot/smolvla_base --dataset.repo_id=...
  --batch_size=64 --steps=200000``); ``lerobot/pi05_base`` (policies/pretrained.py).
- ``lerobot-train`` = lerobot.scripts.lerobot_train:main; draccus fields of TrainPipelineConfig
  (configs/train.py): ``seed, steps, batch_size, output_dir, job_name, save_freq, rename_map,
  use_policy_training_preset``; DatasetConfig (configs/default.py): ``repo_id, root``;
  PreTrainedConfig (configs/policies.py): ``push_to_hub, device``. With ``--policy.path`` the
  pretrained config's input features are kept (policies/factory.py make_policy), so dataset keys
  must be mapped with ``rename_map`` (validate_visual_features_consistency runs only without
  one). The checkpoint's expected keys are read from its config.json at the lab: the arm config
  key ``rename_map`` has no default.
- SmolVLAConfig / PI05Config: ``chunk_size, n_action_steps, optimizer_lr,
  scheduler_warmup_steps, scheduler_decay_steps, max_state_dim=32, max_action_dim=32,
  empty_cameras``; SmolVLA flow steps ``num_steps``; pi0.5 ``num_inference_steps``.
  pi0.5's preprocessor raises "State is required for PI05" without ``observation.state``.
- Checkpoints: ``<output_dir>/checkpoints/last/pretrained_model`` (utils/constants.py).
- Inference: ``lerobot.policies.utils.prepare_observation_for_inference`` (HWC uint8 images ->
  CHW float in [0, 1], batch dim, ``task``), then the pre-processor, ``predict_action_chunk``
  and the post-processor (common/control_utils.py predict_action).

Action representation (identical to the GR00T arm, usintent.policies.gr00t_adapter): per-frame
10-D action = tool-frame step t->t+1 (6-D) + one-hot operator mode at t+1. LeRobot loads the
``chunk_size`` actions of frames t..t+n-1, which compose to our chunk at t; m_hat = argmax of the
mode part of the first step. Blind arms get a constant zero state (see VLAArmSpec.state_dim).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from usintent.data.chunks import chunks_from_poses, compose_actions, policy_target
from usintent.data.loaders import MODES, Episode
from usintent.policies.base import Policy
from usintent.policies.gr00t_adapter import TASK, action_vector

FAMILIES = {"smolvla": "num_steps", "pi05": "num_inference_steps"}   # family -> flow-steps field
IMG_KEY, CAM_KEY, STATE_KEY = "observation.images.bmode", "observation.images.camera", "observation.state"
VLA_ACT_DIM = 6 + len(MODES)


@dataclass(frozen=True)
class VLAArmSpec:
    arm_id: str
    family: str                    # "smolvla" | "pi05"
    n_steps: int                   # chunk length = chunk_size = n_action_steps
    use_camera: bool
    use_proprio: bool

    @property
    def state_dim(self) -> int:
        """Blind arms get a constant zero state of size 2, not 1: LeRobot stores a shape-(1,)
        feature as a scalar column and returns it without its feature axis (found by
        tests/test_external_baselines.py), while the VLA state projections expect (B, d)."""
        return 12 if self.use_proprio else 2


def arm_spec_from_config(arm_cfg: dict[str, Any], n_steps: int, use_camera_topic: bool) -> VLAArmSpec:
    """configs/arms/{SmolVLA,PI05}-*.yaml -> VLAArmSpec. Intent-aware VLA arms are not defined:
    the scheduler mode would have to enter through language (scripts/run_language_check.py)."""
    fam = str(arm_cfg["family"])
    if fam not in FAMILIES:
        raise ValueError(f"{arm_cfg['arm_id']} is not a LeRobot VLA arm")
    inputs = set(arm_cfg["inputs"])
    if arm_cfg["intent_aware"] or "mode_embedding" in inputs or "bmode_history" in inputs:
        raise ValueError(f"{arm_cfg['arm_id']}: VLA arms are intent-blind single-frame arms")
    return VLAArmSpec(str(arm_cfg["arm_id"]), fam, n_steps,
                      use_camera="camera" in inputs and use_camera_topic,
                      use_proprio="proprio" in inputs)


def state_rows(spec: VLAArmSpec, ep: Episode) -> NDArray[np.float32]:
    if spec.use_proprio:
        return ep.state.astype(np.float32)
    return np.zeros((len(ep), spec.state_dim), np.float32)


def export_vla_dataset(episodes: list[Episode], root: Path, spec: VLAArmSpec,
                       repo_id: str = "local/usintent_vla") -> Path:
    """Write the LeRobot v3.0 training copy for a VLA arm (images lossless PNG, no video)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(root)
    if root.exists():
        raise FileExistsError(root)
    if not episodes or any(e.bmode is None for e in episodes):
        raise ValueError("VLA export needs B-mode frames")
    fps = episodes[0].fps
    if any(e.fps != fps for e in episodes) or abs(fps - round(fps)) > 1e-9:
        raise ValueError("LeRobot needs one integer fps for all episodes")
    h, w = episodes[0].bmode.shape[1:]  # type: ignore[union-attr]
    feats: dict[str, dict[str, Any]] = {
        IMG_KEY: {"dtype": "image", "shape": (h, w, 3), "names": ["height", "width", "channel"]},
        STATE_KEY: {"dtype": "float32", "shape": (spec.state_dim,), "names": None},
        "action": {"dtype": "float32", "shape": (VLA_ACT_DIM,), "names": None},
    }
    if spec.use_camera:
        if any(e.camera is None for e in episodes):
            raise ValueError("arm uses the camera but an episode has none")
        feats[CAM_KEY] = {"dtype": "image", "shape": tuple(episodes[0].camera.shape[1:]),  # type: ignore[union-attr]
                          "names": ["height", "width", "channel"]}
    ds = LeRobotDataset.create(repo_id, fps=round(fps), features=feats, root=root,
                               robot_type="ur7e", use_videos=False)
    for ep in episodes:
        act, st = action_vector(ep), state_rows(spec, ep)
        assert ep.bmode is not None
        for k in range(len(ep)):
            frame: dict[str, Any] = {"task": TASK, STATE_KEY: st[k], "action": act[k],
                                     IMG_KEY: np.repeat(ep.bmode[k][:, :, None], 3, axis=2)}
            if spec.use_camera:
                assert ep.camera is not None
                frame[CAM_KEY] = ep.camera[k]
            ds.add_frame(frame)
        ds.save_episode()
    ds.finalize()
    (root / "meta" / "usintent_vla.json").write_text(json.dumps(
        {"arm_id": spec.arm_id, "family": spec.family, "n_steps": spec.n_steps,
         "episode_indices": [int(e.episode_index) for e in episodes]}, indent=2))
    return root


def decode_chunk(actions: NDArray[np.float64]) -> tuple[NDArray[np.float64], str]:
    """(n, 10) predicted per-frame actions -> our (n, 5) chunk and m_hat."""
    a = np.asarray(actions, dtype=np.float64)
    return policy_target(compose_actions(a[:, :6])), MODES[int(np.argmax(a[0, 6:]))]


def vla_passthrough(root: Path, episodes: list[Episode], n_steps: int, tol: float = 1e-5,
                    max_starts: int = 400, seed: int = 0,
                    repo_id: str = "local/usintent_vla") -> dict[str, float]:
    """CLAUDE.md rule 4 for the VLA path: load the exported copy with **LeRobot's own loader**
    and ``delta_timestamps`` (exactly how lerobot-train builds action chunks), decode with
    :func:`decode_chunk`, and require the chunk computed from recorded poses and the operator mode
    at t+1. ``tol`` covers float32 storage of the per-step actions."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from usintent.data.passthrough import PassthroughError

    fps = episodes[0].fps
    ds = LeRobotDataset(repo_id, root=Path(root),
                        delta_timestamps={"action": [k / fps for k in range(n_steps)]})
    rng = np.random.default_rng(seed)
    offset, worst, bad_mode, pads, n = 0, 0.0, 0, 0, 0
    per_ep = max(1, max_starts // len(episodes))
    for ep in episodes:
        starts = np.arange(0, len(ep) - n_steps - 1)
        if len(starts) > per_ep:
            starts = np.sort(rng.choice(starts, per_ep, False))
        ref = policy_target(chunks_from_poses(ep.state[:, :6], starts, n_steps))
        for i, s in enumerate(starts):
            item = ds[int(offset + s)]
            if int(item["frame_index"]) != int(s):
                raise PassthroughError(f"frame order mismatch at episode {ep.episode_index}, {s}")
            pads += int(item["action_is_pad"].any())
            chunk, m = decode_chunk(item["action"].numpy())
            worst = max(worst, float(np.max(np.abs(chunk - ref[i]))))
            bad_mode += int(m != ep.contact_cmd[s + 1])
            n += 1
        offset += len(ep)
    out = {"max_err": worst, "mode_mismatches": float(bad_mode), "padded_chunks": float(pads),
           "n_chunks": float(n)}
    if worst > tol or bad_mode or pads:
        raise PassthroughError(f"VLA passthrough failed: {out}")
    return out


def decode_passthrough(episodes: list[Episode], n_steps: int, tol: float = 1e-6,
                       max_starts: int = 2000, seed: int = 0) -> dict[str, float]:
    """Rule 4 at evaluation time (no export needed): the recorded per-frame actions of the test
    episodes, decoded by :func:`decode_chunk`, reproduce the pose chunks and operator modes."""
    from usintent.data.passthrough import PassthroughError

    rng = np.random.default_rng(seed)
    per_ep = max(1, max_starts // max(1, len(episodes)))
    worst, bad, n = 0.0, 0, 0
    for ep in episodes:
        a = action_vector(ep).astype(np.float64)
        starts = np.arange(0, len(ep) - n_steps - 1)
        if len(starts) > per_ep:
            starts = np.sort(rng.choice(starts, per_ep, False))
        ref = policy_target(chunks_from_poses(ep.state[:, :6], starts, n_steps))
        for i, s in enumerate(starts):
            chunk, m = decode_chunk(a[s:s + n_steps])
            worst = max(worst, float(np.max(np.abs(chunk - ref[i]))))
            bad += int(m != ep.contact_cmd[s + 1])
            n += 1
    out = {"max_err": worst, "mode_mismatches": float(bad), "n_chunks": float(n)}
    if worst > tol or bad:
        raise PassthroughError(f"VLA decode passthrough failed: {out}")
    return out


def train_command(spec: VLAArmSpec, dataset: Path, base_checkpoint: str, out_dir: Path, seed: int,
                  steps: int, batch_size: int, lr: float, rename_map: dict[str, str],
                  extra: dict[str, Any] | None = None, device: str = "cuda",
                  repo_id: str = "local/usintent_vla") -> list[str]:
    """``lerobot-train`` command: fine-tune the released base checkpoint with LeRobot's own
    policy preset (AdamW + cosine schedule), our shared steps / batch size / peak lr and seed.
    The cosine decay spans the run (``scheduler_decay_steps = steps``). ``extra`` adds
    ``--key=value`` overrides from the arm config (e.g. ``policy.empty_cameras``)."""
    cmd = ["lerobot-train", f"--policy.path={base_checkpoint}",
           f"--dataset.repo_id={repo_id}", f"--dataset.root={dataset}",
           f"--output_dir={out_dir}", f"--job_name={spec.arm_id}_seed{seed}", f"--seed={seed}",
           f"--steps={steps}", f"--batch_size={batch_size}", f"--save_freq={steps}",
           f"--policy.optimizer_lr={lr}", f"--policy.scheduler_decay_steps={steps}",
           f"--policy.chunk_size={spec.n_steps}", f"--policy.n_action_steps={spec.n_steps}",
           f"--policy.device={device}", "--policy.push_to_hub=false",
           f"--rename_map={json.dumps(rename_map, separators=(',', ':'))}"]
    for k, v in (extra or {}).items():
        cmd.append(f"--{k}={json.dumps(v) if isinstance(v, (dict, list, bool)) else v}")
    return cmd


def checkpoint_dir(run_dir: Path) -> Path:
    """Final checkpoint written by lerobot-train under our run directory."""
    p = Path(run_dir) / "lerobot" / "checkpoints" / "last" / "pretrained_model"
    if not p.is_dir():
        raise FileNotFoundError(f"{p}: no lerobot-train checkpoint")
    return p


def observation(spec: VLAArmSpec, history: dict[str, Any]) -> dict[str, NDArray[Any]]:
    """Raw observation in LeRobot's inference format (HWC uint8 images, 1-D state), before
    ``prepare_observation_for_inference``. Uses the latest frame only."""
    obs: dict[str, NDArray[Any]] = {
        IMG_KEY: np.repeat(np.asarray(history["bmode"][-1], np.uint8)[:, :, None], 3, axis=2),
        STATE_KEY: (np.asarray(history["state"][-1], np.float32) if spec.use_proprio
                    else np.zeros(spec.state_dim, np.float32))}
    if spec.use_camera:
        obs[CAM_KEY] = np.asarray(history["camera"][-1], np.uint8)
    return obs


class LeRobotVLAPolicy(Policy):
    """A lerobot-train checkpoint (SmolVLA or pi0.5) behind our Policy interface.

    Needs the policy's weights and tokenizer (lab GPU machine). ``deterministic_noise`` seeds
    torch before every call; ``integration_steps`` sets the flow-matching step count.
    """

    def __init__(self, spec: VLAArmSpec, checkpoint: Path, rename_map: dict[str, str],
                 device: str = "cuda", seed: int = 0, deterministic_noise: bool = True,
                 integration_steps: int | None = None):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        if not (Path(checkpoint) / "config.json").is_file():   # else LeRobot reads it as a hub id
            raise FileNotFoundError(f"{checkpoint}: not a lerobot-train checkpoint directory")
        self.arm_id, self.spec, self.seed = spec.arm_id, spec, seed
        self.deterministic_noise, self.device = deterministic_noise, device
        cfg = PreTrainedConfig.from_pretrained(str(checkpoint))
        cfg.device = device
        if integration_steps is not None:
            setattr(cfg, FAMILIES[spec.family], integration_steps)
        self.model = get_policy_class(cfg.type).from_pretrained(str(checkpoint), config=cfg)
        self.pre, self.post = make_pre_post_processors(
            cfg, pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": device},
                                    "rename_observations_processor": {"rename_map": rename_map}})

    def _predict(self, history: Any, mode: str | None) -> tuple[NDArray[np.float64], str]:
        import torch
        from lerobot.policies.utils import prepare_observation_for_inference

        if self.deterministic_noise:
            torch.manual_seed(self.seed)
        obs = prepare_observation_for_inference(observation(self.spec, history),
                                                torch.device(self.device), TASK, "ur7e")
        with torch.inference_mode():
            act = self.post(self.model.predict_action_chunk(self.pre(obs)))
        return decode_chunk(act[0].float().cpu().numpy())
