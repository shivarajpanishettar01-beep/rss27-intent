"""GR00T-H (Isaac-GR00T) adapter: read the installed repo before coding. SPEC section 10. M6.

Source read (2026-11-04): NVIDIA/Isaac-GR00T, shallow clone at commit c56f541 ("fix(n1d7): stop
padded action dimensions from affecting valid predictions"), model package ``gr00t_n1d7``.
Files and APIs used (quoted from that source):
- getting_started/finetune_new_embodiment.md, data_preparation.md: data must be "GR00T-flavored
  LeRobot v2" with ``meta/modality.json``, ``episodes.jsonl``, ``tasks.jsonl``, ``info.json``,
  ``stats.json``; videos as MP4 at ``videos/chunk-{episode_chunk:03d}/{video_key}/
  episode_{episode_index:06d}.mp4``; parquet at ``data/chunk-{episode_chunk:03d}/
  episode_{episode_index:06d}.parquet`` with ``observation.state`` / ``action`` (1-D float32
  concatenations), ``timestamp``, ``task_index``, ``annotation.human.task_description``.
- gr00t/data/types.py: ``ModalityConfig(delta_indices, modality_keys, ...)``,
  ``ActionConfig(rep, type, format, state_key=None)``, ``ActionRepresentation.{RELATIVE,DELTA,
  ABSOLUTE}``, ``ActionType.{EEF,NON_EEF}``, ``ActionFormat.{DEFAULT,XYZ_ROT6D,XYZ_ROTVEC}``.
- gr00t/configs/data/embodiment_configs.py: ``register_modality_config(config, embodiment_tag=
  EmbodimentTag.NEW_EMBODIMENT)``.
- gr00t/experiment/launch_finetune.py + gr00t/configs/finetune_config.py: CLI flags
  ``--base-model-path --dataset-path --embodiment-tag --modality-config-path --tune-llm
  --tune-visual --tune-projector --tune-diffusion-model --state-dropout-prob --global-batch-size
  --learning-rate --max-steps --output-dir --save-steps``. The seed is ``config.data.seed``
  (gr00t/configs/data/data_config.py, default 42; ``set_seed(config.data.seed)`` in
  gr00t/experiment/experiment.py) and is NOT exposed by the CLI: scripts/gr00t_finetune_seeded.py
  sets it.
- gr00t/policy/gr00t_policy.py: ``Gr00tPolicy(embodiment_tag, model_path, *, device)``,
  ``get_action(observation) -> (action_dict, info)``; observation = {"video": {key: uint8
  (B,T,H,W,C)}, "state": {key: float32 (B,T,D)}, "language": {key: [[str]] * B}}.
- gr00t/model/gr00t_n1d7/gr00t_n1d7.py: the action head embeds the state with
  ``self.state_encoder`` (CategorySpecificMLP, per-embodiment), then
  ``sa_embs = torch.cat((state_features, action_features), dim=1)`` is the token sequence of the
  flow-matching DiT, which cross-attends to the frozen VL backbone features. State features are
  dropped with probability ``state_dropout_prob`` during training.

Design (see reports/REVIEW_FIXES.md):
- **Action**: our per-step tool-frame displacement ``tool_step`` (6-D) as a NON_EEF / ABSOLUTE
  key, so no pose state is needed (A1/A2 have no proprioception); the chunk of n steps is composed
  in our tested decode path (usintent.data.chunks.compose_actions) and projected to 5-D. Second
  action key ``contact_cmd`` (4-D one-hot of the operator mode at the next step) is the contact
  head; m_hat = argmax of its first step.
- **Where the A5 mode enters**: as a state key ``mode`` (4-D one-hot of the operator key-event
  mode). It is embedded by the action head's ``state_encoder`` (a learned per-embodiment MLP, i.e.
  a learned embedding of the mode) and enters the DiT as the state token concatenated with the
  noisy-action tokens (``sa_embs``). ``state_dropout_prob`` must be 0 for every arm (shared
  schedule), otherwise the mode is randomly hidden during training.
- A3/A6 proprioception: state key ``tcp_pose`` (6-D); A6 also ``time_since_motion`` (1-D).
  Arms without any state input get a constant 1-D ``bias`` state (GR00T requires a state key).
- A7 = A5-type model whose mode input is the when-head's committed m_hat, not the scheduler's
  (usintent.whenhead.commit.WhenHeadPolicy); A7 itself never receives intent.
- Language: one fixed instruction; the language-mode check exports a variant whose instruction
  encodes the operator mode (scripts/run_language_check.py swaps it at test time).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from usintent.data.chunks import compose_actions, policy_target
from usintent.data.loaders import MODES, Episode
from usintent.data.segmentation import causal_elapsed
from usintent.policies.base import Policy

GR00T_SOURCE = "NVIDIA/Isaac-GR00T@c56f541"
EMBODIMENT_TAG = "NEW_EMBODIMENT"
LANG_KEY = "annotation.human.task_description"
TASK = "hold the probe on the vessel and scan the stations"
MODE_INSTRUCTIONS = {"HOLD": "hold the probe still on the vessel",
                     "SWEEP": "sweep the probe to the next station",
                     "LIFT": "lift the probe off the phantom",
                     "RECONTACT": "bring the probe back into contact"}


@dataclass(frozen=True)
class Gr00tArmSpec:
    arm_id: str
    n_steps: int                       # action horizon (chunk steps)
    video_delta: tuple[int, ...]       # (0,) or (-(k-1), ..., 0)
    use_camera: bool
    state_keys: tuple[tuple[str, int], ...]   # (name, dim) in observation.state order
    language_mode: bool = False        # language-delivery check variant

    @property
    def state_dim(self) -> int:
        return sum(d for _, d in self.state_keys)


def arm_spec_from_config(arm_cfg: dict[str, Any], n_steps: int, history_k: int,
                         camera_present: bool, language_mode: bool = False) -> Gr00tArmSpec:
    """configs/arms/A*.yaml -> Gr00tArmSpec. A7 is built as its A5-type base model."""
    if arm_cfg["family"] != "gr00t_h":
        raise ValueError(f"{arm_cfg['arm_id']} is not a GR00T-H arm")
    inputs = set(arm_cfg["inputs"])
    keys: list[tuple[str, int]] = []
    if "proprio" in inputs:
        keys.append(("tcp_pose", 6))
    if "time_since_motion" in inputs:
        keys.append(("time_since_motion", 1))
    if "mode_embedding" in inputs or "whenhead" in inputs:
        keys.append(("mode", len(MODES)))     # A5: operator mode; A7: when-head m_hat
    if not keys:
        keys.append(("bias", 1))
    delta = tuple(range(-(history_k - 1), 1)) if "bmode_history" in inputs else (0,)
    return Gr00tArmSpec(str(arm_cfg["arm_id"]), n_steps, delta,
                        camera_present and "camera" in inputs, tuple(keys), language_mode)


# ------------------------------------------------------------------------------- configs


def modality_json(spec: Gr00tArmSpec) -> dict[str, Any]:
    """meta/modality.json for the exported dataset."""
    state, start = {}, 0
    for name, dim in spec.state_keys:
        state[name] = {"start": start, "end": start + dim}
        start += dim
    video = {"bmode": {"original_key": "observation.images.bmode"}}
    if spec.use_camera:
        video["camera"] = {"original_key": "observation.images.camera"}
    return {"state": state,
            "action": {"tool_step": {"start": 0, "end": 6},
                       "contact_cmd": {"start": 6, "end": 6 + len(MODES)}},
            "video": video,
            "annotation": {"human.task_description": {"original_key": "task_index"}}}


def modality_config_source(spec: Gr00tArmSpec) -> str:
    """Python file for ``launch_finetune.py --modality-config-path`` (GR00T's own classes)."""
    video_keys = ["bmode"] + (["camera"] if spec.use_camera else [])
    state_keys = [k for k, _ in spec.state_keys]
    return f'''"""Modality config for arm {spec.arm_id} (generated by usintent.policies.gr00t_adapter;
GR00T source {GR00T_SOURCE}). Do not edit by hand."""
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig

_ABS = ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT)
config = {{
    "video": ModalityConfig(delta_indices={list(spec.video_delta)}, modality_keys={video_keys}),
    "state": ModalityConfig(delta_indices=[0], modality_keys={state_keys}),
    "action": ModalityConfig(delta_indices=list(range({spec.n_steps})),
                             modality_keys=["tool_step", "contact_cmd"],
                             action_configs=[_ABS, _ABS]),
    "language": ModalityConfig(delta_indices=[0], modality_keys=["{LANG_KEY}"]),
}}
register_modality_config(config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
'''


# ------------------------------------------------------------------------------- export


def _onehot(cmds: NDArray[np.str_]) -> NDArray[np.float32]:
    out = np.zeros((len(cmds), len(MODES)), np.float32)
    for i, m in enumerate(MODES):
        out[:, i] = cmds == m
    return out


def state_vector(spec: Gr00tArmSpec, ep: Episode, v_hold_mps: float | None,
                 mode: NDArray[np.str_] | None = None) -> NDArray[np.float32]:
    """(N, state_dim) observation.state for an episode. ``mode`` defaults to the operator
    key-event mode (contact_cmd); for A7 the caller passes the when-head's m_hat."""
    cols = []
    for name, _ in spec.state_keys:
        if name == "tcp_pose":
            cols.append(ep.state[:, :6].astype(np.float32))
        elif name == "time_since_motion":
            if v_hold_mps is None:
                raise ValueError("time_since_motion needs v_hold_mps (data.yaml)")
            e = causal_elapsed(ep, v_hold_mps)          # NaN while moving -> -1 marker
            cols.append(np.nan_to_num(e, nan=-1.0)[:, None].astype(np.float32))
        elif name == "mode":
            cols.append(_onehot(ep.contact_cmd if mode is None else mode))
        elif name == "bias":
            cols.append(np.zeros((len(ep), 1), np.float32))
        else:
            raise KeyError(name)
    return np.concatenate(cols, axis=1)


def action_vector(ep: Episode) -> NDArray[np.float32]:
    """(N, 10) per-frame action: tool_step[t] (step t->t+1, tool frame at t) and the one-hot
    operator mode at t+1 (last frame repeats)."""
    nxt = np.concatenate([ep.contact_cmd[1:], ep.contact_cmd[-1:]])
    return np.concatenate([ep.action.astype(np.float32), _onehot(nxt)], axis=1)


def write_mp4(frames: NDArray[np.uint8], path: Path, fps: float) -> None:
    """Grey (N,H,W) -> H.264 crf 0, full-range yuvj420p: exactly lossless for grey (verified by
    round trip in tests). RGB (N,H,W,3) -> H.264 crf 18 yuv420p (lossy; camera only)."""
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames.shape[1:3]
    if h % 2 or w % 2:
        raise ValueError("H.264 4:2:0 needs even frame sizes; crop or pad the frames first")
    grey = frames.ndim == 3
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=round(fps))
        s.width, s.height = w, h
        s.pix_fmt = "yuvj420p" if grey else "yuv420p"
        s.options = {"crf": "0" if grey else "18"}
        for f in frames:
            fr = av.VideoFrame.from_ndarray(f, format="gray" if grey else "rgb24")
            for p in s.encode(fr):
                c.mux(p)
        for p in s.encode():
            c.mux(p)


def _stats(x: NDArray[np.float32]) -> dict[str, list[float]]:
    return {"mean": x.mean(0).tolist(), "std": x.std(0).tolist(), "min": x.min(0).tolist(),
            "max": x.max(0).tolist(), "q01": np.quantile(x, 0.01, axis=0).tolist(),
            "q99": np.quantile(x, 0.99, axis=0).tolist()}


def export_gr00t_dataset(episodes: list[Episode], out: Path, spec: Gr00tArmSpec,
                         v_hold_mps: float | None = None) -> Path:
    """Write GR00T-flavored LeRobot v2 (data/, videos/, meta/ incl. modality.json, stats.json).
    Our LeRobot v3.0 dataset stays the lossless archive; this is the training copy."""
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    fps = episodes[0].fps
    tasks = list(MODE_INSTRUCTIONS.values()) if spec.language_mode else [TASK]
    all_s, all_a, all_t, gidx = [], [], [], 0
    for ep in episodes:
        if ep.bmode is None:
            raise ValueError("GR00T export needs B-mode frames")
        s, a = state_vector(spec, ep, v_hold_mps), action_vector(ep)
        task_idx = (np.array([tasks.index(MODE_INSTRUCTIONS[m]) for m in ep.contact_cmd])
                    if spec.language_mode else np.zeros(len(ep), dtype=np.int64))
        n = len(ep)
        df = pd.DataFrame({
            "observation.state": list(s), "action": list(a),
            "timestamp": (ep.time - ep.time[0]).astype(np.float32),
            "frame_index": np.arange(n), "episode_index": ep.episode_index,
            "index": gidx + np.arange(n), "task_index": task_idx,
            LANG_KEY: task_idx, "next.reward": np.zeros(n, np.float32),
            "next.done": np.arange(n) == n - 1})
        gidx += n
        chunk = ep.episode_index // 1000
        p = out / f"data/chunk-{chunk:03d}/episode_{ep.episode_index:06d}.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p, index=False)
        write_mp4(ep.bmode, out / f"videos/chunk-{chunk:03d}/observation.images.bmode/"
                  f"episode_{ep.episode_index:06d}.mp4", fps)
        if spec.use_camera:
            if ep.camera is None:
                raise ValueError("arm uses the camera but the episode has none")
            write_mp4(ep.camera, out / f"videos/chunk-{chunk:03d}/observation.images.camera/"
                      f"episode_{ep.episode_index:06d}.mp4", fps)
        all_s.append(s)
        all_a.append(a)
        all_t.append(df["timestamp"].to_numpy()[:, None])
    meta = out / "meta"
    meta.mkdir(parents=True)
    h, w = episodes[0].bmode.shape[1:]  # type: ignore[union-attr]
    vid = {"dtype": "video", "shape": [h, w, 3], "names": ["height", "width", "channels"],
           "info": {"video.height": h, "video.width": w, "video.codec": "h264",
                    "video.pix_fmt": "yuvj420p", "video.fps": fps, "video.channels": 3,
                    "has_audio": False}}
    feats: dict[str, Any] = {
        "observation.state": {"dtype": "float32", "shape": [spec.state_dim],
                              "names": [k for k, d in spec.state_keys for _ in range(d)]},
        "action": {"dtype": "float32", "shape": [6 + len(MODES)], "names": None},
        "observation.images.bmode": vid, "timestamp": {"dtype": "float32", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]}, "episode_index": {"dtype": "int64", "shape": [1]},
        "index": {"dtype": "int64", "shape": [1]}, "task_index": {"dtype": "int64", "shape": [1]}}
    if spec.use_camera:
        ch, cw = episodes[0].camera.shape[1:3]  # type: ignore[union-attr]
        feats["observation.images.camera"] = {**vid, "shape": [ch, cw, 3]}
    info = {"codebase_version": "v2.1", "robot_type": "ur7e", "total_episodes": len(episodes),
            "total_frames": gidx, "total_tasks": len(tasks), "chunks_size": 1000, "fps": fps,
            "splits": {"train": f"0:{len(episodes)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": feats}
    (meta / "info.json").write_text(json.dumps(info, indent=2))
    (meta / "modality.json").write_text(json.dumps(modality_json(spec), indent=2))
    (meta / "tasks.jsonl").write_text("".join(json.dumps({"task_index": i, "task": t}) + "\n"
                                              for i, t in enumerate(tasks)))
    (meta / "episodes.jsonl").write_text("".join(
        json.dumps({"episode_index": e.episode_index,
                    "tasks": sorted({MODE_INSTRUCTIONS[m] for m in e.contact_cmd})
                    if spec.language_mode else [TASK], "length": len(e)}) + "\n"
        for e in episodes))
    (meta / "stats.json").write_text(json.dumps({
        "observation.state": _stats(np.concatenate(all_s)), "action": _stats(np.concatenate(all_a)),
        "timestamp": _stats(np.concatenate(all_t))}, indent=2))
    (out / "modality_config.py").write_text(modality_config_source(spec))
    return out


def finetune_command(spec: Gr00tArmSpec, dataset: Path, base_model: str, out_dir: Path, seed: int,
                     steps: int, batch_size: int, lr: float, gr00t_root: Path) -> list[str]:
    """Command line for scripts/gr00t_finetune_seeded.py (GR00T's launch_finetune + seed).
    Flags from gr00t/configs/finetune_config.py; frozen VL backbone (tune_llm/visual False),
    action head + projectors tuned (these are launch_finetune's defaults, so they are not
    passed); state dropout 0 (see module docstring)."""
    return ["python", "scripts/gr00t_finetune_seeded.py", "--gr00t-root", str(gr00t_root),
            "--seed", str(seed), "--",
            "--base-model-path", base_model, "--dataset-path", str(dataset),
            "--embodiment-tag", EMBODIMENT_TAG,
            "--modality-config-path", str(Path(dataset) / "modality_config.py"),
            "--state-dropout-prob", "0.0", "--global-batch-size", str(batch_size),
            "--learning-rate", str(lr), "--max-steps", str(steps), "--output-dir", str(out_dir),
            "--save-steps", str(steps)]


def gr00t_passthrough(episodes: list[Episode], n_steps: int, tol: float = 1e-6,
                      max_starts: int = 2000, seed: int = 0) -> dict[str, float]:
    """CLAUDE.md rule 4 for the GR00T path: the exported per-frame action rows t..t+n-1 (exactly
    what GR00T trains on and predicts) decoded by :func:`decode_action` must reproduce the chunk
    computed from recorded poses, and the contact head must reproduce the operator mode."""
    from usintent.data.chunks import chunks_from_poses
    from usintent.data.passthrough import PassthroughError

    rng = np.random.default_rng(seed)
    worst, bad_mode, n = 0.0, 0, 0
    for ep in episodes:
        a = action_vector(ep)
        starts = np.arange(0, len(ep) - n_steps - 1)
        if len(starts) > max_starts // max(1, len(episodes)):
            starts = np.sort(rng.choice(starts, max_starts // max(1, len(episodes)), False))
        ref = policy_target(chunks_from_poses(ep.state[:, :6], starts, n_steps))
        for i, s in enumerate(starts):
            act = {"tool_step": a[None, s:s + n_steps, :6], "contact_cmd": a[None, s:s + n_steps, 6:]}
            chunk, m = decode_action(act)
            worst = max(worst, float(np.max(np.abs(chunk - ref[i]))))
            bad_mode += int(m != ep.contact_cmd[s + 1])
            n += 1
    out = {"max_err": worst, "mode_mismatches": float(bad_mode), "n_chunks": float(n)}
    if worst > tol or bad_mode:
        raise PassthroughError(f"GR00T passthrough failed: {out}")
    return out


# ------------------------------------------------------------------------------- inference


def decode_action(action: dict[str, Any]) -> tuple[NDArray[np.float64], str]:
    """GR00T action dict (keys 'tool_step', 'contact_cmd', optionally 'action.'-prefixed; values
    (1, n, d)) -> our (n, 5) chunk and m_hat."""
    def get(k: str) -> NDArray[np.float64]:
        v = action[k] if k in action else action[f"action.{k}"]
        arr: NDArray[np.float64] = np.asarray(v, dtype=np.float64)[0]
        return arr

    chunk = policy_target(compose_actions(get("tool_step")))
    return chunk, MODES[int(np.argmax(get("contact_cmd")[0]))]


class GR00TPolicy(Policy):
    """Fine-tuned GR00T-H arm behind our Policy interface (intent gating from the base class).

    Requires Isaac-GR00T installed in the environment (lab GPU machine). Sampler controls:
    ``deterministic_noise`` seeds torch before every call; ``integration_steps`` sets the action
    head's ``num_inference_timesteps`` (gr00t_n1d7.py).
    """

    def __init__(self, arm_id: str, spec: Gr00tArmSpec, checkpoint: str, device: str = "cuda:0",
                 seed: int = 0, deterministic_noise: bool = True,
                 integration_steps: int | None = None, v_hold_mps: float | None = None,
                 instruction: str = TASK):
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        self.arm_id = arm_id
        self.spec = spec
        self.inner = Gr00tPolicy(EMBODIMENT_TAG, checkpoint, device=device)
        self.seed, self.deterministic_noise = seed, deterministic_noise
        self.v_hold_mps, self.instruction = v_hold_mps, instruction
        if integration_steps is not None:
            hits = [m for m in self.inner.model.modules() if hasattr(m, "num_inference_timesteps")]
            if not hits:
                raise RuntimeError("no module with num_inference_timesteps in this GR00T build")
            for m in hits:
                m.num_inference_timesteps = integration_steps

    def observation(self, history: dict[str, Any], mode: str | None) -> dict[str, Any]:
        k = len(self.spec.video_delta)
        b = history["bmode"][-k:]
        video = {"bmode": np.repeat(b[..., None], 3, axis=-1)[None].astype(np.uint8)}
        if self.spec.use_camera:
            video["camera"] = history["camera"][-k:][None].astype(np.uint8)
        cols = []
        for name, _ in self.spec.state_keys:
            if name == "tcp_pose":
                cols.append(history["state"][-1, :6])
            elif name == "time_since_motion":
                e = float(history["e_proprio"][-1])
                cols.append(np.array([-1.0 if np.isnan(e) else e]))
            elif name == "mode":
                if mode is None:
                    raise ValueError(f"{self.arm_id} needs a mode input")
                cols.append(_onehot(np.array([mode]))[0])
            elif name == "bias":
                cols.append(np.zeros(1))
        state = np.concatenate(cols).astype(np.float32)
        offsets = np.cumsum([0] + [d for _, d in self.spec.state_keys])
        states = {n: state[None, None, offsets[i]:offsets[i + 1]]
                  for i, (n, _) in enumerate(self.spec.state_keys)}
        return {"video": video, "state": states, "language": {LANG_KEY: [[self.instruction]]}}

    def _predict(self, history: Any, mode: str | None) -> tuple[NDArray[np.float64], str]:
        import torch

        if self.deterministic_noise:
            torch.manual_seed(self.seed)
        action, _ = self.inner.get_action(self.observation(history, mode))
        return decode_action(action)
