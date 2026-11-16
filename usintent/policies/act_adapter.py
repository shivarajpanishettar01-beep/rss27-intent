"""LeRobot ACT (Action Chunking with Transformers) adapter: an external comparison family.

API read from the installed lerobot==0.6.1 source (lerobot/policies/act/):
- ``ACTConfig(input_features, output_features, n_obs_steps=1, chunk_size, n_action_steps,
  vision_backbone="resnet18", pretrained_backbone_weights, dim_model, n_heads, dim_feedforward,
  n_encoder_layers, n_decoder_layers, use_vae=True, latent_dim, kl_weight, ...)``.
  ``__post_init__`` raises unless ``n_obs_steps == 1`` ("Multiple observation steps not handled
  yet"); ``validate_features`` needs at least one image or ``observation.environment_state``.
- ``ACTPolicy(config).forward(batch) -> (loss, loss_dict)`` with batch keys
  ``observation.state`` (B, d), image keys (B, 3, H, W), optional
  ``observation.environment_state`` (B, d_env), ``action`` (B, chunk_size, d_act),
  ``action_is_pad`` (B, chunk_size). Loss = masked L1 + ``kl_weight`` * KL (VAE).
- ``ACTPolicy.predict_action_chunk(batch) -> (B, chunk_size, d_act)``. At inference the VAE
  encoder is skipped and the latent is all zeros, so ACT inference is **deterministic**: there is
  no sampler noise to control (unlike DP and GR00T).
- ``ACT.forward`` reads ``batch["observation.state"].device`` even when the config has no state
  feature, so a state tensor is always passed (as in our DP adapter).
- The scheduler mode of ACT-5 enters as ``observation.environment_state`` (one-hot), which ACT
  projects to one encoder token (``encoder_env_state_input_proj``). No change to LeRobot.

Our conventions are identical to usintent.policies.dp_adapter: 9-D action per step (5-D
tool-frame displacement + one-hot operator contact command), min-max normalised to [-1, 1];
m_hat = argmax of the one-hot part of the first predicted step; arms without proprioception get a
constant 1-D zero state. Because ACT takes one observation step, ACT-5 has no B-mode history
(reports/BASELINES.md, Decisions).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from usintent.data.chunks import ChunkSampler
from usintent.data.loaders import MODES
from usintent.policies.base import Policy
from usintent.policies.dp_adapter import ACT_DIM, CAM_KEY, IMG_KEY, DPPolicy, MinMax


@dataclass(frozen=True)
class ACTArmSpec:
    arm_id: str                 # "ACT-1" | "ACT-5"
    use_proprio: bool
    use_mode: bool
    n_steps: int                # chunk length (H * fps) = ACT chunk_size = n_action_steps
    image_shape: tuple[int, int]
    camera_shape: tuple[int, int] | None = None
    pretrained_backbone_weights: str | None = None   # None = from scratch, as our DP arms
    dim_model: int = 512                              # ACTConfig defaults (lerobot 0.6.1)
    n_heads: int = 8
    dim_feedforward: int = 3200
    n_encoder_layers: int = 4
    n_decoder_layers: int = 1
    latent_dim: int = 32
    kl_weight: float = 10.0
    n_obs_steps: int = 1        # fixed by ACT; kept so the shared sampler check reads one field


def arm_spec_from_config(arm_cfg: dict[str, Any], n_steps: int, image_shape: tuple[int, int],
                         camera_shape: tuple[int, int] | None = None,
                         **overrides: Any) -> ACTArmSpec:
    """Translate configs/arms/ACT-*.yaml into an ACTArmSpec. ``overrides`` set model sizes
    (tests use small ones); the arm's ``pretrained_backbone`` key is passed through."""
    if arm_cfg["family"] != "act":
        raise ValueError(f"{arm_cfg['arm_id']} is not an ACT arm")
    inputs = set(arm_cfg["inputs"])
    if "bmode_history" in inputs:
        raise ValueError(f"{arm_cfg['arm_id']}: ACT takes one observation step (no history)")
    use_mode = "mode_embedding" in inputs
    if use_mode != bool(arm_cfg["intent_aware"]):
        raise ValueError(f"{arm_cfg['arm_id']}: mode input must match intent_aware")
    overrides.setdefault("pretrained_backbone_weights", arm_cfg.get("pretrained_backbone"))
    return ACTArmSpec(arm_id=str(arm_cfg["arm_id"]), use_proprio="proprio" in inputs,
                      use_mode=use_mode, n_steps=n_steps, image_shape=image_shape,
                      camera_shape=camera_shape if "camera" in inputs else None, **overrides)


def make_policy(spec: ACTArmSpec, seed: int) -> Any:
    import torch
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy

    torch.manual_seed(seed)
    h, w = spec.image_shape
    inputs = {"observation.state": PolicyFeature(FeatureType.STATE, (12 if spec.use_proprio else 1,)),
              IMG_KEY: PolicyFeature(FeatureType.VISUAL, (3, h, w))}
    if spec.camera_shape is not None:
        ch, cw = spec.camera_shape
        inputs[CAM_KEY] = PolicyFeature(FeatureType.VISUAL, (3, ch, cw))
    if spec.use_mode:
        inputs["observation.environment_state"] = PolicyFeature(FeatureType.ENV, (len(MODES),))
    cfg = ACTConfig(
        input_features=inputs, output_features={"action": PolicyFeature(FeatureType.ACTION, (ACT_DIM,))},
        n_obs_steps=1, chunk_size=spec.n_steps, n_action_steps=spec.n_steps,
        pretrained_backbone_weights=spec.pretrained_backbone_weights, dim_model=spec.dim_model,
        n_heads=spec.n_heads, dim_feedforward=spec.dim_feedforward,
        n_encoder_layers=spec.n_encoder_layers, n_decoder_layers=spec.n_decoder_layers,
        latent_dim=spec.latent_dim, kl_weight=spec.kl_weight, device="cpu")
    return ACTPolicy(cfg)


def _obs_tensors(spec: ACTArmSpec, hists: list[dict[str, Any]], modes: list[str] | None) -> dict[str, Any]:
    """Latest frame only: images (B, 3, H, W) in [0, 1], state (B, d), env state (B, 4)."""
    import torch

    img = np.stack([h["bmode"][-1] for h in hists]).astype(np.float32) / 255.0        # B,H,W
    batch = {IMG_KEY: torch.as_tensor(img)[:, None].repeat(1, 3, 1, 1)}
    if spec.camera_shape is not None:
        cam = np.stack([h["camera"][-1] for h in hists]).astype(np.float32) / 255.0   # B,H,W,3
        batch[CAM_KEY] = torch.as_tensor(cam).permute(0, 3, 1, 2)
    if spec.use_proprio:
        st = np.stack([h["state"][-1] for h in hists]).astype(np.float32)
    else:
        st = np.zeros((len(hists), 1), np.float32)
    batch["observation.state"] = torch.as_tensor(st)
    if spec.use_mode:
        assert modes is not None
        onehot = np.zeros((len(hists), len(MODES)), np.float32)
        for i, m in enumerate(modes):
            onehot[i, MODES.index(m)] = 1.0
        batch["observation.environment_state"] = torch.as_tensor(onehot)
    return batch


class ACTPolicy(Policy):
    """Trained (or untrained) LeRobot ACT behind our Policy interface with intent gating.

    Training targets, normalisation and the training loop are the DP arm's (shared code path),
    so ACT and DP differ only in the model. ``deterministic_noise`` is accepted for the shared
    evaluation grid but has no effect: ACT inference uses a zero latent.
    """

    def __init__(self, spec: ACTArmSpec, seed: int = 0, norm: MinMax | None = None):
        self.arm_id = spec.arm_id
        self.spec = spec
        self.seed = seed
        self.model = make_policy(spec, seed)
        self.norm = norm
        self.deterministic_noise = True

    def targets(self, sampler: ChunkSampler, rows: pd.DataFrame) -> NDArray[np.float64]:
        """(m, n, 9) targets, identical to DPPolicy.targets (uses only ``spec.n_steps``)."""
        return DPPolicy.targets(self, sampler, rows)  # type: ignore[arg-type]

    def batch(self, sampler: ChunkSampler, rows: pd.DataFrame) -> dict[str, Any]:
        import torch

        hists = [sampler.history(int(e), int(s)) for e, s in zip(rows["episode_index"], rows["start"])]
        modes = None
        if self.spec.use_mode:
            eps = {e.episode_index: e for e in sampler.episodes}
            modes = [str(eps[int(e)].contact_cmd[int(s)])   # intent = operator key-event mode
                     for e, s in zip(rows["episode_index"], rows["start"])]
        batch = _obs_tensors(self.spec, hists, modes)
        assert self.norm is not None
        z = self.norm.encode(self.targets(sampler, rows)).astype(np.float32)   # (B, n, 9)
        batch["action"] = torch.as_tensor(z)
        batch["action_is_pad"] = torch.zeros(z.shape[:2], dtype=torch.bool)
        return batch

    def train_steps(self, sampler: ChunkSampler, n_steps: int, batch_size: int, lr: float,
                    rng: np.random.Generator, weights: NDArray[np.float64] | None = None) -> list[float]:
        """Same plain Adam loop as DPPolicy.train_steps (shared schedule)."""
        return DPPolicy.train_steps(self, sampler, n_steps, batch_size, lr, rng, weights)  # type: ignore[arg-type]

    def _predict(self, history: Any, mode: str | None) -> tuple[NDArray[np.float64], str]:
        import torch

        if self.spec.use_mode and mode is None:
            raise ValueError(f"{self.arm_id} needs the scheduler mode")
        modes = [mode] if mode is not None and self.spec.use_mode else None
        batch = _obs_tensors(self.spec, [history], modes)
        with torch.no_grad():
            z = self.model.predict_action_chunk(batch)[0].numpy().astype(np.float64)
        assert self.norm is not None
        dec = self.norm.decode(z)
        chunk = np.asarray(dec[:, :5], dtype=np.float64)
        m_hat = MODES[int(np.argmax(dec[0, 5:]))]
        return chunk, m_hat
