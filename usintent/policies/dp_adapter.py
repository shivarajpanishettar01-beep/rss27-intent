"""LeRobot Diffusion Policy adapter. SPEC section 10. M6.

API read from the installed lerobot==0.6.1 source (lerobot/policies/diffusion/):
- ``DiffusionConfig(input_features, output_features, n_obs_steps, horizon, n_action_steps,
  down_dims, pretrained_backbone_weights, use_group_norm, num_inference_steps, ...)``;
  ``horizon % 2**len(down_dims) == 0`` is enforced in ``__post_init__``.
- ``DiffusionPolicy(config).forward(batch) -> (loss, None)`` with batch keys
  ``observation.state`` (B, n_obs, d), image keys (B, n_obs, 3, H, W),
  ``observation.environment_state`` (B, n_obs, d_env), ``action`` (B, horizon, 5),
  ``action_is_pad`` (B, horizon).
- ``policy.diffusion.generate_actions(batch, noise)`` returns actions
  ``[n_obs_steps - 1 : n_obs_steps - 1 + n_action_steps]`` of the predicted horizon.
- Global conditioning (``DiffusionModel._prepare_global_conditioning``) is the concatenation of
  the state, the image features and ``observation.environment_state``, fed to the 1-D UNet's
  FiLM layers. **DP-5's scheduler mode enters exactly there**, as a one-hot
  ``observation.environment_state`` (no change to LeRobot). Blind arms have no environment state.

Contact-command head (paper Sec. III/IV: lift-off is the policy's decision): the DP action is
9-D per step = our 5-D displacement + a one-hot of the operator's contact command
(HOLD, SWEEP, LIFT, RECONTACT) at that future step; m_hat = argmax of the one-hot part of the
first predicted step.

Our conventions: target = our 5-D chunk (n steps after the current observation), placed at
horizon positions n_obs-1 .. n_obs-2+n; earlier and trailing positions are padding
(``action_is_pad``). Actions are min-max normalised to [-1, 1] (DP clips samples to that range).
Arms without proprioception (DP-1, as A1) get a constant 1-D zero state, because LeRobot's DP
always reads ``observation.state``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from usintent.data.chunks import ChunkSampler, policy_target
from usintent.data.loaders import MODES
from usintent.policies.base import Policy

IMG_KEY = "observation.images.bmode"
CAM_KEY = "observation.images.camera"
ACT_DIM = 5 + len(MODES)          # displacement + contact-command one-hot


def contact_onehot(cmds: Any) -> NDArray[np.float64]:
    """(..., ) array of mode strings -> (..., 4) one-hot over MODES."""
    arr = np.asarray(cmds)
    out = np.zeros((*arr.shape, len(MODES)))
    for i, m in enumerate(MODES):
        out[..., i] = arr == m
    return out


@dataclass(frozen=True)
class MinMax:
    lo: NDArray[np.float64]
    hi: NDArray[np.float64]

    @classmethod
    def fit(cls, x: NDArray[np.float64]) -> MinMax:
        flat = x.reshape(-1, x.shape[-1])
        lo, hi = flat.min(axis=0), flat.max(axis=0)
        return cls(lo, np.where(hi - lo < 1e-9, lo + 1e-9, hi))

    def encode(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(2.0 * (x - self.lo) / (self.hi - self.lo) - 1.0)

    def decode(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray((z + 1.0) / 2.0 * (self.hi - self.lo) + self.lo)


@dataclass(frozen=True)
class DPArmSpec:
    arm_id: str                 # "DP-1" | "DP-5"
    n_obs_steps: int            # 1 for DP-1; history k for DP-5
    use_proprio: bool
    use_mode: bool
    n_steps: int                # chunk length (H * fps)
    image_shape: tuple[int, int]
    down_dims: tuple[int, ...] = (64, 128)
    num_inference_steps: int = 16
    noise_scheduler_type: str = "DDIM"
    camera_shape: tuple[int, int] | None = None   # probe camera (rows, cols) if the arm uses it

    @property
    def horizon(self) -> int:
        need = self.n_obs_steps - 1 + self.n_steps
        f = 2 ** len(self.down_dims)
        return int(np.ceil(need / f) * f)


def arm_spec_from_config(arm_cfg: dict[str, Any], n_steps: int, image_shape: tuple[int, int],
                         history_k: int, camera_shape: tuple[int, int] | None = None,
                         **overrides: Any) -> DPArmSpec:
    """Translate configs/arms/DP-*.yaml into a DPArmSpec.

    ``camera_shape``: (rows, cols) of the probe camera, or None when the rig has no camera
    (rig.yaml topics.camera == "none"). The camera is used only if the arm lists it.
    """
    if arm_cfg["family"] != "diffusion_policy":
        raise ValueError(f"{arm_cfg['arm_id']} is not a diffusion-policy arm")
    inputs = set(arm_cfg["inputs"])
    use_mode = "mode_embedding" in inputs
    if use_mode != bool(arm_cfg["intent_aware"]):
        raise ValueError(f"{arm_cfg['arm_id']}: mode input must match intent_aware")
    return DPArmSpec(arm_id=str(arm_cfg["arm_id"]),
                     n_obs_steps=history_k if "bmode_history" in inputs else 1,
                     use_proprio="proprio" in inputs, use_mode=use_mode, n_steps=n_steps,
                     image_shape=image_shape,
                     camera_shape=camera_shape if "camera" in inputs else None, **overrides)


def make_policy(spec: DPArmSpec, seed: int) -> Any:
    import torch
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

    torch.manual_seed(seed)
    h, w = spec.image_shape
    inputs = {"observation.state": PolicyFeature(FeatureType.STATE, (12 if spec.use_proprio else 1,)),
              IMG_KEY: PolicyFeature(FeatureType.VISUAL, (3, h, w))}
    if spec.camera_shape is not None:
        ch, cw = spec.camera_shape
        inputs[CAM_KEY] = PolicyFeature(FeatureType.VISUAL, (3, ch, cw))
    if spec.use_mode:
        inputs["observation.environment_state"] = PolicyFeature(FeatureType.ENV, (len(MODES),))
    cfg = DiffusionConfig(
        input_features=inputs, output_features={"action": PolicyFeature(FeatureType.ACTION, (ACT_DIM,))},
        n_obs_steps=spec.n_obs_steps, horizon=spec.horizon, n_action_steps=spec.n_steps,
        down_dims=spec.down_dims, pretrained_backbone_weights=None, use_group_norm=True,
        crop_shape=None, num_inference_steps=spec.num_inference_steps,
        noise_scheduler_type=spec.noise_scheduler_type, device="cpu")
    return DiffusionPolicy(cfg)


def _obs_tensors(spec: DPArmSpec, hists: list[dict[str, Any]], modes: list[str] | None) -> dict[str, Any]:
    import torch

    k = spec.n_obs_steps
    img = np.stack([h["bmode"][-k:] for h in hists]).astype(np.float32) / 255.0      # B,k,H,W
    batch = {IMG_KEY: torch.as_tensor(img)[:, :, None].repeat(1, 1, 3, 1, 1)}
    if spec.camera_shape is not None:
        cam = np.stack([h["camera"][-k:] for h in hists]).astype(np.float32) / 255.0  # B,k,H,W,3
        batch[CAM_KEY] = torch.as_tensor(cam).permute(0, 1, 4, 2, 3)
    if spec.use_proprio:
        st = np.stack([h["state"][-k:] for h in hists]).astype(np.float32)
    else:
        st = np.zeros((len(hists), k, 1), np.float32)
    batch["observation.state"] = torch.as_tensor(st)
    if spec.use_mode:
        assert modes is not None
        onehot = np.zeros((len(hists), k, len(MODES)), np.float32)
        for i, m in enumerate(modes):
            onehot[i, :, MODES.index(m)] = 1.0
        batch["observation.environment_state"] = torch.as_tensor(onehot)
    return batch


class DPPolicy(Policy):
    """Trained (or untrained) LeRobot DP behind our Policy interface with intent gating."""

    def __init__(self, spec: DPArmSpec, seed: int = 0, norm: MinMax | None = None):
        self.arm_id = spec.arm_id
        self.spec = spec
        self.seed = seed
        self.model = make_policy(spec, seed)
        self.norm = norm
        self.deterministic_noise = True

    # ------------------------------------------------------------------ training
    def batch(self, sampler: ChunkSampler, rows: pd.DataFrame) -> dict[str, Any]:
        import torch

        hists = [sampler.history(int(e), int(s)) for e, s in zip(rows["episode_index"], rows["start"])]
        modes = None
        if self.spec.use_mode:
            eps = {e.episode_index: e for e in sampler.episodes}
            modes = [str(eps[int(e)].contact_cmd[int(s)])   # intent = operator key-event mode
                     for e, s in zip(rows["episode_index"], rows["start"])]
        batch = _obs_tensors(self.spec, hists, modes)
        target = self.targets(sampler, rows)
        assert self.norm is not None
        z = self.norm.encode(target).astype(np.float32)
        hz = self.spec.horizon
        act = np.zeros((len(rows), hz, ACT_DIM), np.float32)
        pad = np.ones((len(rows), hz), bool)
        a0 = self.spec.n_obs_steps - 1
        act[:, a0:a0 + self.spec.n_steps] = z
        pad[:, a0:a0 + self.spec.n_steps] = False
        batch["action"] = torch.as_tensor(act)
        batch["action_is_pad"] = torch.as_tensor(pad)
        return batch

    def targets(self, sampler: ChunkSampler, rows: pd.DataFrame) -> NDArray[np.float64]:
        """(m, n, 9) training targets: 5-D chunk + one-hot operator contact command at the
        chunk's steps (start+1 .. start+n)."""
        disp = policy_target(sampler.chunks6(rows))
        eps = {e.episode_index: e for e in sampler.episodes}
        n = self.spec.n_steps
        cmds = np.stack([eps[int(e)].contact_cmd[int(s) + 1:int(s) + 1 + n]
                         for e, s in zip(rows["episode_index"], rows["start"])])
        return np.concatenate([disp, contact_onehot(cmds)], axis=-1)

    def train_steps(self, sampler: ChunkSampler, n_steps: int, batch_size: int, lr: float,
                    rng: np.random.Generator, weights: NDArray[np.float64] | None = None) -> list[float]:
        """Plain Adam training loop; chunk starts drawn by ``weights`` (intervention) or uniformly."""
        import torch

        if sampler.history_k < self.spec.n_obs_steps:
            raise ValueError("sampler history_k smaller than the arm's n_obs_steps")
        if self.norm is None:
            rows = sampler.index.iloc[rng.choice(len(sampler.index), min(2000, len(sampler.index)), False)]
            self.norm = MinMax.fit(self.targets(sampler, rows))
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        losses = []
        for _ in range(n_steps):
            rows = sampler.sample(batch_size, rng, weights)
            loss, _ = self.model.forward(self.batch(sampler, rows))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        return losses

    # ------------------------------------------------------------------ inference
    def _predict(self, history: Any, mode: str | None) -> tuple[NDArray[np.float64], str]:
        import torch

        if self.spec.use_mode and mode is None:
            raise ValueError(f"{self.arm_id} needs the scheduler mode")
        modes = [mode] if mode is not None and self.spec.use_mode else None
        batch = _obs_tensors(self.spec, [history], modes)
        # same stacking DiffusionPolicy.forward does before compute_loss (lerobot 0.6.1)
        keys = [IMG_KEY] + ([CAM_KEY] if self.spec.camera_shape is not None else [])
        batch["observation.images"] = torch.stack([batch[k] for k in keys], dim=-4)
        noise = None
        if self.deterministic_noise:
            g = torch.Generator().manual_seed(self.seed)
            noise = torch.randn((1, self.spec.horizon, ACT_DIM), generator=g)
        self.model.eval()
        with torch.no_grad():
            z = self.model.diffusion.generate_actions(batch, noise=noise)[0].numpy().astype(np.float64)
        assert self.norm is not None
        dec = self.norm.decode(z)
        chunk = np.asarray(dec[:, :5], dtype=np.float64)
        m_hat = MODES[int(np.argmax(dec[0, 5:]))]
        return chunk, m_hat
