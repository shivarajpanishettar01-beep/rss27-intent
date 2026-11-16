"""Run US-VLA's own code for the USVLA-1 arm (US-VLA environment, lab GPU).

Runs with the US-VLA checkout as working directory (its train config loads
``./checkpoints/usfm/USFM_latest.pth`` relative to it), inside its uv environment:

  cd <US-VLA> && uv run python <rss27-intent>/scripts/usvla_run.py build-dataset --export <dir>
  cd <US-VLA> && uv run python <rss27-intent>/scripts/usvla_run.py train --run-json <run.json>
  cd <US-VLA> && uv run python <rss27-intent>/scripts/usvla_run.py serve --run-json <run.json> --port 8000

The exact commands for each seed are written to ``run.json`` by scripts/run_train_arm.py.
This file imports only numpy, the LeRobot commit US-VLA pins (``lerobot.common.datasets``,
format v2.1) and openpi; it never imports usintent, whose LeRobot 0.6.1 cannot share the env.

build-dataset  LeRobot v2.1 dataset at ``$HF_LEROBOT_HOME/<repo_id>`` (where openpi's
               ``create_torch_dataset`` looks), then reads it back with
               ``delta_timestamps = [k / fps for k in range(action_horizon)]`` exactly as openpi
               does and requires the stored actions, states and B-mode frames (rule 4).
train          ``get_config("pi05_ur_usfm_fusion")`` with only these fields replaced: repo_id,
               exp_name, seed, num_train_steps, batch_size, model.action_horizon, the cosine
               schedule's peak_lr and decay_lr (their schedule is flat after warm-up, so both =
               our shared lr), save_interval = steps, keep_period = None, overwrite, wandb.
               Then the body of their scripts/compute_norm_stats.py main (it only takes a
               config *name*) and their scripts/train.py main.
serve          Same config, ``policy_config.create_trained_policy`` on the final checkpoint and
               openpi's WebsocketPolicyServer. Per request, ``__usintent_noise_seed`` fixes the
               flow-matching noise (``Policy.infer(obs, noise=...)``) and
               ``__usintent_num_steps`` sets ``num_steps``. Metadata names arm, seed and run.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import runpy
import sys
from pathlib import Path
from typing import Any

import numpy as np

NOISE_KEY, STEPS_KEY = "__usintent_noise_seed", "__usintent_num_steps"   # = usvla_adapter


# --------------------------------------------------------------------------------- dataset


def build_dataset(export: Path, check_starts: int = 200) -> dict[str, float]:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    man = json.loads((export / "manifest.json").read_text())
    repo, fps, n = man["repo_id"], int(man["fps"]), int(man["n_steps"])
    eps = [dict(np.load(export / f)) for f in man["episodes"]]
    us_hw = eps[0]["ultrasound"].shape[1:3]
    wr_hw = eps[0]["wrist"].shape[1:3] if man["has_wrist"] else tuple(us_hw)
    side_hw = tuple(man["blank_shape"])

    def img(hw: Any) -> dict[str, Any]:
        return {"dtype": "image", "shape": (int(hw[0]), int(hw[1]), 3),
                "names": ["height", "width", "channel"]}

    feats = {"base_rgb": img(side_hw), "wrist_rgb": img(wr_hw), "ultrasound_rgb": img(us_hw),
             "state": {"dtype": "float32", "shape": (6,), "names": None},
             "actions": {"dtype": "float32", "shape": (6,), "names": None}}
    from lerobot.common.constants import HF_LEROBOT_HOME

    if (HF_LEROBOT_HOME / repo).exists():            # built by an earlier seed: check only
        return check_dataset(repo, eps, fps, n, man["task"], check_starts)
    ds = LeRobotDataset.create(repo, fps=fps, robot_type="ur7e", features=feats,
                               use_videos=False, image_writer_threads=4)
    side = np.zeros((*side_hw, 3), np.uint8)
    for e in eps:
        for k in range(len(e["state"])):
            us = np.repeat(e["ultrasound"][k][:, :, None], 3, axis=2)
            wrist = e["wrist"][k] if man["has_wrist"] else np.zeros((*wr_hw, 3), np.uint8)
            ds.add_frame({"base_rgb": side, "wrist_rgb": wrist, "ultrasound_rgb": us,
                          "state": e["state"][k], "actions": e["actions"][k], "task": man["task"]})
        ds.save_episode()
    return check_dataset(repo, eps, fps, n, man["task"], check_starts)


def check_dataset(repo: str, eps: list[dict[str, Any]], fps: int, n: int, task: str,
                  check_starts: int) -> dict[str, float]:
    """Read back the way openpi's create_torch_dataset does; compare with the export."""
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo, delta_timestamps={"actions": [k / fps for k in range(n)]})
    rng = np.random.default_rng(0)
    offset, worst, bad, n_chk = 0, 0.0, 0, 0
    per_ep = max(1, check_starts // len(eps))
    for e in eps:
        T = len(e["state"])
        starts = np.arange(0, T - n)
        if len(starts) > per_ep:
            starts = np.sort(rng.choice(starts, per_ep, False))
        for s in starts:
            it = ds[int(offset + s)]
            if int(it["frame_index"]) != int(s) or bool(it["actions_is_pad"].any()) or it["task"] != task:
                bad += 1
            worst = max(worst, float(np.abs(it["actions"].numpy() - e["actions"][s:s + n]).max()),
                        float(np.abs(it["state"].numpy() - e["state"][s]).max()))
            us = (it["ultrasound_rgb"].numpy() * 255).round().astype(np.uint8)[0]
            bad += int(not np.array_equal(us, e["ultrasound"][s]))
            n_chk += 1
        offset += T
    out = {"max_err": worst, "mismatches": float(bad), "n_checked": float(n_chk)}
    if worst > 0 or bad:
        raise SystemExit(f"US-VLA dataset check failed: {out}")
    return out


# ---------------------------------------------------------------------------------- config


def build_config(p: dict[str, Any], checkpoint_base_dir: str | None = None) -> Any:
    import openpi.training.config as _config

    cfg = _config.get_config(p["config_name"])
    lr = float(p["lr"])
    fields: dict[str, Any] = {
        "exp_name": p["exp_name"], "seed": int(p["seed"]), "num_train_steps": int(p["steps"]),
        "batch_size": int(p["batch_size"]), "save_interval": int(p["steps"]), "keep_period": None,
        "overwrite": True, "wandb_enabled": bool(p["wandb"]),
        "model": dataclasses.replace(cfg.model, action_horizon=int(p["action_horizon"])),
        "data": dataclasses.replace(cfg.data, repo_id=p["repo_id"]),
        "lr_schedule": dataclasses.replace(cfg.lr_schedule, peak_lr=lr, decay_lr=lr)}
    if checkpoint_base_dir:
        fields["checkpoint_base_dir"] = checkpoint_base_dir
    return dataclasses.replace(cfg, **fields)


def norm_stats(cfg: Any, root: Path) -> Path:
    """Body of US-VLA's scripts/compute_norm_stats.py main() for a config object."""
    from openpi.shared import normalize

    mod = runpy.run_path(str(root / "scripts" / "compute_norm_stats.py"))
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    loader, _ = mod["create_torch_dataloader"](
        data_config, cfg.model.action_horizon, cfg.batch_size, cfg.model, cfg.num_workers, None)
    stats = {k: normalize.RunningStats() for k in ("state", "actions")}
    for batch in loader:
        for k, st in stats.items():
            st.update(np.asarray(batch[k]))
    out = cfg.assets_dirs / data_config.repo_id
    normalize.save(out, {k: s.get_statistics() for k, s in stats.items()})
    return Path(out)


# ----------------------------------------------------------------------------------- serve


class SeededPolicy:
    """openpi Policy wrapper: per-request noise seed and flow-step count."""

    def __init__(self, policy: Any, horizon: int, action_dim: int, meta: dict[str, Any]):
        self._p, self._shape, self.metadata = policy, (horizon, action_dim), meta

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        seed, steps = obs.pop(NOISE_KEY, None), obs.pop(STEPS_KEY, None)
        if steps is not None:
            self._p._sample_kwargs["num_steps"] = int(steps)   # read by Policy.infer each call
        noise = (np.random.default_rng(int(seed)).standard_normal(self._shape).astype(np.float32)
                 if seed is not None else None)
        return dict(self._p.infer(obs, noise=noise))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-dataset")
    b.add_argument("--export", type=Path, required=True)
    for name in ("train", "serve"):
        s = sub.add_parser(name)
        s.add_argument("--run-json", type=Path, required=True)
        if name == "serve":
            s.add_argument("--port", type=int, required=True)
            s.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    root = Path.cwd()
    if args.cmd == "build-dataset":
        print(json.dumps(build_dataset(args.export)))
        return
    if not (root / "src" / "openpi").is_dir():
        raise SystemExit("run from the US-VLA checkout root (cd <US-VLA>)")
    run = json.loads(args.run_json.read_text())
    params = run["usvla_params"]
    cfg = build_config(params, str(args.run_json.parent / "openpi_checkpoints"))
    if args.cmd == "train":
        print("norm stats ->", norm_stats(cfg, root))
        sys.argv = [sys.argv[0]]
        runpy.run_path(str(root / "scripts" / "train.py"))["main"](cfg)
        return
    from openpi.policies import policy_config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    ckpt = cfg.checkpoint_dir / str(cfg.num_train_steps - 1)
    if not ckpt.is_dir():
        raise SystemExit(f"{ckpt}: no final checkpoint")
    policy = policy_config.create_trained_policy(cfg, ckpt)
    meta = {**dict(policy.metadata or {}),
            "usintent": {"arm_id": params["arm_id"], "seed": params["seed"],
                         "run": str(args.run_json.parent), "checkpoint": str(ckpt)}}
    srv = SeededPolicy(policy, cfg.model.action_horizon, cfg.model.action_dim, meta)
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
    WebsocketPolicyServer(policy=srv, host=args.host, port=args.port, metadata=meta).serve_forever()


if __name__ == "__main__":
    main()
