"""Train one arm x seed (M6). Logs seed, git hash, config hash and data hash.

  Lab:   python scripts/run_train_arm.py --arm DP-1 --seed 0 --train $USINTENT_DATA/cued/<train> \
             [--target-departure-fraction 0.08]
  Dry run (SYNTHETIC, cloud): python scripts/run_train_arm.py --arm DP-5 --seed 0 --dry-run
  GR00T arm (lab GPU): ... --arm A5 --seed 0 --train <root> --gr00t-root <Isaac-GR00T> --launch
  ACT arm: same as DP (--arm ACT-1 / ACT-5), trained in the shared loop.
  LeRobot VLA arm (lab GPU): ... --arm SmolVLA-1 --seed 0 --train <root> --launch
  US-VLA arm (lab GPU): ... --arm USVLA-1 --seed 0 --train <root> --usvla-root <US-VLA> --launch
Schedule (steps, batch size, lr) comes from configs/arms/<arm>.yaml and is shared by all arms;
nulls refuse to run except in --dry-run (10 steps, batch 4). GR00T arms export a GR00T LeRobot-v2
copy and run Isaac-GR00T's launch_finetune with the seed (scripts/gr00t_finetune_seeded.py).
SmolVLA / pi0.5 arms export a LeRobot v3.0 training copy, pass it through LeRobot's own loader
(rule 4) and run ``lerobot-train`` from the released base checkpoint with the shared schedule.
Outputs: $USINTENT_RESULTS/train/<arm>[_p<target>]/seed<k>/{model.pt, run.json}; dry runs write
to a temporary directory only.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from usintent.config import load, require
from usintent.data.chunks import ChunkSampler
from usintent.intervene.sampler import epoch_indices, intervention_weights, realized_fraction
from usintent.paths import RESULTS
from usintent.provenance import data_hash, stamp


def train_gr00t(args: argparse.Namespace, arm: dict) -> None:
    """GR00T-H arm: export the GR00T LeRobot-v2 training copy (once per arm and data hash),
    check the GR00T passthrough, write the seeded fine-tune command, optionally launch it."""
    import subprocess

    from usintent.data.loaders import load_dataset
    from usintent.policies.gr00t_adapter import (
        arm_spec_from_config as gr00t_spec,
    )
    from usintent.policies.gr00t_adapter import (
        export_gr00t_dataset,
        finetune_command,
        gr00t_passthrough,
    )
    from usintent.protocol.recorder import camera_topic

    if args.target_departure_fraction is not None:
        raise SystemExit("the training-data intervention is implemented for DP arms only: GR00T's "
                         "sharded episode sampler cannot take per-chunk weights (reports/REVIEW_FIXES.md)")
    data = load("data")
    k = int(data["history_frames_k"])
    if args.dry_run:
        from usintent.sim.holdworld import HoldWorldConfig, generate

        eps = generate(HoldWorldConfig(image_shape=(32, 32), fps=10.0, n_stations=(2, 3)), 3)
        steps, bs, lr, dhash, source, cam = 10, 4, 1e-4, "synthetic", "synthetic", False
        base = "<base checkpoint>"
        root = Path(tempfile.mkdtemp(prefix="usintent_gr00t_dry_"))
        v_hold = 0.002
    else:
        eps = load_dataset(args.train)
        steps = int(require(arm, "schedule.steps"))
        bs = int(require(arm, "schedule.batch_size"))
        lr = float(require(arm, "schedule.lr"))
        base = str(require(arm, "checkpoint"))
        dhash, source = data_hash(args.train), "cued"
        cam = camera_topic(load("rig")) is not None
        root = RESULTS / "gr00t"
        v_hold = float(require(data, "segmentation.v_hold_mps"))
    n = round(float(data["chunk_horizon_s"]) * eps[0].fps)
    spec = gr00t_spec(arm, n, k, cam, language_mode=args.language_mode)
    tag = args.arm + ("_lang" if args.language_mode else "")
    export = root / "data" / f"{tag}_{dhash}"
    if not export.exists():
        export_gr00t_dataset(eps, export, spec, v_hold)
    pt = gr00t_passthrough(eps, n)
    out = root / "train" / tag / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "usintent_spec.json").write_text(json.dumps(spec.__dict__, indent=2))
    gr00t_root = args.gr00t_root or Path("<Isaac-GR00T checkout>")
    cmd = finetune_command(spec, export, base, out, args.seed, steps, bs, lr, gr00t_root)
    run = {"arm": args.arm, "seed": args.seed, "family": "gr00t_h", "steps": steps,
           "batch_size": bs, "lr": lr, "data_hash": dhash, "export": str(export),
           "gr00t_passthrough": pt, "command": cmd, "language_mode": args.language_mode,
           "provenance": stamp(["data", f"arms/{args.arm}"], source, seed=args.seed)}
    (out / "run.json").write_text(json.dumps(run, indent=2, default=float))
    print(json.dumps(run, indent=2, default=float))
    if args.launch:
        if args.dry_run or args.gr00t_root is None:
            raise SystemExit("--launch needs real data and --gr00t-root")
        raise SystemExit(subprocess.call(cmd))


def train_vla(args: argparse.Namespace, arm: dict) -> None:
    """SmolVLA / pi0.5 arm: export the LeRobot v3.0 training copy (once per arm and data hash),
    run the passthrough through LeRobot's loader, write the lerobot-train command, optionally
    launch it. The dry run stops after the passthrough and command (no weights downloaded)."""
    import subprocess

    from usintent.policies.lerobot_vla_adapter import (
        arm_spec_from_config as vla_spec,
    )
    from usintent.policies.lerobot_vla_adapter import (
        export_vla_dataset,
        train_command,
        vla_passthrough,
    )
    from usintent.protocol.recorder import camera_topic

    if args.target_departure_fraction is not None:
        raise SystemExit("the training-data intervention is implemented for DP arms only")
    data = load("data")
    if args.dry_run:
        from usintent.sim.holdworld import HoldWorldConfig, generate

        eps = generate(HoldWorldConfig(image_shape=(32, 32), fps=10.0, n_stations=(2, 3)), 3)
        steps, bs, lr, dhash, source, cam = 10, 4, 1e-4, "synthetic", "synthetic", False
        rename = arm.get("rename_map") or {"<our key>": "<checkpoint key>"}
        root = Path(tempfile.mkdtemp(prefix="usintent_vla_dry_"))
    else:
        from usintent.data.loaders import load_dataset

        eps = load_dataset(args.train)
        steps = int(require(arm, "schedule.steps"))
        bs = int(require(arm, "schedule.batch_size"))
        lr = float(require(arm, "schedule.lr"))
        rename = dict(require(arm, "rename_map"))
        dhash, source = data_hash(args.train), "cued"
        cam = camera_topic(load("rig")) is not None
        root = RESULTS / "lerobot_vla"
    base = str(require(arm, "checkpoint"))
    n = round(float(data["chunk_horizon_s"]) * eps[0].fps)
    spec = vla_spec(arm, n, cam)
    export = root / "data" / f"{args.arm}_{dhash}"
    if not export.exists():
        export_vla_dataset(eps, export, spec)
    pt = vla_passthrough(export, eps, n)
    out = root / "train" / args.arm / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "usintent_spec.json").write_text(json.dumps(spec.__dict__, indent=2))
    cmd = train_command(spec, export, base, out / "lerobot", args.seed, steps, bs, lr, rename,
                        arm.get("lerobot_overrides") or {})
    run = {"arm": args.arm, "seed": args.seed, "family": spec.family, "steps": steps,
           "batch_size": bs, "lr": lr, "data_hash": dhash, "export": str(export),
           "base_checkpoint": base, "rename_map": rename, "vla_passthrough": pt, "command": cmd,
           "provenance": stamp(["data", f"arms/{args.arm}"], source, seed=args.seed)}
    (out / "run.json").write_text(json.dumps(run, indent=2, default=float))
    print(json.dumps(run, indent=2, default=float))
    if args.launch:
        if args.dry_run:
            raise SystemExit("--launch needs real data")
        raise SystemExit(subprocess.call(cmd))


def train_usvla(args: argparse.Namespace, arm: dict) -> None:
    """US-VLA arm: our half of the training copy (npz + manifest) with the decode passthrough,
    then run.json with the commands for the US-VLA environment (scripts/usvla_run.py).
    ``--launch`` runs build-dataset and train in ``--usvla-root`` with ``uv run``."""
    import subprocess

    from usintent.policies.usvla_adapter import (
        arm_spec_from_config as usvla_spec,
    )
    from usintent.policies.usvla_adapter import (
        decode_passthrough,
        export_usvla_episodes,
        run_commands,
        train_params,
    )
    from usintent.protocol.recorder import camera_topic

    if args.target_departure_fraction is not None:
        raise SystemExit("the training-data intervention is implemented for DP arms only")
    data = load("data")
    if args.dry_run:
        from usintent.sim.holdworld import HoldWorldConfig, generate

        eps = generate(HoldWorldConfig(image_shape=(32, 32), fps=10.0, n_stations=(2, 3)), 3)
        steps, bs, lr, dhash, source, cam = 10, 4, 5e-5, "synthetic", "synthetic", False
        root = Path(tempfile.mkdtemp(prefix="usintent_usvla_dry_"))
    else:
        from usintent.data.loaders import load_dataset

        eps = load_dataset(args.train)
        steps = int(require(arm, "schedule.steps"))
        bs = int(require(arm, "schedule.batch_size"))
        lr = float(require(arm, "schedule.lr"))
        dhash, source = data_hash(args.train), "cued"
        cam = camera_topic(load("rig")) is not None
        root = RESULTS / "usvla"
    n = round(float(data["chunk_horizon_s"]) * eps[0].fps)
    spec = usvla_spec(arm, n, cam)
    repo_id = f"usintent/{args.arm}_{dhash}"
    export = root / "data" / f"{args.arm}_{dhash}"
    if not export.exists():
        export_usvla_episodes(eps, export, spec, repo_id)
    pt = decode_passthrough(eps, n)
    out = root / "train" / args.arm / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "usintent_spec.json").write_text(json.dumps(spec.__dict__, indent=2))
    script = Path(__file__).resolve().parent / "usvla_run.py"
    cmds = run_commands(script, export, out / "run.json")
    run = {"arm": args.arm, "seed": args.seed, "family": "usvla", "steps": steps,
           "batch_size": bs, "lr": lr, "data_hash": dhash, "export": str(export),
           "usvla_commit": arm.get("usvla_commit"), "decode_passthrough": pt,
           "usvla_params": train_params(spec, repo_id, args.seed, steps, bs, lr),
           "commands": cmds, "cwd": str(args.usvla_root or "<US-VLA checkout>"),
           "provenance": stamp(["data", f"arms/{args.arm}"], source, seed=args.seed)}
    (out / "run.json").write_text(json.dumps(run, indent=2, default=float))
    print(json.dumps(run, indent=2, default=float))
    if args.launch:
        if args.dry_run or args.usvla_root is None:
            raise SystemExit("--launch needs real data and --usvla-root")
        for step in ("build_dataset", "train"):   # build-dataset only checks if it already exists
            code = subprocess.call(cmds[step], cwd=args.usvla_root)
            if code:
                raise SystemExit(code)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--train", type=Path)
    ap.add_argument("--target-departure-fraction", type=float)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--gr00t-root", type=Path, help="Isaac-GR00T checkout (GR00T arms)")
    ap.add_argument("--usvla-root", type=Path, help="US-VLA checkout (USVLA-1)")
    ap.add_argument("--launch", action="store_true", help="GR00T: run the fine-tune (GPU)")
    ap.add_argument("--language-mode", action="store_true",
                    help="GR00T: instruction encodes the operator mode (language-delivery check)")
    args = ap.parse_args()
    arm = load(f"arms/{args.arm}")
    if arm["family"] == "gr00t_h":
        train_gr00t(args, arm)
        return
    if arm["family"] in ("smolvla", "pi05"):
        train_vla(args, arm)
        return
    if arm["family"] == "usvla":
        train_usvla(args, arm)
        return
    import torch

    from usintent.policies.act_adapter import ACTPolicy
    from usintent.policies.act_adapter import arm_spec_from_config as act_spec
    from usintent.policies.dp_adapter import DPPolicy, arm_spec_from_config

    data = load("data")
    if args.dry_run:
        from usintent.sim.holdworld import HoldWorldConfig, generate

        eps = generate(HoldWorldConfig(image_shape=(32, 32), fps=10.0, n_stations=(2, 3)), 4)
        steps, bs, lr, dhash, source = 10, 4, 1e-4, "synthetic", "synthetic"
        out = Path(tempfile.mkdtemp(prefix="usintent_dry_"))
    else:
        from usintent.data.loaders import load_dataset

        eps = load_dataset(args.train)
        steps = int(require(arm, "schedule.steps"))
        bs = int(require(arm, "schedule.batch_size"))
        lr = float(require(arm, "schedule.lr"))
        dhash, source = data_hash(args.train), "cued"
        tag = args.arm + (f"_p{args.target_departure_fraction:g}" if args.target_departure_fraction else "")
        out = RESULTS / "train" / tag / f"seed{args.seed}"
    k = int(data["history_frames_k"])
    sampler = ChunkSampler(eps, float(data["chunk_horizon_s"]), history_k=k)
    n = sampler.n_steps(eps[0])
    img = tuple(eps[0].bmode.shape[1:])
    is_act = arm["family"] == "act"
    if is_act and args.target_departure_fraction is not None:
        raise SystemExit("the training-data intervention is defined for the DP arms (paper Sec. VI)")
    spec = act_spec(arm, n, img) if is_act else arm_spec_from_config(arm, n, img, k)
    rng = np.random.default_rng(args.seed)
    weights, realized = None, None
    if args.target_departure_fraction is not None:
        iw = intervention_weights(sampler.index, args.target_departure_fraction)
        weights = iw.weights
        realized = realized_fraction(sampler.index, epoch_indices(weights, rng))
    policy = ACTPolicy(spec, seed=args.seed) if is_act else DPPolicy(spec, seed=args.seed)  # type: ignore[arg-type]
    losses = policy.train_steps(sampler, steps, bs, lr, rng, weights)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": policy.model.state_dict(), "norm_lo": policy.norm.lo,
                "norm_hi": policy.norm.hi, "spec": spec.__dict__}, out / "model.pt")
    run = {"arm": args.arm, "seed": args.seed, "family": arm["family"], "steps": steps,
           "batch_size": bs, "lr": lr,
           "target_departure_fraction": args.target_departure_fraction,
           "realized_departure_fraction_one_epoch": realized, "final_loss": losses[-1],
           "data_hash": dhash, "provenance": stamp(["data", f"arms/{args.arm}"], source,
                                                   seed=args.seed)}
    (out / "run.json").write_text(json.dumps(run, indent=2, default=float))
    print(json.dumps(run, indent=2, default=float))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
