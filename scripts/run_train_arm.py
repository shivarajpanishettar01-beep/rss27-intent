"""Train one arm x seed (M6). Logs seed, git hash, config hash and data hash.

  Lab:   python scripts/run_train_arm.py --arm DP-1 --seed 0 --train $USINTENT_DATA/cued/<train> \
             [--target-departure-fraction 0.08]
  Dry run (SYNTHETIC, cloud): python scripts/run_train_arm.py --arm DP-5 --seed 0 --dry-run
  GR00T arm (lab GPU): ... --arm A5 --seed 0 --train <root> --gr00t-root <Isaac-GR00T> --launch
Schedule (steps, batch size, lr) comes from configs/arms/<arm>.yaml and is shared by all arms;
nulls refuse to run except in --dry-run (10 steps, batch 4). GR00T arms export a GR00T LeRobot-v2
copy and run Isaac-GR00T's launch_finetune with the seed (scripts/gr00t_finetune_seeded.py).
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--train", type=Path)
    ap.add_argument("--target-departure-fraction", type=float)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--gr00t-root", type=Path, help="Isaac-GR00T checkout (GR00T arms)")
    ap.add_argument("--launch", action="store_true", help="GR00T: run the fine-tune (GPU)")
    ap.add_argument("--language-mode", action="store_true",
                    help="GR00T: instruction encodes the operator mode (language-delivery check)")
    args = ap.parse_args()
    arm = load(f"arms/{args.arm}")
    if arm["family"] == "gr00t_h":
        train_gr00t(args, arm)
        return
    import torch

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
    spec = arm_spec_from_config(arm, n, tuple(eps[0].bmode.shape[1:]), k)
    rng = np.random.default_rng(args.seed)
    weights, realized = None, None
    if args.target_departure_fraction is not None:
        iw = intervention_weights(sampler.index, args.target_departure_fraction)
        weights = iw.weights
        realized = realized_fraction(sampler.index, epoch_indices(weights, rng))
    policy = DPPolicy(spec, seed=args.seed)
    losses = policy.train_steps(sampler, steps, bs, lr, rng, weights)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": policy.model.state_dict(), "norm_lo": policy.norm.lo,
                "norm_hi": policy.norm.hi, "spec": spec.__dict__}, out / "model.pt")
    run = {"arm": args.arm, "seed": args.seed, "steps": steps, "batch_size": bs, "lr": lr,
           "target_departure_fraction": args.target_departure_fraction,
           "realized_departure_fraction_one_epoch": realized, "final_loss": losses[-1],
           "data_hash": dhash, "provenance": stamp(["data", f"arms/{args.arm}"], source,
                                                   seed=args.seed)}
    (out / "run.json").write_text(json.dumps(run, indent=2, default=float))
    print(json.dumps(run, indent=2, default=float))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
