"""Train one arm x seed (M6). Logs seed, git hash, config hash and data hash.

  Lab:   python scripts/run_train_arm.py --arm DP-1 --seed 0 --train $USINTENT_DATA/cued/<train> \
             [--target-departure-fraction 0.08]
  Dry run (SYNTHETIC, cloud): python scripts/run_train_arm.py --arm DP-5 --seed 0 --dry-run
Schedule (steps, batch size, lr) comes from configs/arms/<arm>.yaml and is shared by all arms;
nulls refuse to run except in --dry-run (10 steps, batch 4). GR00T arms are blocked (no source).
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--train", type=Path)
    ap.add_argument("--target-departure-fraction", type=float)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    arm = load(f"arms/{args.arm}")
    if arm["family"] != "diffusion_policy":
        from usintent.policies.gr00t_adapter import BLOCKED_REASON

        raise SystemExit(f"{args.arm}: {BLOCKED_REASON}")
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
