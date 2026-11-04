"""Instruction-swap test (paper prelim check 3): does the instruction change GR00T-H's actions?

  python scripts/run_language_check.py --arm A5 --checkpoints <seed0 ckpt> <seed1 ckpt> \
      --dataset $USINTENT_DATA/cued/<test> [--stride 10]
Uses checkpoints trained with run_train_arm.py --language-mode (instruction = operator mode).
Base instruction = the true mode's instruction; swaps = the other modes' instructions. Seed
noise = same instruction, two seeds. Writes results/language_check/<arm>.json.
For the existing (Open-H) checkpoint use usintent.policies.evaluate.instruction_swap_effect with
your own inference call (scripts/LAB_RUN_M4.md).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from usintent.config import load
from usintent.data.chunks import ChunkSampler
from usintent.data.loaders import load_dataset
from usintent.paths import RESULTS
from usintent.policies.evaluate import EvalContext, instruction_swap_effect, load_policy
from usintent.policies.gr00t_adapter import MODE_INSTRUCTIONS
from usintent.provenance import stamp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--checkpoints", type=Path, nargs=2, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=10)
    args = ap.parse_args()
    data = load("data")
    eps = load_dataset(args.dataset)
    ctx = EvalContext(ChunkSampler(eps, float(data["chunk_horizon_s"]),
                                   history_k=int(data["history_frames_k"])),
                      float(data["segmentation"]["v_hold_mps"] or 0.0) or None)
    pols = {str(c): load_policy(args.arm, c) for c in args.checkpoints}
    rows = ctx.sampler.index.iloc[:: args.stride]

    def predict(ckpt, instruction, e, s):  # type: ignore[no-untyped-def]
        pol = pols[str(ckpt)]
        pol.instruction = instruction
        mode = ctx.intent_mode(e, s) if pol.intent_aware else None
        return pol.predict(ctx.history(e, s), mode)[0]

    out = {}
    for mode, base in MODE_INSTRUCTIONS.items():
        sub = rows[[ctx.intent_mode(int(e), int(s)) == mode for e, s in zip(rows.episode_index, rows.start)]]
        if len(sub):
            swaps = [v for k, v in MODE_INSTRUCTIONS.items() if k != mode]
            out[mode] = instruction_swap_effect(predict, sub, base, swaps,
                                                (str(args.checkpoints[0]), str(args.checkpoints[1])))
    payload = {"arm": args.arm, "by_mode": out, "provenance": stamp(["data"], "cued")}
    dest = RESULTS / "language_check" / f"{args.arm}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2, default=float))
    print(json.dumps(payload, indent=2, default=float))


if __name__ == "__main__":
    main()
