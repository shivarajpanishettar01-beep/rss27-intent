"""Convert recorded MCAP bags (one per episode) to a LeRobot v3.0 dataset. M5.

  python scripts/run_convert_bags.py --day 2026-11-20 --out $USINTENT_DATA/cued/<name> bags/*.mcap
Topic names and the B-mode type come from configs/rig.yaml (nulls refuse). After writing, the
dataset is read back and the ground-truth passthrough check must pass (exit code 1 otherwise).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from usintent.config import load
from usintent.data.chunks import ChunkSampler
from usintent.data.loaders import read_lerobot, write_lerobot
from usintent.data.passthrough import PassthroughError, passthrough_check
from usintent.protocol.recorder import BagTopics, convert_bag


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bags", nargs="+", type=Path)
    ap.add_argument("--day", required=True, help="recording day id (split unit)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    topics = BagTopics.from_rig(load("rig"))
    eps = [convert_bag(b, topics, i, args.day) for i, b in enumerate(sorted(args.bags))]
    write_lerobot(eps, args.out)
    back = read_lerobot(args.out)
    try:
        res = passthrough_check(ChunkSampler(back, float(load("data")["chunk_horizon_s"])))
    except PassthroughError as e:
        print(f"PASSTHROUGH FAILED: {e}")
        sys.exit(1)
    print(json.dumps({"episodes": len(back), "holds": int(sum(len(e.holds) for e in back)),
                      "passthrough": res}, indent=2))


if __name__ == "__main__":
    main()
