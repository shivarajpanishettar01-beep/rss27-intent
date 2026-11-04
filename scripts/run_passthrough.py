"""Ground-truth passthrough check on a dataset (CLAUDE.md rule 4). Exit code 1 on failure.

  python scripts/run_passthrough.py --openh <root> [--repo-id ID] [--episodes 1]
  python scripts/run_passthrough.py --dataset <root>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from usintent.config import load
from usintent.data.chunks import ChunkSampler
from usintent.data.loaders import load_dataset, load_openh
from usintent.data.passthrough import PassthroughError, passthrough_check
from usintent.provenance import stamp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--openh", type=Path)
    src.add_argument("--dataset", type=Path)
    ap.add_argument("--repo-id", default="openh/local")
    ap.add_argument("--episodes", type=int, default=None, help="check only the first N episodes")
    ap.add_argument("--max-chunks", type=int, default=20000)
    args = ap.parse_args()
    cfg = load("data")
    if args.openh:
        eps, source = load_openh(args.openh, cfg["openh"], repo_id=args.repo_id), "openh"
    else:
        eps, source = load_dataset(args.dataset), "cued"
    eps = eps[: args.episodes] if args.episodes else eps
    sampler = ChunkSampler(eps, float(cfg["chunk_horizon_s"]))
    try:
        res = passthrough_check(sampler, max_chunks=args.max_chunks)
        ok = True
    except PassthroughError as e:
        res, ok = {"error": str(e)}, False
    print(json.dumps({"passed": ok, **res, "provenance": stamp(["data"], source,
                                                             n_episodes=len(eps))}, indent=2))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
