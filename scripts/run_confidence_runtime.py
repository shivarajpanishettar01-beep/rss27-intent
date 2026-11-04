"""Measure confidence-map runtime per frame (SPEC 9: record it in reports/).

  python scripts/run_confidence_runtime.py --synthetic           # SYNTHETIC frames (cloud)
  python scripts/run_confidence_runtime.py --dataset <root>      # real B-mode frames (lab)
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np

from usintent.whenhead.confidence import measure_runtime


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--synthetic", action="store_true")
    g.add_argument("--dataset", type=Path)
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()
    if args.synthetic:
        from usintent.sim.holdworld import HoldWorldConfig, generate

        frames = generate(HoldWorldConfig(seed=3, n_stations=(2, 2)), 1)[0].bmode
        source = "synthetic"
    else:
        from usintent.data.loaders import load_dataset

        frames = np.concatenate([e.bmode for e in load_dataset(args.dataset)[:2]])
        source = "cued"
    frames = frames[: args.n]
    out = [measure_runtime(frames, "karamalis", s).__dict__ for s in ((16, 16), (32, 32), (64, 64))]
    out.append(measure_runtime(frames, "proxy").__dict__)
    print(json.dumps({"source": source, "frame_shape": list(frames.shape[1:]),
                      "machine": platform.processor() or platform.machine(), "results": out},
                     indent=2))


if __name__ == "__main__":
    main()
