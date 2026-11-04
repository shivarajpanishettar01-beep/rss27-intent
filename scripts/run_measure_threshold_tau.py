"""Measure the departure threshold and tau from recorded cued episodes (paper check 4).

  Lab:  python scripts/run_measure_threshold_tau.py --dataset $USINTENT_DATA/cued/<pilot>
  SYNTHETIC check: python scripts/run_measure_threshold_tau.py --synthetic
Prints thr (m) and tau (s); copy them into configs/metrics.yaml (departure_threshold_m, tau_s).
Real runs also write results/calibration/threshold_tau.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from usintent.config import load
from usintent.data.chunks import ChunkSampler
from usintent.data.passthrough import passthrough_check
from usintent.metrics.calibration import measure_tau, tau_summary, zero_motion_threshold
from usintent.paths import RESULTS
from usintent.provenance import stamp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dataset", type=Path)
    g.add_argument("--synthetic", action="store_true")
    ap.add_argument("--quantile", type=float, default=0.999)
    ap.add_argument("--margin", type=float, default=1.5)
    args = ap.parse_args()
    if args.synthetic:
        from usintent.sim.holdworld import HoldWorldConfig, generate

        eps, source = generate(HoldWorldConfig(render_images=False, pose_jitter_m=0.0002), 60), "synthetic"
    else:
        from usintent.data.loaders import load_dataset

        eps, source = load_dataset(args.dataset), "cued"
    sampler = ChunkSampler(eps, float(load("data")["chunk_horizon_s"]))
    passthrough_check(sampler, max_chunks=5000)
    thr = zero_motion_threshold(sampler, args.quantile, args.margin)
    tau = tau_summary(measure_tau(eps, thr["threshold_m"]))
    payload = {"threshold": thr, "tau": tau, "provenance": stamp(["data"], source)}
    print(json.dumps(payload, indent=2, default=float))
    print(f"\n-> metrics.yaml: departure_threshold_m: {thr['threshold_m']:.6g}   tau_s: {tau['tau_s']:.4g}")
    if source != "synthetic":
        out = RESULTS / "calibration" / "threshold_tau.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, default=float))


if __name__ == "__main__":
    main()
