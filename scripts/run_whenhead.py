"""Fit the when-head and report image share, LL gain and commit-rule timing (SPEC 9, M7).

  SYNTHETIC check: python scripts/run_whenhead.py --source synthetic [--leak]
  cued data (lab): python scripts/run_whenhead.py --source cued --train <root> --test <root>
Real runs read whenhead.yaml bin_width_s and coupling_threshold (nulls refuse) and write
results/whenhead/<name>.json; synthetic runs only print.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from usintent.config import load, require
from usintent.paths import RESULTS
from usintent.provenance import stamp
from usintent.whenhead.commit import commit_times
from usintent.whenhead.features import FeatureParams, hold_bins
from usintent.whenhead.model import WhenHead, image_share


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["synthetic", "cued"], required=True)
    ap.add_argument("--train", type=Path)
    ap.add_argument("--test", type=Path)
    ap.add_argument("--leak", action="store_true", help="synthetic only")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    wh = load("whenhead")
    if args.source == "synthetic":
        from dataclasses import replace

        from usintent.sim.holdworld import HoldWorldConfig, generate

        p = FeatureParams(0.1, 60.0, "proxy")
        cfg = HoldWorldConfig(image_shape=(32, 32), fps=10.0, attenuation=150.0, leak=args.leak)
        tr_eps, te_eps = generate(replace(cfg, seed=1), 250), generate(replace(cfg, seed=2), 150)
    else:
        from usintent.data.loaders import load_dataset

        p = FeatureParams(float(require(wh, "bin_width_s")), float(require(wh, "coupling_threshold")),
                          str(wh["confidence_impl"]))
        tr_eps, te_eps = load_dataset(args.train), load_dataset(args.test)
    tr, te = hold_bins(tr_eps, p), hold_bins(te_eps, p)
    full = WhenHead(list(wh["g_img_hidden"]), seed=args.seed).fit(tr)
    eonly = WhenHead(list(wh["g_img_hidden"]), image=False, seed=args.seed).fit(tr)
    ct = commit_times(full, te, p.bin_width_s, float(wh["commit_threshold"]))
    res = {"share_img_train": image_share(full, tr), "share_img_test": image_share(full, te),
           "ll_gain_per_hold_test": eonly.nll_per_hold(te) - full.nll_per_hold(te),
           "timing_error_s": float(ct["err"].mean()),
           "timing_error_eonly_s": float(commit_times(eonly, te, p.bin_width_s)["err"].mean()),
           "exit_type_accuracy": float((ct["exit_pred"] == ct["exit_type"]).mean()),
           "n_train_holds": int(tr["hold_key"].nunique()), "n_test_holds": int(te["hold_key"].nunique()),
           "params": p.__dict__,
           "provenance": stamp(["whenhead"], args.source, seed=args.seed, leak=args.leak)}
    print(json.dumps(res, indent=2, default=float))
    if args.source != "synthetic":
        out = RESULTS / "whenhead" / f"{args.source}_seed{args.seed}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
