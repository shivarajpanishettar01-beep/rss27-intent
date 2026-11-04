"""Leakage table (H3a): does anything beyond e predict hold end? SPEC 7 and 13.

  SYNTHETIC check (cloud):  python scripts/make_table_leakage.py --source synthetic --out /tmp/x
                            (runs leak off and leak on; expect no gain / clear gain)
  Open-H (lab):  python scripts/make_table_leakage.py --source openh --dataset <root> --repo-id <id>
  cued (lab):    python scripts/make_table_leakage.py --source cued --dataset <root>
Real runs read whenhead.yaml coupling_threshold (null refuses). Writes leakage_<source>.{json,md}.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from usintent.config import load, require
from usintent.data.loaders import load_dataset, load_openh
from usintent.data.segmentation import params_from_config, segment_threshold
from usintent.leakage.classifier import (
    build_dataset,
    evaluate,
    evaluate_remaining,
    permutation_test,
    permutation_test_remaining,
)
from usintent.paths import FIGURES
from usintent.provenance import output_dir, stamp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["openh", "cued", "synthetic"], required=True)
    ap.add_argument("--dataset", type=Path)
    ap.add_argument("--repo-id", default="openh/local")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--subsample-s", type=float, default=0.5)
    args = ap.parse_args()
    cfg = load("data")
    horizon = float(cfg["chunk_horizon_s"])
    out = output_dir(args.source, args.out, FIGURES)
    out.mkdir(parents=True, exist_ok=True)
    runs = {}
    if args.source == "synthetic":
        from usintent.sim.holdworld import HoldWorldConfig, generate

        for leak in (False, True):
            eps = generate(HoldWorldConfig(leak=leak, image_shape=(24, 24), fps=10.0, seed=5), 60,
                           sessions=6)
            runs[f"synthetic_leak_{'on' if leak else 'off'}"] = build_dataset(
                eps, horizon, args.subsample_s, coupling_threshold=60.0)
    else:
        thr = float(require(load("whenhead"), "coupling_threshold"))
        if args.source == "openh":
            eps = load_openh(args.dataset, cfg["openh"], repo_id=args.repo_id)
            p = params_from_config(cfg)
            segs = {e.episode_index: segment_threshold(e, p) for e in eps}
            runs["openh"] = build_dataset(eps, horizon, args.subsample_s, thr, segmentations=segs)
        else:
            runs["cued"] = build_dataset(load_dataset(args.dataset), horizon, args.subsample_s, thr)
    result = {}
    md = ["| data | model | features | log-loss | AUROC | n | holds | sessions |", "|---|---|---|---|---|---|---|---|"]
    for name, data in runs.items():
        tab = evaluate(data)
        perm = permutation_test(data, n_perm=args.n_perm)
        reg = evaluate_remaining(data)
        perm_r = permutation_test_remaining(data, n_perm=args.n_perm)
        result[name] = {"table": tab.to_dict(orient="records"), "permutation": perm,
                        "remaining_time": reg.to_dict(orient="records"),
                        "permutation_remaining": perm_r}
        md += [f"| {name} | {r.model} (remaining time) | {r.features} | MAE {r.mae_s:.3f} s | | {r.n} | | |"
               for r in reg.itertuples()]
        md.append(f"| {name} | permutation (remaining) | gain {perm_r['statistic_mae_gain_s']:.3f} s | "
                  f"p = {perm_r['p_value']:.4f} | | {perm_r['n_perm']} perms | | |")
        md += [f"| {name} | {r.model} | {r.features} | {r.logloss:.4f} | {r.auroc:.3f} | {r.n} | "
               f"{r.n_holds} | {r.n_sessions} |" for r in tab.itertuples()]
        md.append(f"| {name} | permutation | gain {perm['statistic_logloss_gain']:.4f} | "
                  f"p = {perm['p_value']:.4f} | | {perm['n_perm']} perms | | |")
    prov = stamp(["data", "whenhead"], args.source, horizon_s=horizon)
    (out / f"leakage_{args.source}.json").write_text(json.dumps({"runs": result, "provenance": prov},
                                                              indent=2, default=float))
    (out / f"leakage_{args.source}.md").write_text(f"git {prov['git_hash']}\n\n" + "\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
