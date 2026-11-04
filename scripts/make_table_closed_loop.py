"""Closed-loop table: timing error per arm vs predictions; image/force metrics; interventions.
SPEC 5, 11, 13 (M10).

  Lab:  python scripts/make_table_closed_loop.py --trials results/closedloop/*.parquet
        (one row per hold: arm, trial_id, D, T [, view_frac, force_ok, interventions, kappa])
  SYNTHETIC check (cloud): python scripts/make_table_closed_loop.py --synthetic --out /tmp/x
        scripted median rule / calibrated imitator / memoryless hedger on the mock robot, paired
        trials, both hold distributions; must reproduce Fig. 1c within Monte-Carlo error.
Timing CIs: cluster bootstrap over whole trials (the resampling unit). Writes
closed_loop.{json,md,csv} with git and config hashes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from usintent.config import load
from usintent.paths import FIGURES
from usintent.provenance import output_dir, stamp
from usintent.stats.tests import cluster_bootstrap, paired_cluster_bootstrap
from usintent.theory import (
    bound_median_closed,
    err_calibrated_imitator_closed,
    err_memoryless_hedger_closed,
)

PRED = {"scripted-median": bound_median_closed, "scripted-imitator": err_calibrated_imitator_closed,
        "scripted-hedger": lambda d: err_memoryless_hedger_closed(d)[0]}


def synthetic_trials(n_trials: int, dt: float) -> pd.DataFrame:
    from usintent.closedloop.trials import ScriptedLeaver, TrialConfig, run_paired
    from usintent.protocol.scheduler import distribution_from_config, plan_session

    proto = {**load("protocol"), "stations_per_episode": {"min": 4, "max": 4}}   # synthetic only
    out = []
    for name in ("uniform", "shifted_exp"):
        dist = distribution_from_config(proto, name)
        plans = plan_session(proto, f"syn-{name}", 7, name, 0, "syn", n_trials).episodes
        pols = {f"scripted-{r}": ScriptedLeaver(r, dist, 10, 0.001, seed=11) for r in
                ("median", "imitator", "hedger")}
        df = run_paired(pols, plans, TrialConfig(dt_s=dt, thr_m=0.002, max_hold_s=200.0))
        df["distribution"] = name
        out.append(df)
    return pd.concat(out, ignore_index=True)


def summarize(df: pd.DataFrame, proto: dict, reference: str | None) -> list[dict]:
    from usintent.protocol.scheduler import distribution_from_config

    rows = []
    for (dname, arm), g in df.groupby(["distribution", "arm"]):
        ci = cluster_bootstrap(g["err"].to_numpy(), g["trial_id"].to_numpy())
        row = {"distribution": dname, "arm": arm, "n_holds": len(g), "n_trials": g["trial_id"].nunique(),
               "timing_error_s": ci["estimate"], "ci_lo": ci["lo"], "ci_hi": ci["hi"],
               "never_left": int(g["T"].isna().sum())}
        if arm in PRED:
            row["predicted_s"] = float(PRED[arm](distribution_from_config(proto, dname)))
        for c in ("view_frac", "kappa", "interventions"):
            if c in g:
                row[c] = float(g[c].mean())
        if "force_ok" in g:
            row["force_in_band_frac"] = float(g["force_ok"].mean())
        if reference and arm != reference:
            ref = df[(df["distribution"] == dname) & (df["arm"] == reference)]
            m = g.merge(ref, on=["trial_id", "hold_id"], suffixes=("", "_ref"))
            p = paired_cluster_bootstrap(m["err"].to_numpy(), m["err_ref"].to_numpy(),
                                         m["trial_id"].to_numpy())
            row.update({"paired_diff_vs_ref_s": p["estimate"], "paired_lo": p["lo"], "paired_hi": p["hi"]})
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=Path, nargs="*")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--n-trials", type=int, default=600)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--reference", default=None)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    source = "synthetic" if args.synthetic else "lab"
    out = output_dir(source, args.out, FIGURES)
    out.mkdir(parents=True, exist_ok=True)
    proto = load("protocol")
    if args.synthetic:
        df = synthetic_trials(args.n_trials, args.dt)
        ref = args.reference or "scripted-median"
    else:
        df = pd.concat([pd.read_parquet(p) for p in args.trials], ignore_index=True)
        ref = args.reference
    rows = summarize(df, proto, ref)
    prov = stamp(["protocol"], source, n_rows=len(df), dt_s=args.dt if args.synthetic else None)
    (out / "closed_loop.json").write_text(json.dumps({"rows": rows, "provenance": prov}, indent=2,
                                                     default=float))
    pd.DataFrame(rows).to_csv(out / "closed_loop.csv", index=False)
    md = [f"git {prov['git_hash']}  source {source}", "",
          "| dist | arm | holds | trials | timing err (s) [95% CI] | predicted (s) |", "|---|---|---|---|---|---|"]
    md += [f"| {r['distribution']} | {r['arm']} | {r['n_holds']} | {r['n_trials']} | "
           f"{r['timing_error_s']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] | "
           f"{r.get('predicted_s', float('nan')):.3f} |" for r in rows]
    (out / "closed_loop.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
