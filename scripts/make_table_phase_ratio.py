"""Prelim check 1: open-loop error ratio to B2 (constant velocity) by phase. SPEC 13, M4.

Real data (lab):
  python scripts/make_table_phase_ratio.py --source openh --dataset <root> --repo-id <id> \
      --predictions results/predictions/<model>.parquet
SYNTHETIC end-to-end check (cloud; B1 zero-motion predictions stand in for a model):
  python scripts/make_table_phase_ratio.py --source synthetic --out /tmp/check
Writes phase_ratio_<model>.{json,csv,md} with git and config hashes. The passthrough check runs
first; the script aborts if it fails.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from usintent.config import load
from usintent.data.chunks import build_sampler
from usintent.data.loaders import load_dataset, load_openh
from usintent.data.passthrough import passthrough_check
from usintent.paths import FIGURES
from usintent.policies.evaluate import (
    EvalContext,
    align_predictions,
    load_predictions,
    phase_ratio_table,
    predict_chunks,
    save_predictions,
)
from usintent.provenance import output_dir, stamp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["openh", "cued", "synthetic"], required=True)
    ap.add_argument("--dataset", type=Path)
    ap.add_argument("--repo-id", default="openh/local")
    ap.add_argument("--predictions", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--synthetic-episodes", type=int, default=40)
    args = ap.parse_args()
    cfg = load("data")
    out = output_dir(args.source, args.out, FIGURES)
    out.mkdir(parents=True, exist_ok=True)
    if args.source == "synthetic":
        from usintent.baselines.b1_zero import ZeroMotion
        from usintent.sim.holdworld import HoldWorldConfig, generate

        episodes = generate(HoldWorldConfig(render_images=False, seed=101), args.synthetic_episodes,
                            sessions=4)
    elif args.source == "openh":
        episodes = load_openh(args.dataset, cfg["openh"], repo_id=args.repo_id)
    else:
        episodes = load_dataset(args.dataset)
    sampler = build_sampler(episodes, args.source, cfg, history_k=2)
    pt = passthrough_check(sampler, max_chunks=20000)
    if args.source == "synthetic":
        rows = sampler.index.iloc[::3]
        n = sampler.n_steps(episodes[0])
        pred, m_hat = predict_chunks(ZeroMotion(n), EvalContext(sampler, None), rows)
        args.predictions = out / "synthetic_B1.parquet"
        save_predictions(args.predictions, rows, pred, m_hat, {"model": "B1-synthetic", "arm_id": "B1"})
    pred_rows, chunks, meta = load_predictions(args.predictions)
    rows, chunks = align_predictions(sampler.index, pred_rows, chunks)
    tab = phase_ratio_table(sampler, rows, chunks)
    name = str(meta.get("model", args.predictions.stem))
    payload = {"model": meta, "passthrough": pt, "table": tab.to_dict(orient="records"),
               "provenance": stamp(["data"], args.source, n_episodes=len(episodes),
                                   predictions=str(args.predictions))}
    (out / f"phase_ratio_{name}.json").write_text(json.dumps(payload, indent=2, default=float))
    tab.to_csv(out / f"phase_ratio_{name}.csv", index=False)
    md = [f"Phase ratio to B2, model {name} ({args.source}); git {payload['provenance']['git_hash']}",
          "", "| phase | n | policy err (m) | B2 err (m) | ratio | ratio excl. transitions | B2≈0 |",
          "|---|---|---|---|---|---|---|"]
    md += [f"| {r.phase} | {r.n} | {r.mean_policy:.4g} | {r.mean_cv:.4g} | {r.ratio:.3g} | "
           f"{r.ratio_excl_transitions:.3g} | {'yes' if r.unstable else ''} |" for r in tab.itertuples()]
    (out / f"phase_ratio_{name}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
