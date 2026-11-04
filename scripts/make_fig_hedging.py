"""Prelim check 2 / H2: predicted pi-bar_tau vs measured departure-like rate. SPEC 13, M4.

"existing model" mode (lab): a fine-tuned GR00T-H checkpoint's open-loop predictions on Open-H
holds vs the training-data departure fraction:
  python scripts/make_fig_hedging.py --source openh --train-dataset <root> --eval-dataset <root> \
      --repo-id <id> --predictions results/predictions/<model>.parquet
  (thr = metrics.yaml departure_threshold_m, tau = metrics.yaml tau_s; nulls refuse to run)
SYNTHETIC end-to-end check (cloud): the recorded chunks themselves are scored as an oracle
imitator, so the measured rate must equal pi-bar_tau:
  python scripts/make_fig_hedging.py --source synthetic --out /tmp/check --thr 0.002 --tau 0.342
Writes hedging_<model>.{json,png,pdf} with git and config hashes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from usintent.config import load, require
from usintent.data.chunks import build_sampler, policy_target
from usintent.data.loaders import load_dataset, load_openh
from usintent.data.passthrough import passthrough_check
from usintent.data.splits import assert_disjoint
from usintent.paths import FIGURES
from usintent.policies.evaluate import (
    align_predictions,
    hedging_summary,
    intervention_prediction,
    load_predictions,
    save_predictions,
)
from usintent.provenance import output_dir, stamp


def _load(source: str, path: Path | None, cfg: dict, repo_id: str, seed: int):
    if source == "synthetic":
        from usintent.sim.holdworld import HoldWorldConfig, generate

        return generate(HoldWorldConfig(render_images=False, speeds_mps=(0.02,), seed=seed), 80,
                        sessions=4)
    if source == "openh":
        return load_openh(path, cfg["openh"], repo_id=repo_id)
    return load_dataset(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["openh", "cued", "synthetic"], required=True)
    ap.add_argument("--train-dataset", type=Path)
    ap.add_argument("--eval-dataset", type=Path)
    ap.add_argument("--repo-id", default="openh/local")
    ap.add_argument("--predictions", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--thr", type=float, help="synthetic only")
    ap.add_argument("--tau", type=float, help="synthetic only")
    ap.add_argument("--intervention-rows", type=Path,
                    help="intervention_rows.json from run_offline_eval.py (H2 panel)")
    args = ap.parse_args()
    cfg = load("data")
    out = output_dir(args.source, args.out, FIGURES)
    out.mkdir(parents=True, exist_ok=True)
    if args.source == "synthetic":
        if args.thr is None or args.tau is None:
            ap.error("--source synthetic needs --thr and --tau")
        thr, tau = args.thr, args.tau
    else:
        if args.thr is not None or args.tau is not None:
            ap.error("real data: thr and tau come from configs/metrics.yaml only")
        mcfg = load("metrics")
        thr, tau = float(require(mcfg, "departure_threshold_m")), float(require(mcfg, "tau_s"))
    train = build_sampler(_load(args.source, args.train_dataset, cfg, args.repo_id, 201),
                          args.source, cfg)
    ev = build_sampler(_load(args.source, args.eval_dataset, cfg, args.repo_id, 202), args.source, cfg)
    assert_disjoint(train.episodes, ev.episodes)          # CLAUDE.md rule 5
    pt = passthrough_check(ev, max_chunks=20000)
    if args.source == "synthetic":
        rows = ev.index[ev.index["phase"] == "hold"]
        args.predictions = out / "synthetic_oracle.parquet"
        save_predictions(args.predictions, rows, policy_target(ev.chunks6(rows)),
                         ["HOLD"] * len(rows), {"model": "oracle-imitator-synthetic"})
    pred_rows, chunks, meta = load_predictions(args.predictions)
    rows, chunks = align_predictions(ev.index, pred_rows, chunks)
    summ = hedging_summary(train, rows, chunks, thr, tau, m_hat=pred_rows["m_hat"].astype(str).tolist())
    name = str(meta.get("model", args.predictions.stem))
    prov = stamp(["data", "metrics"], args.source, predictions=str(args.predictions))
    payload = {"model": meta, "summary": summ, "passthrough": pt, "provenance": prov}
    (out / f"hedging_{name}.json").write_text(json.dumps(payload, indent=2, default=float))
    interv = json.loads(args.intervention_rows.read_text()) if args.intervention_rows else []
    if interv:
        payload["intervention"] = [{**r, "predicted": intervention_prediction(
            r["p"], summ["horizon_s"], tau)} for r in interv]
        (out / f"hedging_{name}.json").write_text(json.dumps(payload, indent=2, default=float))
    fig, axes = plt.subplots(1, 2 if interv else 1, figsize=(7.5 if interv else 4.2, 2.8),
                             squeeze=False)
    ax = axes[0, 0]
    labels = ["train departure\nfraction", "predicted\n" + r"$\bar\pi_\tau$", f"measured\n{name}"]
    vals = [summ["train_departure_fraction"], summ["pibar_tau_pred"], summ["measured_rate"]]
    ax.bar(range(3), vals, color=["#9aa", "#68a", "#c64"])
    ax.errorbar([2], [summ["measured_rate"]], yerr=[[summ["measured_rate"] - summ["measured_lo"]],
                [summ["measured_hi"] - summ["measured_rate"]]], color="k", capsize=3)
    ax.set_xticks(range(3), labels, fontsize=7)
    ax.set_ylabel("hold chunks departure-like")
    ax.set_title(f"{args.source}  git {prov['git_hash'][:8]}", fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    if interv:
        ax2 = axes[0, 1]
        ps = np.array([r["p"] for r in interv])
        ax2.scatter(ps, [r["rate"] for r in interv], s=10, label="runs (one point per seed)")
        grid = np.linspace(0, ps.max() * 1.1, 20)
        ax2.plot(grid, [intervention_prediction(g, summ["horizon_s"], tau) for g in grid], "k--",
                 lw=1, label=r"$p\,(H-\tau)/H$")
        ax2.set_xlabel("training departure fraction p")
        ax2.legend(fontsize=6, frameon=False)
        ax2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"hedging_{name}.{ext}", dpi=200)
    print(json.dumps(payload, indent=2, default=float))


if __name__ == "__main__":
    main()
