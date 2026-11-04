"""Departure-like rate vs elapsed hold time e under both hold distributions (SPEC 13, M8).

  Lab:  python scripts/make_fig_rate_vs_e.py --eval-chunks results/eval/eval_chunks.parquet \
            --names B4 DP-1/DP-1
  SYNTHETIC check: python scripts/make_fig_rate_vs_e.py --synthetic --out /tmp/x --tau 0.342
Theory overlay: threshold-corrected hazard P(D <= e + H - tau | D > e) for each distribution
(protocol.yaml), tau = metrics.yaml tau_s (null refuses; synthetic: --tau). In --synthetic mode
the recorded chunks are scored (oracle imitator), so the measured curve must follow the overlay.
Writes rate_vs_e.{json,png,pdf} with git and config hashes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from usintent.config import load, require
from usintent.metrics.open_loop import rate_vs_e
from usintent.paths import FIGURES
from usintent.protocol.scheduler import distribution_from_config
from usintent.provenance import output_dir, stamp
from usintent.theory import hazard_closed

EDGES = np.arange(0.0, 12.5, 0.5)


def synthetic_chunks(tau_thr: float) -> pd.DataFrame:
    from dataclasses import replace

    from usintent.data.chunks import ChunkSampler, policy_target
    from usintent.metrics.open_loop import departure_like
    from usintent.sim.holdworld import HoldWorldConfig, generate
    from usintent.theory import ShiftedExp

    rows = []
    base = HoldWorldConfig(render_images=False, speeds_mps=(0.02,))
    for name, dist, seed in (("uniform", base.hold_dist, 401), ("shifted_exp", ShiftedExp(2, 4), 402)):
        s = ChunkSampler(generate(replace(base, hold_dist=dist, distribution_name=name, seed=seed), 250), 1.0)
        idx = s.index[s.index["phase"] == "hold"]
        flags = departure_like(policy_target(s.chunks6(idx)), tau_thr)
        rows.append(pd.DataFrame({"name": "oracle-synthetic", "e": idx["e"].to_numpy(),
                                  "phase": "hold", "departure_like": flags, "distribution": name}))
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-chunks", type=Path)
    ap.add_argument("--names", nargs="*")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--tau", type=float)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    source = "synthetic" if args.synthetic else "cued"
    out = output_dir(source, args.out, FIGURES)
    out.mkdir(parents=True, exist_ok=True)
    h = float(load("data")["chunk_horizon_s"])
    if args.synthetic:
        if args.tau is None:
            raise SystemExit("--synthetic needs --tau")
        tau = args.tau
        per = synthetic_chunks(0.002)
    else:
        tau = float(require(load("metrics"), "tau_s"))
        per = pd.read_parquet(args.eval_chunks)
        if args.names:
            per = per[per["name"].isin(args.names)]
    proto = load("protocol")
    result: dict[str, dict[str, object]] = {}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    for ax, dname in zip(axes, ("uniform", "shifted_exp")):
        dist = distribution_from_config(proto, dname)
        mid = (EDGES[:-1] + EDGES[1:]) / 2
        theory = [hazard_closed(dist, float(e), h - tau) for e in mid]
        ax.plot(mid, theory, "k--", lw=1, label=r"theory $\pi(e)$, H$-\tau$")
        result[dname] = {"theory": dict(zip(mid.tolist(), theory))}
        sub = per[(per["distribution"] == dname) & (per["phase"] == "hold")]
        for name, grp in sub.groupby("name"):
            tab = rate_vs_e(grp["departure_like"].to_numpy(bool), grp["e"].to_numpy(float), EDGES)
            ok = tab["n"] >= 30
            ax.plot(((tab["e_lo"] + tab["e_hi"]) / 2)[ok], tab["rate"][ok], marker="o", ms=3, label=name)
            result[dname][str(name)] = tab.to_dict(orient="records")
        ax.set_title(dname, fontsize=8)
        ax.set_xlabel("elapsed hold time e (s)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("departure-like rate")
    axes[0].legend(fontsize=6, frameon=False)
    prov = stamp(["data", "metrics", "protocol"], source, tau_s=tau)
    fig.suptitle(f"{source}  git {prov['git_hash'][:8]}", fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"rate_vs_e.{ext}", dpi=200)
    (out / "rate_vs_e.json").write_text(json.dumps({"curves": result, "provenance": prov},
                                                  indent=2, default=float))
    print(f"wrote {out}/rate_vs_e.*")


if __name__ == "__main__":
    main()
