"""Fig. 1c: predicted hold-timing errors from usintent.theory (M0, SPEC sections 2 and 13).

Writes <out>/teaser_c.json (values + provenance) and <out>/teaser_c.{png,pdf}.
Data source: theory (Monte Carlo, 2M samples, seed 0, cross-checked against closed forms).

Run: python scripts/make_fig_teaser_c.py [--out figures]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from usintent.config import load
from usintent.paths import FIGURES
from usintent.provenance import stamp
from usintent.theory import (
    ShiftedExp,
    Uniform,
    bound_median,
    bound_median_closed,
    err_calibrated_imitator,
    err_calibrated_imitator_closed,
    err_memoryless_hedger,
    err_memoryless_hedger_closed,
    pibar,
    pibar_tau,
)


def compute(horizon: float) -> dict[str, dict[str, float]]:
    proto = load("protocol")["hold_distributions"]
    dists = {
        "uniform": Uniform(proto["uniform"]["a"], proto["uniform"]["b"]),
        "shifted_exp": ShiftedExp(proto["shifted_exp"]["shift"], proto["shifted_exp"]["scale"]),
    }
    out: dict[str, dict[str, float]] = {}
    for name, dist in dists.items():
        hedger, early = err_memoryless_hedger(dist)
        hedger_cf, early_cf = err_memoryless_hedger_closed(dist)
        out[name] = {
            "median_rule": bound_median(dist),
            "median_rule_closed": bound_median_closed(dist),
            "calibrated_imitator": err_calibrated_imitator(dist),
            "calibrated_imitator_closed": err_calibrated_imitator_closed(dist),
            "memoryless_hedger": hedger,
            "memoryless_hedger_closed": hedger_cf,
            "hedger_leaves_early": early,
            "hedger_leaves_early_closed": early_cf,
            "pibar_H": pibar(dist, horizon),
            "pibar_tau_0.2": pibar_tau(dist, horizon, 0.2),
            "pibar_tau_0.4": pibar_tau(dist, horizon, 0.4),
        }
    return out


def plot(values: dict[str, dict[str, float]], out: Path) -> None:
    labels = {"median_rule": "median rule (Eq. 2 bound)", "calibrated_imitator": "calibrated imitator",
              "memoryless_hedger": "memoryless hedger"}
    names = {"uniform": "D ~ U[2,10] s", "shifted_exp": "D = 2 + Exp(4) s"}
    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    width = 0.26
    for i, (key, lab) in enumerate(labels.items()):
        xs = [j + (i - 1) * width for j in range(len(names))]
        ys = [values[d][key] for d in names]
        bars = ax.bar(xs, ys, width, label=lab)
        ax.bar_label(bars, fmt="%.2f", fontsize=7)
    ax.set_xticks(range(len(names)), list(names.values()))
    ax.set_ylabel("predicted E|T - D| (s)")
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"teaser_c.{ext}", dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=FIGURES)
    args = ap.parse_args()
    horizon = float(load("data")["chunk_horizon_s"])
    values = compute(horizon)
    args.out.mkdir(parents=True, exist_ok=True)
    plot(values, args.out)
    payload = {"values": values, "horizon_s": horizon,
               "provenance": stamp(["protocol", "data"], "theory", n_mc=2_000_000, seed=0)}
    (args.out / "teaser_c.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
