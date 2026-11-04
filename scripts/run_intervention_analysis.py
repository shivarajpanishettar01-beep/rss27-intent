"""H2 causal test: departure-like rate ~ training departure fraction p across seeds (SPEC 8).

  python scripts/run_intervention_analysis.py results/eval/intervention_rows.json
Input: JSON list of {"arm", "p" (target departure fraction), "seed", "rate"} - one point per
trained run (CLAUDE.md rule 6), produced by the M8 evaluation harness. Output: OLS slope of rate
on p with a permutation null interval, printed and written next to the input.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from usintent.provenance import stamp
from usintent.stats.tests import slope_with_permutation


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("rows", type=Path)
    ap.add_argument("--source", default="cued", choices=["cued", "openh", "synthetic"])
    ap.add_argument("--n-perm", type=int, default=10000)
    args = ap.parse_args()
    rows = json.loads(args.rows.read_text())
    out = {}
    for arm in sorted({r["arm"] for r in rows}):
        sel = [r for r in rows if r["arm"] == arm]
        p = np.array([r["p"] for r in sel], float)
        y = np.array([r["rate"] for r in sel], float)
        out[arm] = {**slope_with_permutation(p, y, n_perm=args.n_perm), "n_runs": len(sel),
                    "points": sel}
    payload = {"by_arm": out, "provenance": stamp(["data"], args.source)}
    print(json.dumps(payload, indent=2, default=float))
    args.rows.with_name(args.rows.stem + "_analysis.json").write_text(
        json.dumps(payload, indent=2, default=float))


if __name__ == "__main__":
    main()
