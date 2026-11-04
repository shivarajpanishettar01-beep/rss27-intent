"""Print Open-H robotic-ultrasound episode/frame counts from dataset metadata (M2).

Run on the lab machine after downloading:  python scripts/run_openh_counts.py <root> [<root> ...]
Reads meta/info.json of each LeRobot v3.0 dataset folder; prints JSON with provenance and
writes results/openh_counts.json. Compare with the paper's 1,629 real vascular / 4,752 simulated
liver episodes (docs/RIG_FACTS.md, status user-reported).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from usintent.data.loaders import dataset_info
from usintent.paths import RESULTS
from usintent.provenance import stamp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", nargs="+", type=Path)
    args = ap.parse_args()
    rows = {str(r): dataset_info(r) for r in args.roots}
    payload = {"datasets": rows, "provenance": stamp(["data"], "openh")}
    print(json.dumps(payload, indent=2))
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "openh_counts.json").write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
