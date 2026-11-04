"""Segmentation sensitivity under +-50 % threshold changes (SPEC section 4, M2).

Real data:      python scripts/run_segmentation_sensitivity.py --openh <root> [--repo-id ID]
                (thresholds from configs/data.yaml; nulls refuse to run) -> results/
Our own data:   python scripts/run_segmentation_sensitivity.py --dataset <root>
Synthetic test: python scripts/run_segmentation_sensitivity.py --synthetic 50 \
                    --v-hold 0.002 --t-hold-min 1.0 --t-onset 0.5     (printed only; SYNTHETIC)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from usintent.config import load
from usintent.data.loaders import load_dataset, load_openh
from usintent.data.segmentation import SegParams, params_from_config, sensitivity
from usintent.paths import RESULTS
from usintent.provenance import stamp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--openh", type=Path)
    src.add_argument("--dataset", type=Path)
    src.add_argument("--synthetic", type=int, metavar="N_EPISODES")
    ap.add_argument("--repo-id", default="openh/local")
    ap.add_argument("--v-hold", type=float)
    ap.add_argument("--t-hold-min", type=float)
    ap.add_argument("--t-onset", type=float)
    args = ap.parse_args()
    cfg = load("data")
    horizon = float(cfg["chunk_horizon_s"])
    if args.synthetic:
        from usintent.sim.holdworld import HoldWorldConfig, generate

        if None in (args.v_hold, args.t_hold_min, args.t_onset):
            ap.error("--synthetic needs explicit --v-hold --t-hold-min --t-onset")
        params = SegParams(args.v_hold, args.t_hold_min, args.t_onset,
                           float(cfg["segmentation"]["transition_window_s"]))
        episodes, source = generate(HoldWorldConfig(render_images=False), args.synthetic), "synthetic"
    else:
        params = params_from_config(cfg)
        if args.openh:
            episodes, source = load_openh(args.openh, cfg["openh"], repo_id=args.repo_id), "openh"
        else:
            episodes, source = load_dataset(args.dataset), "cued"
    rows = sensitivity(episodes, params, horizon_s=horizon)
    payload = {"rows": rows, "base_params": params.__dict__,
               "provenance": stamp(["data"], source, n_episodes=len(episodes))}
    print(json.dumps(payload, indent=2))
    if source != "synthetic":
        out = RESULTS / "segmentation_sensitivity" / f"{source}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
