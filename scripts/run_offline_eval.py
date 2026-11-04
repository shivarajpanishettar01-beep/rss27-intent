"""Offline evaluation: every arm x seed x sampler setting on held-out cued sessions (M8).

  Lab:  python scripts/run_offline_eval.py --train <root> --test <root> --runs results/train/*/seed*
  SYNTHETIC check: python scripts/run_offline_eval.py --synthetic --out /tmp/x
Baselines B1-B4 and B6(B2) are always evaluated (B4 fitted on the training split; B3 transition
matrix from training labels). Every result row is preceded by a passing passthrough check.
thr = metrics.yaml departure_threshold_m and v_hold = data.yaml segmentation.v_hold_mps (nulls
refuse) except in --synthetic. Writes eval_table.{csv,json}, eval_chunks.parquet and
intervention_rows.json (runs trained with a target departure fraction).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from usintent.baselines.b1_zero import ZeroMotion
from usintent.baselines.b2_const_vel import ConstantVelocity
from usintent.baselines.b3_imm import IMMBaseline, IMMParams, estimate_transition_matrix
from usintent.baselines.b4_hazard_optimal import HazardOptimalBlind
from usintent.baselines.b6_stop_on_hold import StopOnHold
from usintent.config import load, require
from usintent.data.chunks import ChunkSampler, policy_target
from usintent.data.splits import assert_disjoint
from usintent.paths import RESULTS
from usintent.policies.evaluate import EvalContext, load_policy, offline_eval
from usintent.provenance import output_dir, stamp


def _b4(train: ChunkSampler) -> HazardOptimalBlind:
    keys, types = [], []
    for ep in train.episodes:
        for _, h in ep.holds.iterrows():
            k = int(np.searchsorted(ep.time, h["key_time"]))
            if np.isfinite(h["key_time"]) and k + train.n_steps(ep) < len(ep):
                keys.append((ep.episode_index, k))
                types.append(str(h["exit_type"]))
    rows = pd.DataFrame(keys, columns=["episode_index", "start"])
    d = np.concatenate([e.holds["D"].dropna().to_numpy() for e in train.episodes])
    return HazardOptimalBlind.fit(float(np.median(d)), policy_target(train.chunks6(rows)), types)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=Path)
    ap.add_argument("--test", type=Path)
    ap.add_argument("--runs", type=Path, nargs="*", default=[])
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--chunk-stride", type=int, default=3)
    args = ap.parse_args()
    data = load("data")
    source = "synthetic" if args.synthetic else "cued"
    out = output_dir(source, args.out, RESULTS / "eval")
    out.mkdir(parents=True, exist_ok=True)
    if args.synthetic:
        from usintent.sim.holdworld import HoldWorldConfig, generate

        tr_eps = generate(HoldWorldConfig(render_images=False, seed=301), 120, sessions=3)
        te_eps = generate(HoldWorldConfig(render_images=False, seed=302), 120, sessions=3)
        thr, v_hold = 0.002, 0.002
    else:
        from usintent.data.loaders import load_dataset

        tr_eps, te_eps = load_dataset(args.train), load_dataset(args.test)
        thr = float(require(load("metrics"), "departure_threshold_m"))
        v_hold = float(require(data, "segmentation.v_hold_mps"))
    h, k = float(data["chunk_horizon_s"]), int(data["history_frames_k"])
    assert_disjoint(tr_eps, te_eps)                       # CLAUDE.md rule 5
    train, test = ChunkSampler(tr_eps, h, history_k=k), ChunkSampler(te_eps, h, history_k=k)
    n = test.n_steps(te_eps[0])
    ctx = EvalContext(test, v_hold)
    rows = test.index.iloc[:: args.chunk_stride]
    p_imm = IMMParams(trans=estimate_transition_matrix(tr_eps), dt=1.0 / te_eps[0].fps)
    entries = [("B1", "B1", ZeroMotion(n)), ("B2", "B2", ConstantVelocity(n)),
               ("B3", "B3", IMMBaseline(p_imm, n)), ("B4", "B4", _b4(train)),
               ("B6(B2)", "B6", StopOnHold(ConstantVelocity(n), n))]
    table, chunks, interv = [], [], []
    for name, arm, pol in entries:
        r, per = offline_eval(name, arm, "-", "-", pol, ctx, rows, thr)
        table.append(r)
        chunks.append(per)
    for run_dir in args.runs:
        run = json.loads((run_dir / "run.json").read_text())
        cfg = load(f"arms/{run['arm']}")
        ckpt = run_dir / "model.pt"
        if run.get("family") == "gr00t_h":
            from usintent.policies.gr00t_adapter import gr00t_passthrough

            gr00t_passthrough(te_eps, n)                    # GR00T action path, rule 4
            ckpts = sorted(run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
            if not ckpts:
                raise SystemExit(f"{run_dir}: no GR00T checkpoint-* directory")
            ckpt = ckpts[-1]
        for det in cfg["sampler_controls"]["deterministic_noise"]:
            for steps in cfg["sampler_controls"]["integration_steps"]:
                pol = load_policy(run["arm"], ckpt, det, steps)
                setting = f"det={det},steps={steps}"
                r, per = offline_eval(f"{run['arm']}/{run_dir.parent.name}", run["arm"],
                                      run["seed"], setting, pol, ctx, rows, thr)
                table.append(r)
                chunks.append(per)
                if run.get("target_departure_fraction") is not None:
                    interv.append({"arm": run["arm"], "p": run["target_departure_fraction"],
                                   "seed": run["seed"], "setting": setting,
                                   "rate": r["departure_like_rate"]})
    tab = pd.DataFrame(table)
    prov = stamp(["data", "metrics"], source, thr_m=thr)
    tab.to_csv(out / "eval_table.csv", index=False)
    (out / "eval_table.json").write_text(json.dumps({"rows": table, "provenance": prov},
                                                    indent=2, default=float))
    pd.concat(chunks, ignore_index=True).to_parquet(out / "eval_chunks.parquet", index=False)
    (out / "intervention_rows.json").write_text(json.dumps(interv, indent=2, default=float))
    print(tab.to_string(index=False))


if __name__ == "__main__":
    main()
