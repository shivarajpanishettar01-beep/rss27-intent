"""Open-loop evaluation harness shared by baselines (M3) and policies (M8). SPEC 5, 10.

Rules enforced here:
- CLAUDE.md rule 4: :func:`evaluate_open_loop` runs the ground-truth passthrough check first and
  refuses to score if it fails.
- Intent gating: the scheduler mode is passed only to intent-aware arms (policies/base.py);
  histories never contain it.
- Elapsed hold time for blind models comes from proprioception only (``e_proprio``).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from usintent.data.chunks import ChunkSampler, policy_target
from usintent.data.passthrough import passthrough_check
from usintent.data.segmentation import causal_elapsed
from usintent.metrics import open_loop as ol
from usintent.policies.base import Policy


class EvalContext:
    """Per-episode proprioceptive extras (e_proprio) for building policy histories."""

    def __init__(self, sampler: ChunkSampler, v_hold_mps: float | None):
        self.sampler = sampler
        self.e_proprio: dict[int, NDArray[np.float64]] = {}
        if v_hold_mps is not None:
            self.e_proprio = {e.episode_index: causal_elapsed(e, v_hold_mps)
                              for e in sampler.episodes}
        self._eps = {e.episode_index: e for e in sampler.episodes}

    def history(self, episode_index: int, start: int) -> dict[str, Any]:
        h: dict[str, Any] = self.sampler.history(episode_index, start)
        if self.e_proprio:
            h["e_proprio"] = self.e_proprio[episode_index][start:start + 1]
        return h

    def scheduler_mode(self, episode_index: int, start: int) -> str:
        return str(self._eps[episode_index].mode[start])


def predict_chunks(policy: Policy, ctx: EvalContext, rows: pd.DataFrame
                   ) -> tuple[NDArray[np.float64], list[str]]:
    """Run ``policy`` at every chunk start in ``rows`` (in time order per episode).

    Returns (m, n, 5) chunks and the contact commands m_hat.
    """
    chunks, m_hat = [], []
    for ep_idx, start in zip(rows["episode_index"].to_numpy(), rows["start"].to_numpy()):
        h = ctx.history(int(ep_idx), int(start))
        mode = ctx.scheduler_mode(int(ep_idx), int(start)) if policy.intent_aware else None
        c, m = policy.predict(h, mode)
        chunks.append(np.asarray(c, dtype=np.float64))
        m_hat.append(m)
    return np.stack(chunks), m_hat


def hold_timing(policy: Policy, ctx: EvalContext, thr_m: float, horizon_after_key_s: float = 12.0
                ) -> pd.DataFrame:
    """Timing per cued hold: T = time from hold start to the first step whose predicted chunk is
    departure-like (SPEC section 5); D from the sidecar. Columns: hold_id, D, T, err.

    Recorded steps are used until the key press. After it, the recorded robot moves on, so a
    policy that would have committed *late* cannot be timed from the recording. The hold is
    therefore extended counterfactually: the last in-hold observation (pose, wrench, frame) is
    repeated with advancing time and e_proprio, up to ``horizon_after_key_s``. This is exact for
    policies that depend on pose, force and e (B1-B4, B6) and an approximation for image policies
    (their true late behaviour is measured in closed loop, M10).
    """
    rows = []
    for ep in ctx.sampler.episodes:
        n = ctx.sampler.n_steps(ep)
        for _, h in ep.holds.iterrows():
            if h["censored"]:
                continue
            steps = np.flatnonzero(ep.hold_id == h["hold_id"])
            steps = steps[steps + n < len(ep)]
            t_first = np.nan
            for k in steps:
                c, _ = _predict_at(policy, ctx, ep.episode_index, int(k), ep, 0.0)
                if ol.max_displacement(c[None])[0] > thr_m:
                    t_first = float(ep.time[k] - h["start_time"])
                    break
            if np.isnan(t_first) and len(steps):
                last = int(steps[-1])
                for j in range(1, round(horizon_after_key_s * ep.fps) + 1):
                    c, _ = _predict_at(policy, ctx, ep.episode_index, last, ep, j * ep.dt)
                    if ol.max_displacement(c[None])[0] > thr_m:
                        t_first = float(ep.time[last] + j * ep.dt - h["start_time"])
                        break
            rows.append({"episode_index": ep.episode_index, "hold_id": int(h["hold_id"]),
                         "D": float(h["D"]), "T": t_first})
    df = pd.DataFrame(rows)
    df["err"] = np.abs(df["T"] - df["D"])
    return df


def _predict_at(policy: Policy, ctx: EvalContext, ep_idx: int, k: int, ep: Any, shift_s: float
                ) -> tuple[NDArray[np.float64], str]:
    """Predict at step k; with shift_s > 0, the observation at k is replayed shift_s later."""
    hist = ctx.history(ep_idx, k)
    if shift_s > 0:
        hist = dict(hist)
        hist["state"] = np.repeat(hist["state"][-1:], len(hist["state"]), axis=0)
        hist["wrench"] = np.repeat(hist["wrench"][-1:], len(hist["wrench"]), axis=0)
        hist["time"] = hist["time"][-1] + shift_s - (len(hist["time"]) - 1 - np.arange(
            len(hist["time"]))) * ep.dt
        if "bmode" in hist:
            hist["bmode"] = np.repeat(hist["bmode"][-1:], len(hist["bmode"]), axis=0)
        if "e_proprio" in hist:
            hist["e_proprio"] = hist["e_proprio"] + shift_s
    # in the extension the scheduler mode stays what it was at the last recorded hold step
    mode = str(ep.mode[k]) if policy.intent_aware else None
    c, m = policy.predict(hist, mode)
    return np.asarray(c, dtype=np.float64), m


def evaluate_open_loop(policy: Policy, ctx: EvalContext, rows: pd.DataFrame, thr_m: float,
                       cv_chunks: NDArray[np.float64] | None = None) -> dict[str, Any]:
    """Passthrough first (raises on failure), then the SPEC section 5 open-loop metrics."""
    pt = passthrough_check(ctx.sampler, max_chunks=5000)
    pred, _ = predict_chunks(policy, ctx, rows)
    gt = policy_target(ctx.sampler.chunks6(rows))
    in_hold = (rows["phase"] == "hold").to_numpy()
    dep = rows["contains_departure"].to_numpy(dtype=bool)
    err_t = ol.chunk_error_trans(pred, gt)
    out: dict[str, Any] = {
        "passthrough": pt,
        "n_chunks": len(rows),
        "chunk_error_trans_m": float(err_t.mean()),
        "chunk_error_rot_rad": float(ol.chunk_error_rot(pred, gt).mean()),
        "hold_drift_m": ol.hold_drift(pred, in_hold, dep),
        "departure_like_rate": ol.departure_like_rate(pred, in_hold, thr_m,
                                                      rows["exit_type"].to_numpy().astype(str)),
        "gt_departure_fraction": float(dep[in_hold].mean()) if in_hold.any() else float("nan"),
    }
    if cv_chunks is not None:
        out["ratio_to_cv"] = ol.ratio_to_cv(err_t, ol.chunk_error_trans(cv_chunks, gt),
                                            rows["phase"].to_numpy().astype(str))
    return out


# ---------------------------------------------------------------------------------------------
# Predictions file format (lab inference writes it; make_* scripts read it)
#   parquet columns: episode_index (int), start (int, step of chunk start), m_hat (str),
#                    chunk (list[float], n*5 row-major: x, y (m), rx, ry, rz (rad), tool frame)
#   sidecar <path>.meta.json: {"model", "arm_id", "seed", "checkpoint", "n_steps", ...}
# ---------------------------------------------------------------------------------------------


def save_predictions(path: Any, rows: pd.DataFrame, chunks: NDArray[np.float64], m_hat: list[str],
                     meta: dict[str, Any]) -> None:
    import json
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"episode_index": rows["episode_index"].to_numpy(dtype=np.int64),
                       "start": rows["start"].to_numpy(dtype=np.int64), "m_hat": m_hat,
                       "chunk": [c.ravel().tolist() for c in chunks]})
    df.to_parquet(path, index=False)
    Path(str(path) + ".meta.json").write_text(json.dumps({**meta, "n_steps": int(chunks.shape[1])},
                                                         indent=2))


def load_predictions(path: Any) -> tuple[pd.DataFrame, NDArray[np.float64], dict[str, Any]]:
    import json
    from pathlib import Path

    df = pd.read_parquet(path)
    meta = json.loads(Path(str(path) + ".meta.json").read_text())
    n = int(meta["n_steps"])
    chunks = np.stack([np.asarray(c, dtype=np.float64).reshape(n, 5) for c in df["chunk"]])
    return df.drop(columns=["chunk"]), chunks, meta


def align_predictions(index: pd.DataFrame, pred: pd.DataFrame, chunks: NDArray[np.float64]
                      ) -> tuple[pd.DataFrame, NDArray[np.float64]]:
    """Rows of the sampler index that have predictions, with chunks in the same order.
    Raises if a prediction refers to a chunk start that the sampler does not know."""
    key = index.reset_index().set_index(["episode_index", "start"])
    pk = list(zip(pred["episode_index"], pred["start"]))
    missing = [k for k in pk if k not in key.index]
    if missing:
        raise KeyError(f"{len(missing)} predictions do not match dataset chunk starts, "
                       f"e.g. {missing[:3]}")
    rows = index.loc[key.loc[pk, "index"].to_numpy()]
    return rows, chunks


# ---------------------------------------------------------------------------------------------
# Preliminary checks 1 and 2 (make_table_phase_ratio.py, make_fig_hedging.py)
# ---------------------------------------------------------------------------------------------


def phase_ratio_table(sampler: ChunkSampler, rows: pd.DataFrame, pred: NDArray[np.float64]
                      ) -> pd.DataFrame:
    """Prelim check 1: policy translation error / B2 error per phase, on the same chunks.
    ``sampler`` must have history_k >= 2 for B2. Includes B2 error and the unstable flag."""
    from usintent.baselines.b2_const_vel import ConstantVelocity

    if sampler.history_k < 2:
        raise ValueError("B2 needs history_k >= 2")
    n = pred.shape[1]
    ctx = EvalContext(sampler, None)
    cv, _ = predict_chunks(ConstantVelocity(n), ctx, rows)
    gt = policy_target(sampler.chunks6(rows))
    err_p = ol.chunk_error_trans(pred, gt)
    err_cv = ol.chunk_error_trans(cv, gt)
    tab = ol.ratio_to_cv(err_p, err_cv, rows["phase"].to_numpy().astype(str))
    no_tr = ~rows["in_transition"].to_numpy(dtype=bool)
    tab2 = ol.ratio_to_cv(err_p[no_tr], err_cv[no_tr], rows["phase"].to_numpy().astype(str)[no_tr])
    tab = tab.merge(tab2[["phase", "ratio"]].rename(columns={"ratio": "ratio_excl_transitions"}),
                    on="phase", how="left")
    return tab


def hedging_summary(train: ChunkSampler, eval_rows: pd.DataFrame, pred: NDArray[np.float64],
                    thr_m: float, tau_s: float, n_boot: int = 2000, seed: int = 0
                    ) -> dict[str, Any]:
    """Prelim check 2: measured departure-like rate on hold chunks vs the training-data
    departure fraction and the threshold-corrected prediction pi-bar_tau (Empirical D of the
    training holds). CI by cluster bootstrap over holds."""
    from usintent.stats.tests import cluster_bootstrap
    from usintent.theory import Empirical, pibar_tau

    tr = train.index
    tr_hold = tr[tr["phase"] == "hold"]
    durations = []
    for ep in train.episodes:
        seg = train.segmentations[ep.episode_index]  # type: ignore[index]
        for a, b in seg.holds:
            if seg.source == "cued":
                d = ep.holds.loc[ep.holds["hold_id"] == ep.hold_id[a], "D"]
                if len(d) and np.isfinite(float(d.iloc[0])):
                    durations.append(float(d.iloc[0]))
            elif b < len(ep):
                durations.append(float(ep.time[b] - ep.time[a]))
    dist = Empirical(np.asarray(durations))
    horizon = train.horizon_s
    in_hold = (eval_rows["phase"] == "hold").to_numpy()
    flags = ol.departure_like(pred, thr_m)[in_hold].astype(np.float64)
    hk = (eval_rows["episode_index"].astype(str) + "/" + eval_rows["seg_hold"].astype(str))
    ci = cluster_bootstrap(flags, hk.to_numpy()[in_hold], n_boot=n_boot, seed=seed)
    return {
        "train_departure_fraction": float(tr_hold["contains_departure"].mean()),
        "pibar_train_holds": float(np.mean(np.minimum(horizon, dist.samples)) / dist.mean()),
        "pibar_tau_pred": pibar_tau(dist, horizon, tau_s),
        "measured_rate": ci["estimate"], "measured_lo": ci["lo"], "measured_hi": ci["hi"],
        "n_eval_hold_chunks": int(in_hold.sum()), "n_train_holds": len(durations),
        "thr_m": thr_m, "tau_s": tau_s, "horizon_s": horizon,
    }
