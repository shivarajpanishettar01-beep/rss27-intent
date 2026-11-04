"""Hold-end / remaining-time classifier with permutation test (H3a). SPEC section 7. M4.

Question: beyond elapsed time e, do observations predict that the current hold ends within H?
- Unit: time steps inside holds, subsampled every ``subsample_s``; grouped by hold and session.
- Target: ``ends_within`` = key press within (t, t + H]; ``remaining_s`` = key - t (optional).
- Feature sets: E = [e]; E+ctx = e, station index, number of stations, time since episode start;
  E+ctx+obs = plus a frozen image embedding, wrench statistics and image-adequacy features.
- Models: logistic regression and gradient boosting, session-grouped CV; log-loss and AUROC.
- Remaining-time version (paper H3(a) wording): regress ``remaining_s`` (ridge and gradient
  boosting); statistic MAE(E) - MAE(E+ctx+obs) (ridge), same permutation null.
- Test statistic: log-loss(E) - log-loss(E+ctx+obs) (logistic regression). Null: obs features
  permuted among samples of the same e bin (breaking any obs-outcome link beyond e), n_perm
  times; p = (1 + #{T_perm >= T}) / (1 + n_perm).
Image embedding: ``pooled_embedding`` (parameter-free, cloud default) or any callable, e.g.
:class:`ResNet18Embedder` with a local ImageNet weights file (lab; human to confirm which).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from usintent.data.loaders import Episode
from usintent.data.segmentation import Segmentation, segment_cued
from usintent.whenhead.features import (
    coupling_indicator,
    depth_profile_features,
    force_variance,
    pooled_embedding,
)

Embedder = Callable[[NDArray[np.uint8]], NDArray[np.float64]]
FEATURE_SETS = ("E", "E+ctx", "E+ctx+obs")


@dataclass
class LeakageData:
    e: NDArray[np.float64]          # (m,) s
    ctx: NDArray[np.float64]        # (m, 3) station index, n stations, time since episode start
    obs: NDArray[np.float64]        # (m, d)
    y: NDArray[np.bool_]            # ends within H
    remaining_s: NDArray[np.float64]
    groups: NDArray[np.str_]        # session ids (CV grouping)
    hold_key: NDArray[np.str_]      # "<episode>/<hold_id>"

    def features(self, name: str) -> NDArray[np.float64]:
        if name == "E":
            return self.e[:, None]
        if name == "E+ctx":
            return np.column_stack([self.e, self.ctx])
        if name == "E+ctx+obs":
            return np.column_stack([self.e, self.ctx, self.obs])
        raise ValueError(name)


def build_dataset(episodes: list[Episode], horizon_s: float, subsample_s: float,
                  coupling_threshold: float, embedder: Embedder = pooled_embedding,
                  force_window_s: float = 1.0,
                  segmentations: dict[int, Segmentation] | None = None) -> LeakageData:
    """Hold-step samples. Default (cued): holds = start .. key press from the logs. With
    threshold ``segmentations`` (Open-H): hold end = first moving step; holds that do not end
    inside the episode are skipped (no observed end)."""
    cols: dict[str, list[Any]] = {k: [] for k in ("e", "ctx", "obs", "y", "rem", "g", "h")}
    for ep in episodes:
        if ep.bmode is None:
            raise ValueError("leakage features need B-mode frames")
        seg = segment_cued(ep, 0.0) if segmentations is None else segmentations[ep.episode_index]
        fvar = force_variance(ep.wrench[:, 2], ep.time, force_window_s)
        keys = ep.holds.set_index("hold_id")["key_time"]
        n_st = float(len(seg.holds))
        stride = max(1, round(subsample_s * ep.fps))
        for a, b in seg.holds:
            if seg.source == "cued":
                hid = int(ep.hold_id[a])
                key = float(keys.loc[hid])
            else:
                if b >= len(ep):
                    continue
                hid, key = a, float(ep.time[b])
            for k in range(a, b, stride):
                t = ep.time[k]
                lo = int(np.searchsorted(ep.time, t - force_window_s))
                frame = ep.bmode[k]
                obs = np.concatenate([
                    embedder(frame),
                    [ep.wrench[lo:k + 1, 2].mean(), fvar[k],
                     coupling_indicator(frame, coupling_threshold)],
                    depth_profile_features(frame),
                ])
                cols["e"].append(seg.elapsed[k])
                cols["ctx"].append([float(ep.station_index[k]), n_st, t - ep.time[0]])
                cols["obs"].append(obs)
                cols["y"].append(np.isfinite(key) and key <= t + horizon_s)
                cols["rem"].append(key - t)
                cols["g"].append(ep.session_id)
                cols["h"].append(f"{ep.episode_index}/{hid}")
    return LeakageData(np.asarray(cols["e"], float), np.asarray(cols["ctx"], float),
                       np.asarray(cols["obs"], float), np.asarray(cols["y"], bool),
                       np.asarray(cols["rem"], float), np.asarray(cols["g"], str),
                       np.asarray(cols["h"], str))


def _model(kind: str, seed: int) -> Any:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    if kind == "gbm":
        return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05,
                                              max_leaf_nodes=15, random_state=seed)
    raise ValueError(kind)


def _folds(groups: NDArray[np.str_], n_splits: int) -> list[tuple[NDArray[np.int64], NDArray[np.int64]]]:
    from sklearn.model_selection import GroupKFold

    k = min(n_splits, len(np.unique(groups)))
    if k < 2:
        raise ValueError("session-grouped CV needs at least 2 sessions")
    dummy = np.zeros(len(groups))
    return [(tr, te) for tr, te in GroupKFold(k).split(dummy, dummy, groups)]


def oof_proba(x: NDArray[np.float64], y: NDArray[np.bool_], groups: NDArray[np.str_],
              kind: str = "logreg", n_splits: int = 5, seed: int = 0) -> NDArray[np.float64]:
    """Out-of-fold P(ends within H) with session-grouped folds."""
    p = np.zeros(len(y))
    for tr, te in _folds(groups, n_splits):
        if len(np.unique(y[tr])) < 2:
            p[te] = float(np.mean(y[tr]))
            continue
        m = _model(kind, seed).fit(x[tr], y[tr])
        p[te] = m.predict_proba(x[te])[:, 1]
    return p


def _scores(y: NDArray[np.bool_], p: NDArray[np.float64]) -> dict[str, float]:
    from sklearn.metrics import log_loss, roc_auc_score

    pc = np.clip(p, 1e-6, 1 - 1e-6)
    return {"logloss": float(log_loss(y, pc, labels=[False, True])),
            "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")}


def evaluate(data: LeakageData, models: tuple[str, ...] = ("logreg", "gbm"), n_splits: int = 5,
             seed: int = 0) -> pd.DataFrame:
    """CV log-loss and AUROC per model x feature set."""
    rows = []
    for kind in models:
        for fs in FEATURE_SETS:
            s = _scores(data.y, oof_proba(data.features(fs), data.y, data.groups, kind,
                                         n_splits, seed))
            rows.append({"model": kind, "features": fs, **s, "n": len(data.y),
                         "n_holds": len(np.unique(data.hold_key)),
                         "n_sessions": len(np.unique(data.groups))})
    return pd.DataFrame(rows)


def _statistic(data: LeakageData, obs: NDArray[np.float64], base_ll: float, n_splits: int,
               seed: int) -> float:
    x = np.column_stack([data.e, data.ctx, obs])
    return base_ll - _scores(data.y, oof_proba(x, data.y, data.groups, "logreg", n_splits,
                                               seed))["logloss"]


def permutation_test(data: LeakageData, n_perm: int = 1000, e_bin_s: float = 0.5,
                     n_splits: int = 5, seed: int = 0) -> dict[str, Any]:
    """Permutation test of the E+ctx+obs improvement over E (see module docstring)."""
    rng = np.random.default_rng(seed)
    base = _scores(data.y, oof_proba(data.features("E"), data.y, data.groups, "logreg",
                                     n_splits, seed))["logloss"]
    stat = _statistic(data, data.obs, base, n_splits, seed)
    bins = np.floor(data.e / e_bin_s).astype(np.int64)
    members = [np.flatnonzero(bins == b) for b in np.unique(bins)]
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = np.arange(len(data.y))
        for m in members:
            perm[m] = rng.permutation(m)
        null[i] = _statistic(data, data.obs[perm], base, n_splits, seed)
    p = (1 + int(np.sum(null >= stat))) / (1 + n_perm)
    return {"statistic_logloss_gain": stat, "p_value": p, "n_perm": n_perm,
            "null_mean": float(null.mean()), "null_q95": float(np.quantile(null, 0.95)),
            "logloss_E": base}


class ResNet18Embedder:
    """Frozen torchvision ResNet-18 (ImageNet) global-pool embedding (512-D).

    Weights must be a local file (``torchvision.models.resnet18`` state dict); this session
    cannot download them. Grey frames are replicated to 3 channels and normalized with the
    ImageNet mean/std. Lab use only until the human confirms the embedding choice.
    """

    def __init__(self, weights_path: str, size: int = 224):
        import torch
        import torchvision

        net = torchvision.models.resnet18(weights=None)
        net.load_state_dict(torch.load(weights_path, map_location="cpu"))
        net.fc = torch.nn.Identity()
        self.net = net.eval()
        self.size = size
        self._torch = torch

    def __call__(self, frame: NDArray[np.uint8]) -> NDArray[np.float64]:
        torch = self._torch
        x = torch.as_tensor(frame, dtype=torch.float32)[None, None] / 255.0
        x = torch.nn.functional.interpolate(x, size=(self.size, self.size), mode="bilinear")
        x = x.repeat(1, 3, 1, 1)
        mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
        with torch.no_grad():
            return np.asarray(self.net((x - mean) / std)[0].numpy(), dtype=np.float64)


# ---------------------------------------------------------------------------------------------
# Remaining-time regression (H3(a): "predict the remaining hold time")
# ---------------------------------------------------------------------------------------------


def _regressor(kind: str, seed: int) -> Any:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if kind == "gbm":
        return HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_leaf_nodes=15,
                                             random_state=seed)
    raise ValueError(kind)


def oof_remaining(x: NDArray[np.float64], y: NDArray[np.float64], groups: NDArray[np.str_],
                  kind: str = "ridge", n_splits: int = 5, seed: int = 0) -> NDArray[np.float64]:
    """Out-of-fold predicted remaining hold time (s), session-grouped folds."""
    p = np.zeros(len(y))
    for tr, te in _folds(groups, n_splits):
        p[te] = _regressor(kind, seed).fit(x[tr], y[tr]).predict(x[te])
    return p


def evaluate_remaining(data: LeakageData, models: tuple[str, ...] = ("ridge", "gbm"),
                       n_splits: int = 5, seed: int = 0) -> pd.DataFrame:
    """CV mean absolute error (s) of the remaining hold time per model x feature set."""
    rows = []
    ok = np.isfinite(data.remaining_s)
    for kind in models:
        for fs in FEATURE_SETS:
            p = oof_remaining(data.features(fs)[ok], data.remaining_s[ok], data.groups[ok], kind,
                              n_splits, seed)
            rows.append({"model": kind, "features": fs,
                         "mae_s": float(np.mean(np.abs(p - data.remaining_s[ok]))),
                         "n": int(ok.sum())})
    return pd.DataFrame(rows)


def permutation_test_remaining(data: LeakageData, n_perm: int = 1000, e_bin_s: float = 0.5,
                               n_splits: int = 5, seed: int = 0) -> dict[str, Any]:
    """Permutation test of the MAE reduction of E+ctx+obs over E (ridge), obs permuted within
    e bins as in :func:`permutation_test`."""
    rng = np.random.default_rng(seed)
    ok = np.isfinite(data.remaining_s)
    y, g, e = data.remaining_s[ok], data.groups[ok], data.e[ok]
    ctx, obs = data.ctx[ok], data.obs[ok]

    def mae(x: NDArray[np.float64]) -> float:
        return float(np.mean(np.abs(oof_remaining(x, y, g, "ridge", n_splits, seed) - y)))

    base = mae(e[:, None])
    stat = base - mae(np.column_stack([e, ctx, obs]))
    bins = np.floor(e / e_bin_s).astype(np.int64)
    members = [np.flatnonzero(bins == b) for b in np.unique(bins)]
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = np.arange(len(y))
        for m in members:
            perm[m] = rng.permutation(m)
        null[i] = base - mae(np.column_stack([e, ctx, obs[perm]]))
    return {"statistic_mae_gain_s": stat, "p_value": (1 + int(np.sum(null >= stat))) / (1 + n_perm),
            "n_perm": n_perm, "mae_E_s": base, "null_q95": float(np.quantile(null, 0.95))}
