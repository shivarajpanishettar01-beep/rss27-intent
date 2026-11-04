"""Discrete-time survival head h = sigmoid(g_img + g_int) + exit-type head. SPEC section 9. M7.

Per hold, bins k = 0, 1, ... of width Delta (= policy period) from hold start; features are
taken at each bin start (causal). Hazard h_k = P(key press in bin k | no key press before):
    h_k = sigmoid(g_img(kappa, s, f) + g_int(e_k))
g_img: MLP 3 -> 32 -> 32 -> 1 (sizes from configs/whenhead.yaml g_img_hidden); g_int: MLP on e.
Exit-type head: softmax over {SWEEP, LIFT} from (kappa, s, f, e) at the event bin.
Loss: survival negative log-likelihood per hold,
    -sum_k [ y_k log h_k + (1 - y_k) log(1 - h_k) ],
y_k = 1 only in the bin that contains the key press; right-censored holds (episode ended before
the key press) contribute only the (1 - y) terms (no event term).
Image share: Var(g_img) / (Var(g_img) + Var(g_int)) over hold bins, and the log-likelihood gain of
the full model over g_int alone (``fit(..., image=False)``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

IMG_FEATURES = ("kappa", "s", "f")
EXIT_CLASSES = ("SWEEP", "LIFT")


def _mlp(n_in: int, hidden: list[int]) -> Any:
    import torch

    layers: list[Any] = []
    d = n_in
    for h in hidden:
        layers += [torch.nn.Linear(d, h), torch.nn.Tanh()]
        d = h
    layers.append(torch.nn.Linear(d, 1))
    return torch.nn.Sequential(*layers)


@dataclass
class Standardizer:
    mean: NDArray[np.float64]
    std: NDArray[np.float64]

    @classmethod
    def fit(cls, x: NDArray[np.float64]) -> Standardizer:
        return cls(x.mean(axis=0), np.maximum(x.std(axis=0), 1e-6))

    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray((x - self.mean) / self.std, dtype=np.float64)


class WhenHead:
    """Two-source survival head. Train with :meth:`fit`; query per-bin hazards with :meth:`hazards`."""

    def __init__(self, hidden: list[int], image: bool = True, seed: int = 0):
        import torch

        torch.manual_seed(seed)
        self.image = image
        self.hidden = list(hidden)
        self.g_img = _mlp(len(IMG_FEATURES), hidden)
        self.g_int = _mlp(1, hidden)
        self.exit_head = torch.nn.Linear(len(IMG_FEATURES) + 1, len(EXIT_CLASSES))
        self.std_img: Standardizer | None = None
        self.e_scale = 1.0
        self.seed = seed

    # ------------------------------------------------------------------ tensors
    def _tensors(self, bins: pd.DataFrame) -> tuple[Any, Any]:
        import torch

        assert self.std_img is not None
        xi = torch.as_tensor(self.std_img(bins[list(IMG_FEATURES)].to_numpy(np.float64)),
                             dtype=torch.float32)
        xe = torch.as_tensor(bins[["e"]].to_numpy(np.float64) / self.e_scale, dtype=torch.float32)
        return xi, xe

    def logits(self, bins: pd.DataFrame) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """(g_img, g_int) per bin, as numpy (g_img is 0 for an e-only model)."""
        import torch

        xi, xe = self._tensors(bins)
        with torch.no_grad():
            gi = self.g_img(xi).squeeze(-1) if self.image else torch.zeros(len(bins))
            ge = self.g_int(xe).squeeze(-1)
        return gi.numpy().astype(np.float64), ge.numpy().astype(np.float64)

    def hazards(self, bins: pd.DataFrame) -> NDArray[np.float64]:
        gi, ge = self.logits(bins)
        return np.asarray(1.0 / (1.0 + np.exp(-(gi + ge))))

    def exit_proba(self, bins: pd.DataFrame) -> NDArray[np.float64]:
        """(m, 2) P(SWEEP), P(LIFT) per bin."""
        import torch

        xi, xe = self._tensors(bins)
        with torch.no_grad():
            p = torch.softmax(self.exit_head(torch.cat([xi, xe], dim=1)), dim=1)
        return np.asarray(p.numpy(), dtype=np.float64)

    # ------------------------------------------------------------------ training
    def fit(self, bins: pd.DataFrame, epochs: int = 600, lr: float = 1e-2,
            weight_decay: float = 1e-4) -> WhenHead:
        """Full-batch Adam on the survival NLL (+ exit-type cross-entropy on event bins)."""
        import torch

        self.std_img = Standardizer.fit(bins[list(IMG_FEATURES)].to_numpy(np.float64))
        self.e_scale = float(max(bins["e"].max(), 1e-6))
        xi, xe = self._tensors(bins)
        y = torch.as_tensor(bins["event"].to_numpy(np.float64), dtype=torch.float32)
        ev = bins["event"].to_numpy(bool)
        lab = torch.as_tensor([EXIT_CLASSES.index(t) if t in EXIT_CLASSES else 0
                               for t in bins.loc[ev, "exit_type"]], dtype=torch.long)
        params = list(self.g_int.parameters()) + list(self.exit_head.parameters())
        if self.image:
            params += list(self.g_img.parameters())
        opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        n_holds = max(1, bins["hold_key"].nunique())
        evt = torch.as_tensor(ev)
        for _ in range(epochs):
            opt.zero_grad()
            logit = self.g_int(xe).squeeze(-1)
            if self.image:
                logit = logit + self.g_img(xi).squeeze(-1)
            nll = torch.nn.functional.binary_cross_entropy_with_logits(logit, y, reduction="sum")
            loss = nll / n_holds
            if evt.any():
                ex = self.exit_head(torch.cat([xi[evt], xe[evt]], dim=1))
                loss = loss + torch.nn.functional.cross_entropy(ex, lab)
            loss.backward()
            opt.step()
        return self

    def nll_per_hold(self, bins: pd.DataFrame) -> float:
        """Mean survival NLL per hold (nats) on ``bins``."""
        return survival_nll(self.hazards(bins), bins["event"].to_numpy(np.float64),
                            bins["hold_key"].to_numpy())


def survival_nll(h: NDArray[np.float64], y: NDArray[np.float64], hold_key: NDArray[Any]) -> float:
    """-sum_k [y log h + (1 - y) log(1 - h)], averaged over holds (censored holds: y = 0)."""
    h = np.clip(h, 1e-9, 1 - 1e-9)
    ll = y * np.log(h) + (1 - y) * np.log(1 - h)
    return float(-ll.sum() / max(1, len(np.unique(hold_key))))


def image_share(model: WhenHead, bins: pd.DataFrame) -> float:
    """Var(g_img) / (Var(g_img) + Var(g_int)) over the given hold bins."""
    gi, ge = model.logits(bins)
    vi, ve = float(np.var(gi)), float(np.var(ge))
    return vi / (vi + ve) if vi + ve > 0 else 0.0
