"""Hold-duration distributions D (seconds). SPEC section 2.

Every distribution exposes ``sample(n, rng)``, ``cdf(x)``, ``median()``, ``mean()`` and, where a
closed form exists, ``laplace(s) = E[exp(-s D)]`` (used for the memoryless-hedger closed form).
All values are in seconds (``s`` in 1/s).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray


class HoldDistribution:
    """Abstract hold-duration distribution over D >= 0 (seconds)."""

    def sample(self, n: int, rng: np.random.Generator) -> NDArray[np.float64]:
        raise NotImplementedError

    def cdf(self, x: ArrayLike) -> NDArray[np.float64]:
        """P(D <= x), elementwise."""
        raise NotImplementedError

    def mean(self) -> float:
        raise NotImplementedError

    def median(self) -> float:
        raise NotImplementedError

    def laplace(self, s: float) -> float:
        """E[exp(-s D)] for s >= 0."""
        raise NotImplementedError


@dataclass(frozen=True)
class Uniform(HoldDistribution):
    """D ~ U[a, b] seconds."""

    a: float
    b: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.a < self.b):
            raise ValueError(f"Uniform needs 0 <= a < b, got a={self.a}, b={self.b}")

    def sample(self, n: int, rng: np.random.Generator) -> NDArray[np.float64]:
        return rng.uniform(self.a, self.b, n)

    def cdf(self, x: ArrayLike) -> NDArray[np.float64]:
        xa = np.asarray(x, dtype=np.float64)
        return np.clip((xa - self.a) / (self.b - self.a), 0.0, 1.0)

    def mean(self) -> float:
        return (self.a + self.b) / 2

    def median(self) -> float:
        return (self.a + self.b) / 2

    def laplace(self, s: float) -> float:
        if s == 0.0:
            return 1.0
        return float((np.exp(-s * self.a) - np.exp(-s * self.b)) / (s * (self.b - self.a)))


@dataclass(frozen=True)
class ShiftedExp(HoldDistribution):
    """D = shift + Exp(mean=scale) seconds."""

    shift: float
    scale: float  # mean of the exponential part

    def __post_init__(self) -> None:
        if self.shift < 0.0 or self.scale <= 0.0:
            raise ValueError(f"ShiftedExp needs shift >= 0, scale > 0, got {self.shift}, {self.scale}")

    def sample(self, n: int, rng: np.random.Generator) -> NDArray[np.float64]:
        return self.shift + rng.exponential(self.scale, n)

    def cdf(self, x: ArrayLike) -> NDArray[np.float64]:
        xa = np.asarray(x, dtype=np.float64)
        return np.where(xa <= self.shift, 0.0, 1.0 - np.exp(-(xa - self.shift) / self.scale))

    def mean(self) -> float:
        return self.shift + self.scale

    def median(self) -> float:
        return self.shift + self.scale * float(np.log(2.0))

    def laplace(self, s: float) -> float:
        return float(np.exp(-s * self.shift) / (1.0 + s * self.scale))


@dataclass(frozen=True)
class Empirical(HoldDistribution):
    """Empirical distribution of observed hold durations (seconds)."""

    samples: NDArray[np.float64] = field(repr=False)

    def __post_init__(self) -> None:
        arr = np.asarray(self.samples, dtype=np.float64)
        if arr.ndim != 1 or arr.size == 0 or not np.all(np.isfinite(arr)):
            raise ValueError("Empirical needs a non-empty 1-D array of finite durations")
        object.__setattr__(self, "samples", np.sort(arr))

    def sample(self, n: int, rng: np.random.Generator) -> NDArray[np.float64]:
        return rng.choice(self.samples, size=n, replace=True)

    def cdf(self, x: ArrayLike) -> NDArray[np.float64]:
        xa = np.asarray(x, dtype=np.float64)
        idx = np.searchsorted(self.samples, xa, side="right")
        return np.asarray(idx / self.samples.size, dtype=np.float64)

    def mean(self) -> float:
        return float(np.mean(self.samples))

    def median(self) -> float:
        return float(np.median(self.samples))

    def laplace(self, s: float) -> float:
        return float(np.mean(np.exp(-s * self.samples)))
