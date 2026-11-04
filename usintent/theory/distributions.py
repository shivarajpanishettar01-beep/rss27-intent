"""Hold-duration distributions D (seconds). SPEC section 2."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


class HoldDistribution:
    def sample(self, n: int, rng: np.random.Generator) -> NDArray[np.float64]:
        raise NotImplementedError

    def mean(self) -> float:
        raise NotImplementedError

    def median(self) -> float:
        raise NotImplementedError


@dataclass(frozen=True)
class Uniform(HoldDistribution):
    a: float
    b: float

    def sample(self, n: int, rng: np.random.Generator) -> NDArray[np.float64]:
        return rng.uniform(self.a, self.b, n)

    def mean(self) -> float:
        return (self.a + self.b) / 2

    def median(self) -> float:
        return (self.a + self.b) / 2


@dataclass(frozen=True)
class ShiftedExp(HoldDistribution):
    shift: float
    scale: float  # mean of the exponential part

    def sample(self, n: int, rng: np.random.Generator) -> NDArray[np.float64]:
        return self.shift + rng.exponential(self.scale, n)

    def mean(self) -> float:
        return self.shift + self.scale

    def median(self) -> float:
        return self.shift + self.scale * float(np.log(2.0))


@dataclass(frozen=True)
class Empirical(HoldDistribution):
    samples: NDArray[np.float64] = field(repr=False)

    def sample(self, n: int, rng: np.random.Generator) -> NDArray[np.float64]:
        return rng.choice(self.samples, size=n, replace=True)

    def mean(self) -> float:
        return float(np.mean(self.samples))

    def median(self) -> float:
        return float(np.median(self.samples))
