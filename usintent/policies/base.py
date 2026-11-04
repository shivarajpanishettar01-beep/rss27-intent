"""Common policy interface. SPEC section 10.

Intent gating (paper Sec. IV): only intent-aware arms may receive the scheduler mode.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

INTENT_AWARE_ARMS = frozenset({"A5", "DP-5", "B6"})


class Policy(ABC):
    arm_id: str

    @property
    def intent_aware(self) -> bool:
        return self.arm_id in INTENT_AWARE_ARMS

    def predict(self, history: Any, mode: str | None = None) -> tuple[Any, str]:
        if mode is not None and not self.intent_aware:
            raise PermissionError(f"Arm {self.arm_id} is intent-blind and must not receive the scheduler mode")
        return self._predict(history, mode)

    @abstractmethod
    def _predict(self, history: Any, mode: str | None) -> tuple[Any, str]:
        """Return (action chunk in tool frame, contact command)."""
