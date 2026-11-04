"""Policy-server logic (M8): one arm at the policy rate, intent gating, output log. SPEC 10, 12.

Gating is structural: ``subscriptions(arm)`` lists the topics a server for that arm may
subscribe to; the scheduler-mode topic appears only for intent-aware arms (A5, DP-5, B6), and
``set_scheduler_mode`` raises for blind arms. The ROS node (us_intent_policy_server) uses only
these functions. Every output is appended to ``log`` (and to a JSONL file if given).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from usintent.policies.base import INTENT_AWARE_ARMS, Policy

MODE_TOPIC = "/protocol/mode_cue"


class GatingError(PermissionError):
    """An intent-blind arm was offered the scheduler mode."""


def subscriptions(arm_id: str, rig_topics: dict[str, str]) -> list[str]:
    """Topics a policy server for ``arm_id`` subscribes to (rig topic names from config)."""
    topics = [rig_topics["bmode"], rig_topics["tcp_pose"], rig_topics["wrench"], "/joint_states"]
    if rig_topics.get("camera"):
        topics.append(rig_topics["camera"])
    if arm_id in INTENT_AWARE_ARMS:
        topics.append(MODE_TOPIC)
    return topics


class PolicyServer:
    def __init__(self, policy: Policy, log_path: Path | None = None):
        self.policy = policy
        self.mode: str | None = None
        self.log: list[dict[str, Any]] = []
        self.log_path = log_path

    def set_scheduler_mode(self, mode: str) -> None:
        if not self.policy.intent_aware:
            raise GatingError(f"arm {self.policy.arm_id} is intent-blind; scheduler mode refused")
        self.mode = mode

    def step(self, t: float, history: dict[str, Any]) -> tuple[NDArray[np.float64], str]:
        mode = self.mode if self.policy.intent_aware else None
        if self.policy.intent_aware and mode is None:
            raise RuntimeError("intent-aware arm has not received a scheduler mode yet")
        chunk, m_hat = self.policy.predict(history, mode)
        chunk = np.asarray(chunk, dtype=np.float64)
        rec = {"t": t, "arm_id": self.policy.arm_id, "mode_in": mode, "m_hat": m_hat,
               "chunk": chunk.ravel().tolist(), "n_steps": int(chunk.shape[0])}
        self.log.append(rec)
        if self.log_path is not None:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        return chunk, m_hat
