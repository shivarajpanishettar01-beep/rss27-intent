"""GR00T-H (Isaac-GR00T) adapter: read the installed repo before coding. SPEC section 10. M6.

BLOCKED in the cloud session (2026-11-04): github.com and huggingface.co are unreachable through
this session's network policy, and Isaac-GR00T is not on PyPI under a name I could confirm, so
I could not read its fine-tuning entry point, data config (modality JSON) or action-head code.
CLAUDE.md rule 8 forbids guessing that API, so this adapter refuses to run instead of pretending.

What it must do once the source is readable (see reports/M6.md):
- load an arm checkpoint (A1, A2, A3, A5, A6, A7) and expose Policy.predict(history, mode);
- A5: add a learned mode embedding to the action head's conditioning (exact insertion point to
  be documented from the source);
- A6: time since last motion as an extra low-dimensional input; A7: when-head gating
  (usintent.whenhead.commit.CommitRule) in front of the action head;
- sampler controls: deterministic vs stochastic initial noise; 4 vs 16 integration steps.
"""
from __future__ import annotations

from typing import Any

from usintent.policies.base import Policy

BLOCKED_REASON = ("Isaac-GR00T source not readable from this environment; the GR00T-H adapter is "
                  "not implemented (CLAUDE.md rule 8: no invented APIs). See reports/M6.md.")


class GR00TPolicy(Policy):
    def __init__(self, arm_id: str, checkpoint: str | None = None):
        self.arm_id = arm_id
        raise NotImplementedError(BLOCKED_REASON)

    def _predict(self, history: Any, mode: str | None) -> tuple[Any, str]:  # pragma: no cover
        raise NotImplementedError(BLOCKED_REASON)
