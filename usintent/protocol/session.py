"""Session start-up: print the session configuration and wait for typed confirmation. M5.

Used by the scheduler node and the dry-run launch (and, for motion nodes, together with the
SAFETY.md checks). Nothing starts unless the operator types exactly ``yes``.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any


class NotConfirmed(RuntimeError):
    """Operator did not confirm."""


def confirm(summary: dict[str, Any], input_fn: Callable[[str], str] = input,
            print_fn: Callable[[str], None] = print) -> None:
    """Print ``summary`` as JSON and require the operator to type 'yes'."""
    print_fn("Session configuration:\n" + json.dumps(summary, indent=2, default=str))
    print_fn("Physical e-stop within reach? Type 'yes' to start, anything else aborts.")
    if input_fn("> ").strip() != "yes":
        raise NotConfirmed("operator did not type 'yes'")
