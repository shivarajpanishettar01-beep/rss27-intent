"""Safety configuration gate (docs/SAFETY.md principle 3 and launch-file convention). M9.

No defaults: every limit in configs/safety.yaml and its human-set sanity range must be filled,
and every limit must lie inside its range, or motion code refuses to start. Real-robot launches
additionally need the flag i_have_read_safety_md:=true and a typed "yes" after the limits print.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from usintent.config import MissingRigFact, require

SCALAR_LIMITS = ("max_contact_force_n", "max_tcp_speed_mps", "max_tcp_accel_mps2",
                 "max_normal_displacement_m", "max_input_age_s", "watchdog_period_s")


class SafetyConfigError(RuntimeError):
    """A limit is missing, malformed or outside its sanity range."""


@dataclass(frozen=True)
class SafetyLimits:
    max_contact_force_n: float
    max_tcp_speed_mps: float
    max_tcp_accel_mps2: float
    max_normal_displacement_m: float
    workspace_box_base_m: tuple[float, float, float, float, float, float]
    max_input_age_s: float
    watchdog_period_s: float


def load_limits(cfg: dict[str, Any]) -> SafetyLimits:
    """Validate a loaded safety config; raises MissingRigFact or SafetyConfigError."""
    vals: dict[str, Any] = {}
    for k in SCALAR_LIMITS:
        v = float(require(cfg, k))
        lo, hi = (float(x) for x in require(cfg, f"sanity.{k}"))
        if not (v > 0 and lo <= v <= hi):
            raise SafetyConfigError(f"{k} = {v} outside sanity range [{lo}, {hi}] or not positive")
        vals[k] = v
    box = require(cfg, "workspace_box_base_m")
    if len(box) != 6 or not all(box[2 * i] < box[2 * i + 1] for i in range(3)):
        raise SafetyConfigError("workspace_box_base_m must be [xmin, xmax, ymin, ymax, zmin, zmax]")
    vals["workspace_box_base_m"] = tuple(float(b) for b in box)
    return SafetyLimits(**vals)


def preflight(cfg: dict[str, Any], i_have_read_safety_md: bool, real_robot: bool,
              input_fn: Callable[[str], str] = input,
              print_fn: Callable[[str], None] = print) -> SafetyLimits:
    """Gate run by every motion node before it creates any publisher."""
    if real_robot and not i_have_read_safety_md:
        raise SafetyConfigError("real robot: pass i_have_read_safety_md:=true (docs/SAFETY.md)")
    try:
        limits = load_limits(cfg)
    except MissingRigFact as e:
        raise SafetyConfigError(f"safety.yaml incomplete: {e}") from e
    print_fn("Active safety limits (second layer; the UR safety configuration is primary):")
    for k, v in limits.__dict__.items():
        print_fn(f"  {k} = {v}")
    if real_robot:
        print_fn("PHYSICAL E-STOP WITHIN REACH? Type 'yes' to continue.")
        if input_fn("> ").strip() != "yes":
            raise SafetyConfigError("operator did not confirm")
    return limits
