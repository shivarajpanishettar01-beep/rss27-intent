"""Fig. 1c values from usintent.theory (M0). Writes figures/teaser_c.json (+ plot in M0)."""
from __future__ import annotations

import json
import subprocess

from usintent.paths import FIGURES
from usintent.theory import (
    ShiftedExp,
    Uniform,
    bound_median,
    err_calibrated_imitator,
    err_memoryless_hedger,
)


def main() -> None:
    out = {}
    for name, dist in {"uniform_2_10": Uniform(2, 10), "shifted_exp_2_4": ShiftedExp(2, 4)}.items():
        hedger, early = err_memoryless_hedger(dist)
        out[name] = {"median_rule": bound_median(dist), "calibrated_imitator": err_calibrated_imitator(dist),
                     "memoryless_hedger": hedger, "hedger_leaves_early": early}
    try:
        out["git_hash"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        out["git_hash"] = "unknown (no commit yet)"
    FIGURES.mkdir(exist_ok=True)
    (FIGURES / "teaser_c.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
