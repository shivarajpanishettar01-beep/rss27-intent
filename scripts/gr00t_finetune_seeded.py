"""Run Isaac-GR00T's own gr00t/experiment/launch_finetune.py with a fixed seed (lab GPU).

launch_finetune.py does not expose the seed (config.data.seed, set_seed in experiment.py), and
seeds are our unit of analysis (CLAUDE.md rule 6). This wrapper patches
gr00t.configs.base_config.get_default_config so the returned config has data.seed = --seed, then
executes launch_finetune.py unchanged with the remaining arguments.

  python scripts/gr00t_finetune_seeded.py --gr00t-root /path/Isaac-GR00T --seed 3 -- <launch_finetune args>
Requires the Isaac-GR00T environment (uv venv from that repo).
"""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gr00t-root", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    import gr00t.configs.base_config as base

    original = base.get_default_config

    def seeded():  # type: ignore[no-untyped-def]
        cfg = original()
        cfg.data.seed = args.seed
        return cfg

    base.get_default_config = seeded
    launch = args.gr00t_root / "gr00t" / "experiment" / "launch_finetune.py"
    rest = args.rest[1:] if args.rest[:1] == ["--"] else args.rest
    sys.argv = [str(launch), *rest]
    runpy.run_path(str(launch), run_name="__main__")


if __name__ == "__main__":
    main()
