"""Print a session plan from configs/protocol.yaml and wait for typed confirmation (no ROS, no
robot). Same plan the scheduler node uses for the same arguments.

  python scripts/run_protocol_dry_run.py --session-id pilot-1 --seed 1 --distribution uniform
"""
from __future__ import annotations

import argparse

from usintent.config import load
from usintent.protocol.scheduler import plan_session
from usintent.protocol.session import NotConfirmed, confirm


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session-id", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--distribution", choices=["uniform", "shifted_exp"], required=True)
    ap.add_argument("--block-id", type=int, default=0)
    ap.add_argument("--operator-id", default="op")
    ap.add_argument("--episodes", type=int, default=1)
    args = ap.parse_args()
    plan = plan_session(load("protocol"), args.session_id, args.seed, args.distribution,
                        args.block_id, args.operator_id, args.episodes)
    try:
        confirm({**plan.info(), "episodes": [[h.__dict__ for h in e.holds] for e in plan.episodes]})
    except NotConfirmed:
        print("aborted")
        raise SystemExit(1) from None
    print("confirmed (dry run: nothing started)")


if __name__ == "__main__":
    main()
