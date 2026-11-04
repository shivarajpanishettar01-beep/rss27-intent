"""Verify repo layout, imports, configs and ROS package wiring.

Run: python scripts/check_structure.py   (also runs as tests/test_structure.py)
"""
from __future__ import annotations

import sys

from usintent.layout import check_all

if __name__ == "__main__":
    problems = check_all()
    for p in problems:
        print("FAIL:", p)
    print("OK: layout, imports, configs and ROS wiring" if not problems else f"{len(problems)} problem(s)")
    sys.exit(1 if problems else 0)
