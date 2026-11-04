"""rosbag2 (MCAP) profile + bag-to-LeRobot converter. M5.

``ros2 run us_intent_recorder recorder_node <out_dir>`` starts ``ros2 bag record -s mcap`` on
every topic of docs/INTERFACES.md (rig topic names from configs/rig.yaml; nulls refuse). Stop
with Ctrl-C at the end of the episode. Conversion: scripts/run_convert_bags.py.
"""
import subprocess
import sys
from pathlib import Path


def main() -> None:
    from usintent.config import load
    from usintent.protocol.recorder import record_command

    if len(sys.argv) < 2:
        raise SystemExit("usage: recorder_node <out_dir>")
    cmd = record_command(load("rig"), Path(sys.argv[1]))
    print("recording:", " ".join(cmd))
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
