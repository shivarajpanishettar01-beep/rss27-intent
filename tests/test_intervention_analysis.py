"""H2 analysis script on a SYNTHETIC run table with a known slope."""
import json
import subprocess
import sys

import numpy as np

from usintent.paths import SCRIPTS


def test_slope_recovered(tmp_path) -> None:
    rng = np.random.default_rng(0)
    rows = [{"arm": "DP-1", "p": p, "seed": s, "rate": 0.02 + 0.8 * p + rng.normal(0, 0.005)}
            for p in (0.08, 0.17, 0.33) for s in range(5)]
    f = tmp_path / "rows.json"
    f.write_text(json.dumps(rows))
    r = subprocess.run([sys.executable, str(SCRIPTS / "run_intervention_analysis.py"), str(f),
                        "--source", "synthetic", "--n-perm", "2000"], capture_output=True, text=True,
                       check=False)
    assert r.returncode == 0, r.stderr
    res = json.loads((tmp_path / "rows_analysis.json").read_text())["by_arm"]["DP-1"]
    assert abs(res["slope"] - 0.8) < 0.05 and res["p_value"] < 0.01 and res["n_runs"] == 15
