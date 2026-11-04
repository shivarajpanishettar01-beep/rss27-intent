"""M0 acceptance: make_fig_teaser_c.py JSON matches reference/rss_predictions.py output."""
import json
import re
import subprocess
import sys

import pytest

from usintent.paths import REFERENCE, SCRIPTS


def _reference_values() -> dict[str, dict[str, float]]:
    txt = subprocess.check_output([sys.executable, str(REFERENCE / "rss_predictions.py")], text=True)
    blocks = txt.split("=== ")[1:]
    out = {}
    for name, block in zip(["uniform", "shifted_exp"], blocks):

        def num(pat: str, block: str = block) -> float:
            m = re.search(pat, block)
            assert m, pat
            return float(m.group(1))

        out[name] = {
            "median_rule": num(r"E\|D-med\(D\)\| = ([\d.]+)"),
            "calibrated_imitator": num(r"T~p_D = ([\d.]+)"),
            "memoryless_hedger": num(r"T~Exp\(E\[D\]\) = ([\d.]+)"),
            "hedger_leaves_early": num(r"leaves early in (\d+)%") / 100,
            "pibar_H": num(r"pi_bar \(H=1 s\)\s+= ([\d.]+)"),
            "pibar_tau_0.2": num(r"tau=0.2 s: \(H-tau\)/E\[D\] = ([\d.]+)"),
            "pibar_tau_0.4": num(r"tau=0.4 s: \(H-tau\)/E\[D\] = ([\d.]+)"),
        }
    return out


def test_teaser_json_matches_reference(tmp_path) -> None:
    subprocess.check_call([sys.executable, str(SCRIPTS / "make_fig_teaser_c.py"), "--out", str(tmp_path)],
                          stdout=subprocess.DEVNULL)
    payload = json.loads((tmp_path / "teaser_c.json").read_text())
    assert (tmp_path / "teaser_c.png").is_file() and (tmp_path / "teaser_c.pdf").is_file()
    prov = payload["provenance"]
    assert prov["git_hash"] and prov["config_hash"] and prov["data_source"] == "theory"
    ref = _reference_values()
    for dist, vals in ref.items():
        for key, v in vals.items():
            # reference prints 2-3 decimals (percent for 'early'); allow rounding + MC error
            tol = 0.01 if key == "hedger_leaves_early" else 0.02
            assert payload["values"][dist][key] == pytest.approx(v, abs=tol), (dist, key)
