"""M4: prelim-check scripts run end to end on SYNTHETIC data and carry provenance."""
import json
import subprocess
import sys

import pytest

from usintent.data.loaders import holds_table
from usintent.paths import FIGURES, SCRIPTS
from usintent.provenance import output_dir
from usintent.sim.holdworld import HoldWorldConfig, departure_tau, generate
from usintent.theory import Empirical, pibar_tau


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *args], capture_output=True, text=True, check=False)


def test_phase_ratio_synthetic(tmp_path) -> None:
    r = _run(str(SCRIPTS / "make_table_phase_ratio.py"), "--source", "synthetic", "--out", str(tmp_path))
    assert r.returncode == 0, r.stderr
    payload = json.loads((tmp_path / "phase_ratio_B1-synthetic.json").read_text())
    prov = payload["provenance"]
    assert prov["git_hash"] and prov["config_hash"] and prov["data_source"] == "synthetic"
    tab = {row["phase"]: row for row in payload["table"]}
    assert tab["sweep"]["ratio"] > 1.5          # zero motion is much worse than CV in sweeps
    assert payload["passthrough"]["max_trans_err_m"] < 1e-6


def test_hedging_synthetic_oracle_matches_pibar_tau(tmp_path) -> None:
    tau = round(departure_tau(0.002, 0.02, 0.5), 6)
    r = _run(str(SCRIPTS / "make_fig_hedging.py"), "--source", "synthetic", "--out", str(tmp_path),
             "--thr", "0.002", "--tau", f"{tau:.6f}")
    assert r.returncode == 0, r.stderr
    payload = json.loads((tmp_path / "hedging_oracle-imitator-synthetic.json").read_text())
    s = payload["summary"]
    assert (tmp_path / "hedging_oracle-imitator-synthetic.png").is_file()
    # train-data prediction (uses the training holds' D)
    eps_tr = generate(HoldWorldConfig(render_images=False, speeds_mps=(0.02,), seed=201), 80, sessions=4)
    assert s["pibar_tau_pred"] == pytest.approx(pibar_tau(Empirical(holds_table(eps_tr)["D"].to_numpy()),
                                                          1.0, tau), abs=1e-9)
    # the oracle's measured rate equals pi-bar_tau of the evaluation holds
    eps_ev = generate(HoldWorldConfig(render_images=False, speeds_mps=(0.02,), seed=202), 80, sessions=4)
    expect = pibar_tau(Empirical(holds_table(eps_ev)["D"].to_numpy()), 1.0, tau)
    assert s["measured_rate"] == pytest.approx(expect, abs=0.005)
    assert s["measured_lo"] < s["measured_rate"] < s["measured_hi"]
    assert payload["provenance"]["warning"].startswith("SYNTHETIC")


def test_real_mode_refuses_null_thresholds(tmp_path) -> None:
    r = _run(str(SCRIPTS / "make_fig_hedging.py"), "--source", "cued", "--train-dataset", str(tmp_path),
             "--eval-dataset", str(tmp_path), "--predictions", str(tmp_path / "x.parquet"))
    assert r.returncode != 0 and "MissingRigFact" in r.stderr


def test_synthetic_outputs_never_under_figures() -> None:
    with pytest.raises(ValueError):
        output_dir("synthetic", FIGURES / "x", FIGURES)
    with pytest.raises(ValueError):
        output_dir("synthetic", None, FIGURES)
    assert output_dir("openh", None, FIGURES) == FIGURES


def test_leakage_table_synthetic(tmp_path) -> None:
    r = _run(str(SCRIPTS / "make_table_leakage.py"), "--source", "synthetic", "--out", str(tmp_path),
             "--n-perm", "100")
    assert r.returncode == 0, r.stderr
    res = json.loads((tmp_path / "leakage_synthetic.json").read_text())
    assert res["runs"]["synthetic_leak_off"]["permutation"]["p_value"] > 0.2
    assert res["runs"]["synthetic_leak_on"]["permutation"]["p_value"] < 0.02
    assert res["runs"]["synthetic_leak_on"]["permutation_remaining"]["p_value"] < 0.02
    assert res["provenance"]["config_hash"]


def test_leakage_on_threshold_segmentation() -> None:
    from usintent.data.segmentation import SegParams, segment_threshold
    from usintent.leakage.classifier import build_dataset

    eps = generate(HoldWorldConfig(image_shape=(16, 16), fps=10.0, seed=7), 6, sessions=2)
    p = SegParams(0.002, 1.0, 0.5, 0.5)
    segs = {e.episode_index: segment_threshold(e, p) for e in eps}
    d = build_dataset(eps, 1.0, 0.5, 60.0, segmentations=segs)
    assert len(d.y) > 50 and 0.05 < d.y.mean() < 0.4


def test_hedging_intervention_panel(tmp_path) -> None:
    from usintent.policies.evaluate import intervention_prediction

    assert intervention_prediction(0.2, 1.0, 0.25) == pytest.approx(0.15)
    rows = tmp_path / "rows.json"
    rows.write_text(json.dumps([{"arm": "DP-1", "p": p, "seed": s, "rate": 0.7 * p}
                                for p in (0.08, 0.33) for s in range(3)]))
    r = _run(str(SCRIPTS / "make_fig_hedging.py"), "--source", "synthetic", "--out", str(tmp_path),
             "--thr", "0.002", "--tau", "0.3", "--intervention-rows", str(rows))
    assert r.returncode == 0, r.stderr
    payload = json.loads((tmp_path / "hedging_oracle-imitator-synthetic.json").read_text())
    assert len(payload["intervention"]) == 6
    assert payload["intervention"][0]["predicted"] == pytest.approx(0.08 * 0.7)
