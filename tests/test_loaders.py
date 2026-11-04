"""Loaders, dataset inventory and the Open-H mapping (SPEC section 4)."""
import numpy as np
import pytest

from usintent.config import MissingRigFact, load
from usintent.data.loaders import dataset_info, load_dataset, load_openh, write_lerobot
from usintent.sim.holdworld import HoldWorldConfig, generate


@pytest.fixture(scope="module")
def ds_root(tmp_path_factory):
    eps = generate(HoldWorldConfig(image_shape=(8, 8), n_stations=(2, 2), tool_yaw_rad=0.3), 2)
    return eps, write_lerobot(eps, tmp_path_factory.mktemp("d") / "ds")


def test_dataset_info_counts(ds_root) -> None:
    eps, root = ds_root
    info = dataset_info(root)
    assert info["codebase_version"] == "v3.0"
    assert info["total_episodes"] == 2 and info["total_frames"] == sum(len(e) for e in eps)
    assert len(load_dataset(root)) == 2


def test_openh_mapping_refuses_nulls(ds_root) -> None:
    with pytest.raises(MissingRigFact):
        load_openh(ds_root[1], load("data")["openh"])


def test_openh_mapping_on_a_lerobot_dataset(ds_root) -> None:
    """Mapping logic exercised on our own (synthetic) dataset standing in for Open-H."""
    eps, root = ds_root
    mapping = {"pose_key": "observation.state", "pos_slice": [0, 3], "rot_slice": [3, 6],
               "rot_format": "rotvec", "bmode_key": "observation.bmode",
               "wrench_key": "observation.wrench", "session_key": "session_id", "day_key": "day"}
    got = load_openh(root, mapping, repo_id="local/usintent")
    assert len(got) == 2
    assert np.allclose(got[0].state[:, :6], eps[0].state[:, :6], atol=1e-6)
    assert np.array_equal(got[0].bmode, eps[0].bmode)
    assert got[0].session_id == eps[0].session_id and np.all(got[0].hold_id == -1)
