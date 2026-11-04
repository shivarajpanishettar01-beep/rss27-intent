"""M9: refuses to start with any null / out-of-range safety value; real-robot gate."""
import copy

import pytest

from tests.control_fixtures import TEST_SAFETY
from usintent.config import load
from usintent.control.safety import SCALAR_LIMITS, SafetyConfigError, load_limits, preflight


def test_repo_safety_config_refuses() -> None:
    with pytest.raises(SafetyConfigError):
        preflight(load("safety"), i_have_read_safety_md=True, real_robot=False, print_fn=lambda _: None)


@pytest.mark.parametrize("key", [*SCALAR_LIMITS, "workspace_box_base_m"])
def test_each_null_refuses(key) -> None:
    cfg = copy.deepcopy(TEST_SAFETY)
    cfg[key] = None
    with pytest.raises(SafetyConfigError):
        preflight(cfg, True, False, print_fn=lambda _: None)


@pytest.mark.parametrize("key", SCALAR_LIMITS)
def test_null_sanity_range_or_out_of_range_refuses(key) -> None:
    cfg = copy.deepcopy(TEST_SAFETY)
    cfg["sanity"][key] = None
    with pytest.raises(SafetyConfigError):
        preflight(cfg, True, False, print_fn=lambda _: None)
    cfg = copy.deepcopy(TEST_SAFETY)
    cfg[key] = cfg["sanity"][key][1] * 2
    with pytest.raises(SafetyConfigError):
        load_limits(cfg)


def test_real_robot_needs_flag_and_yes() -> None:
    printed = []
    with pytest.raises(SafetyConfigError, match="i_have_read_safety_md"):
        preflight(TEST_SAFETY, False, True, input_fn=lambda _: "yes", print_fn=printed.append)
    with pytest.raises(SafetyConfigError, match="confirm"):
        preflight(TEST_SAFETY, True, True, input_fn=lambda _: "y", print_fn=printed.append)
    lim = preflight(TEST_SAFETY, True, True, input_fn=lambda _: "yes", print_fn=printed.append)
    assert lim.max_contact_force_n == 8.0
    assert any("max_tcp_speed_mps = 0.05" in p for p in printed)       # limits are printed


def test_real_robot_launch_file_requires_flag() -> None:
    from usintent.paths import ROS2_WS

    txt = (ROS2_WS / "src/us_intent_control/launch/control_real_robot.launch.py").read_text()
    assert 'default_value="false"' in txt and "i_have_read_safety_md" in txt and "!= \"true\"" in txt
    for f in ("control_mock.launch.py", "control_ursim.launch.py"):
        assert '"real_robot": False' in (ROS2_WS / "src/us_intent_control/launch" / f).read_text()
