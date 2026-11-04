"""Layout contract checker. The contract itself is documented in docs/LAYOUT.md.

If you add a top-level folder, subpackage or ROS package, add it here AND in docs/LAYOUT.md.
"""
from __future__ import annotations

import ast
import importlib
import pkgutil
import re
import xml.etree.ElementTree as ET

import yaml

import usintent
from usintent.paths import CONFIGS, REPO_ROOT, ROS2_WS

REQUIRED_DIRS = [
    "usintent", "configs", "configs/arms", "docs", "prompts", "reference", "scripts", "tests",
    "reports", "figures", "data", "results", "ros2_ws/src",
]
REQUIRED_FILES = [
    "CLAUDE.md", "README.md", "pyproject.toml", ".gitignore",
    "docs/SPEC.md", "docs/INTERFACES.md", "docs/BUILD_PLAN.md", "docs/SAFETY.md",
    "docs/RIG_FACTS.md", "docs/LAYOUT.md", "reference/rss_predictions.py",
    "configs/rig.yaml", "configs/safety.yaml", "configs/data.yaml", "configs/protocol.yaml",
    "configs/metrics.yaml", "configs/whenhead.yaml", "configs/control.yaml",
]
SUBPACKAGES = ["theory", "sim", "data", "metrics", "baselines", "leakage", "intervene",
               "whenhead", "stats", "policies", "protocol", "control", "closedloop"]
ROS_PACKAGES = ["us_intent_msgs", "us_intent_scheduler", "us_intent_teleop", "us_intent_recorder",
                "us_intent_policy_server", "us_intent_control"]
ARMS = ["A1", "A2", "A3", "A5", "A6", "A7", "DP-1", "DP-5"]
SAFETY_KEYS = ["max_contact_force_n", "max_tcp_speed_mps", "max_tcp_accel_mps2",
               "max_normal_displacement_m", "workspace_box_base_m", "max_input_age_s",
               "watchdog_period_s"]
FORBIDDEN_IN_CODE = [re.compile(r"sys\.path\.(append|insert)"), re.compile(r"[\"']/home/"),
                     re.compile(r"[\"']/mnt/"), re.compile(r"[\"']C:\\\\")]


def _paths() -> list[str]:
    out = [f"missing dir: {d}" for d in REQUIRED_DIRS if not (REPO_ROOT / d).is_dir()]
    out += [f"missing file: {f}" for f in REQUIRED_FILES if not (REPO_ROOT / f).is_file()]
    for sp in SUBPACKAGES:
        if not (REPO_ROOT / "usintent" / sp / "__init__.py").is_file():
            out.append(f"usintent/{sp} is not a package (no __init__.py)")
    return out


def _imports() -> list[str]:
    out = []
    for mod in pkgutil.walk_packages(usintent.__path__, prefix="usintent."):
        try:
            importlib.import_module(mod.name)
        except ModuleNotFoundError as e:
            # optional heavy deps (torch, sklearn, lerobot, gr00t) may be absent in light envs
            if e.name and e.name.split(".")[0] in {"torch", "sklearn", "lerobot", "gr00t", "rclpy"}:
                continue
            out.append(f"import failed: {mod.name}: {e}")
        except Exception as e:  # noqa: BLE001
            out.append(f"import failed: {mod.name}: {type(e).__name__}: {e}")
    return out


def _configs() -> list[str]:
    out = []
    for f in sorted(CONFIGS.rglob("*.yaml")):
        try:
            data = yaml.safe_load(f.read_text())
            if not isinstance(data, dict):
                out.append(f"{f.relative_to(REPO_ROOT)} is not a mapping")
        except yaml.YAMLError as e:
            out.append(f"bad YAML {f.relative_to(REPO_ROOT)}: {e}")
    for a in ARMS:
        p = CONFIGS / "arms" / f"{a}.yaml"
        if not p.is_file():
            out.append(f"missing arm config {a}")
            continue
        cfg = yaml.safe_load(p.read_text())
        if cfg.get("arm_id") != a:
            out.append(f"arm config {a}: arm_id is {cfg.get('arm_id')!r}")
        if a not in ("A5", "DP-5") and cfg.get("intent_aware"):
            out.append(f"arm {a} must be intent-blind")
    safety = yaml.safe_load((CONFIGS / "safety.yaml").read_text()) or {}
    for k in SAFETY_KEYS:
        if k not in safety:
            out.append(f"safety.yaml missing key {k}")
    return out


def _ros() -> list[str]:
    out = []
    src = ROS2_WS / "src"
    for pkg in ROS_PACKAGES:
        d = src / pkg
        px = d / "package.xml"
        if not px.is_file():
            out.append(f"ROS package {pkg}: no package.xml")
            continue
        name = ET.parse(px).getroot().findtext("name")
        if name != pkg:
            out.append(f"ROS package folder {pkg} declares name {name!r}")
        if pkg == "us_intent_msgs":
            cm = (d / "CMakeLists.txt").read_text()
            for m in re.findall(r'"(msg/[^"]+\.msg)"', cm):
                if not (d / m).is_file():
                    out.append(f"{pkg}: CMakeLists lists {m} but file is missing")
            continue
        if not (d / "resource" / pkg).is_file():
            out.append(f"{pkg}: missing resource/{pkg} marker")
        if not (d / pkg / "__init__.py").is_file():
            out.append(f"{pkg}: python module folder {pkg}/ missing")
        setup = (d / "setup.py").read_text() if (d / "setup.py").is_file() else ""
        if not setup:
            out.append(f"{pkg}: missing setup.py")
        for target in re.findall(r"=\s*([\w\.]+):main", setup):
            mod_path = d / (target.replace(".", "/") + ".py")
            if not mod_path.is_file():
                out.append(f"{pkg}: entry point {target}:main has no file {mod_path.relative_to(REPO_ROOT)}")
            elif "def main" not in mod_path.read_text():
                out.append(f"{pkg}: {target} has no main()")
    return out


def _hygiene() -> list[str]:
    out = []
    for py in list((REPO_ROOT / "usintent").rglob("*.py")) + list((REPO_ROOT / "scripts").rglob("*.py")) \
            + list((ROS2_WS / "src").rglob("*.py")):
        text = py.read_text()
        for pat in FORBIDDEN_IN_CODE:
            if pat.search(text) and py.name != "layout.py":
                out.append(f"{py.relative_to(REPO_ROOT)}: forbidden pattern {pat.pattern} (use usintent.paths)")
        try:
            ast.parse(text)
        except SyntaxError as e:
            out.append(f"syntax error {py.relative_to(REPO_ROOT)}: {e}")
    return out


def check_all() -> list[str]:
    return _paths() + _imports() + _configs() + _ros() + _hygiene()
