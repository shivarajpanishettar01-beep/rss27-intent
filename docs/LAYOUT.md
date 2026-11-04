# LAYOUT.md — where everything lives and how it is called

This is a contract. `usintent/layout.py` enforces it: `python scripts/check_structure.py` and `tests/test_structure.py` fail if anything here is broken. CI runs both on every push.

## Tree

```
rss27-intent/
├── CLAUDE.md  README.md  pyproject.toml  .gitignore
├── .github/workflows/ci.yml        structure check, lint, types, tests
├── docs/                           SPEC, INTERFACES, BUILD_PLAN, SAFETY, RIG_FACTS, LAYOUT
├── prompts/                        paste-ready prompts per milestone
├── reference/rss_predictions.py    paper numbers (read-only)
├── configs/                        all YAML; unknown rig facts are null
│   ├── rig.yaml  safety.yaml  data.yaml  protocol.yaml  metrics.yaml  whenhead.yaml
│   └── arms/  A1 A2 A3 A5 A6 A7 DP-1 DP-5 (.yaml)
├── usintent/                       the only Python package (pip install -e .)
│   ├── paths.py                    ALL filesystem paths come from here
│   ├── config.py                   load() and require(); raises MissingRigFact on null
│   ├── layout.py                   the checker for this contract
│   ├── theory/     distributions.py  predictions.py          (implemented: example)
│   ├── sim/        holdworld.py                               M1
│   ├── data/       loaders chunks segmentation splits passthrough   M2
│   ├── metrics/    open_loop closed_loop vessel               M3, M10
│   ├── baselines/  b1_zero b2_const_vel b3_imm b4_hazard_optimal b5_speckle b6_stop_on_hold   M3
│   ├── leakage/    classifier                                 M4
│   ├── intervene/  sampler                                    M6
│   ├── whenhead/   features confidence model commit           M7
│   ├── stats/      tests
│   └── policies/   base (intent gating, implemented)  gr00t_adapter  dp_adapter   M6
├── ros2_ws/src/                    ROS 2 packages (colcon build from ros2_ws/)
│   ├── us_intent_msgs/             ament_cmake; msg/ModeEvent SessionInfo ActionChunk FilterEvent
│   ├── us_intent_scheduler/        ament_python; entry point scheduler_node      M5
│   ├── us_intent_teleop/           teleop_node                                   M5
│   ├── us_intent_recorder/         recorder_node + converter                     M5
│   ├── us_intent_policy_server/    policy_server_node                            M8
│   └── us_intent_control/          control_node (SAFETY.md)                      M9
├── scripts/                        thin entry points: check_structure.py, make_*.py, run_*.py
├── tests/                          pytest; mirrors usintent/ (test_<module>.py)
├── reports/                        Mx.md milestone reports, AUDIT.md
├── figures/                        generated figures + JSON (with git hash)
├── data/   (gitignored)            or wherever USINTENT_DATA points
│   ├── synthetic/  openh/  cued/  bags/
└── results/ (gitignored)           or USINTENT_RESULTS; per-run outputs
```

## Calling rules

1. **One package.** All Python logic lives in `usintent/`. Install it once with `pip install -e ".[dev,ml]"`. Import it absolutely: `from usintent.metrics.open_loop import departure_like_rate`. No relative imports across subpackages, and no `sys.path` hacks (the checker rejects them).
2. **Scripts are thin.** `scripts/*.py` parse arguments and call `usintent` functions. They contain no algorithms, so tests cover the logic.
3. **ROS nodes are thin.** Each node in `ros2_ws/src/<pkg>/<pkg>/<node>.py` handles ROS I/O only and imports `usintent` for logic. In the ROS environment, run `pip install -e .` from the repo root before `colcon build`, so nodes can import it.
4. **Paths only from `usintent/paths.py`.** Never hard-code `/home/...`, `/mnt/...` or relative `../data`. Data outside the repo is reached through `USINTENT_DATA` and `USINTENT_RESULTS`.
5. **Configs only through `usintent/config.py`.** Use `load("rig")` and `require(cfg, "topics.bmode")`. A null raises `MissingRigFact`; never catch it to substitute a guess.
6. **Tests mirror modules.** `usintent/metrics/open_loop.py` → `tests/test_open_loop.py`. Mark slow tests `@pytest.mark.slow` and lab-only tests `@pytest.mark.lab`.
7. **Entry-point names are fixed.** ROS console scripts are `<node> = <pkg>.<node>:main`. The checker verifies the file and `main()` exist.
8. **Adding things.** To add a new top-level folder, subpackage, ROS package or arm, update `REQUIRED_*` / `SUBPACKAGES` / `ROS_PACKAGES` / `ARMS` in `usintent/layout.py` **and** this file in the same commit. Never move or rename an existing file without updating every import, plus this file.

## Commands

```bash
pip install -e ".[dev]"            # add ,ml for torch/sklearn
python scripts/check_structure.py  # must print OK before every commit
pytest -m "not slow and not lab"
cd ros2_ws && colcon build && source install/setup.bash   # in a ROS environment
```
