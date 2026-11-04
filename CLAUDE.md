# CLAUDE.md — rss27-intent

Project memory for Claude Code. Read this file and `docs/` before doing anything.

## What this repo is

Code for the RSS 2027 paper **"When to Move Is Not in the Image: Intent Observability in Learned Robotic Ultrasound Scanning."** The paper argues that part of when a sonographer leaves a probe hold is invisible to the robot, so imitation-learned policies hedge. It tests this with a cued-hold protocol on a UR7e, GR00T-H and diffusion-policy observation arms, classical baselines, and a two-source **when-head**.

- Folder/path/import contract: `docs/LAYOUT.md`
- Full technical spec: `docs/SPEC.md`
- Interfaces, topics, units, config schema: `docs/INTERFACES.md`
- Build order and acceptance tests: `docs/BUILD_PLAN.md`
- Robot safety rules: `docs/SAFETY.md`
- What is known and unknown about the rig: `docs/RIG_FACTS.md`
- The paper's numeric predictions: `reference/rss_predictions.py`

## Hard rules

1. **Never invent rig facts.** Topic names, frame rates, sensor types, force limits, frames, gains, latencies: if the value is not in `docs/RIG_FACTS.md` marked `known`, read it from config. Configs for unknown values have **no default**, and the code refuses to start until the value is filled. Do not guess "typical" values for this robot.
2. **Never write code that moves the real robot without the gates in `docs/SAFETY.md`.** All motion code must run first against mock hardware or URSim. Real-robot launch files require an explicit `--i-have-read-safety-md` flag and a filled safety config. Claude never runs anything on the physical robot; only the human does.
3. **Tests first for every metric and formula.** Each metric gets a unit test on synthetic data with a known answer. The synthetic hold-world generator (`usintent/sim/holdworld.py`, milestone M1) is the ground truth for the analysis code. If a metric disagrees with `reference/rss_predictions.py` on synthetic data, the metric is wrong until proven otherwise.
4. **Ground-truth passthrough check.** Every evaluation pipeline must pass recorded ground-truth actions through the identical decode/transform path and get zero error before it scores any policy. This caught several silent failures in earlier work.
5. **Splits by recording session and day, never by episode or frame.**
6. **The unit of analysis for a training configuration is the training run (seed), not the episode.** Report every seed as a point.
7. **No numbers go into the paper unless a script in this repo produced them.** Every figure and table is produced by `scripts/make_*.py` from saved results, with the git hash written into the output.
8. **Do not invent APIs of external repos.** For Isaac-GR00T, LeRobot, the UR ROS 2 driver and the user's `micrus` node, read the installed source or docs first. Pin versions in `pyproject.toml` / `package.xml` once confirmed.
9. **Small commits, one milestone at a time.** End every milestone with all tests passing and a short report in `reports/Mx.md` covering: what was built, test results, open questions, anything that needs a human decision.

## Stack

- Python 3.10+, PyTorch, numpy/scipy, pandas, scikit-learn, matplotlib.
- Formatting and linting: ruff. Tests: pytest. Types: mypy on `usintent/` (strict on `theory/` and `metrics/`).
- Data: LeRobot v3.0 dataset format (the public-release target); ROS 2 bags (MCAP) as raw recordings.
- Policies: NVIDIA Isaac-GR00T (GR00T-H fine-tuning), LeRobot Diffusion Policy.
- Robot: ROS 2 (distro from RIG_FACTS), Universal Robots ROS 2 driver, user's `micrus` ROS 2 node for B-mode.

## Repo layout: already scaffolded, do not restructure

The full folder tree, packaging, configs, ROS 2 package skeletons and stub modules already exist. **`docs/LAYOUT.md` is a contract**, enforced by `usintent/layout.py`.

- Fill the stub files in place; keep every path and module name.
- All paths come from `usintent/paths.py`; all configs go through `usintent/config.py`.
- Absolute imports from the single package `usintent`; no `sys.path` hacks; scripts and ROS nodes stay thin and call `usintent`.
- Run `python scripts/check_structure.py` before every commit; it must print OK. CI runs it too.
- `usintent/theory/` and `usintent/policies/base.py` are implemented examples of the conventions (typed, documented, tested). Match their style.

## Working style

- First command of every session: `pip install -e ".[dev]" && python scripts/check_structure.py && pytest -m "not slow and not lab"`. Fix any failure before new work.
- Before coding a milestone, restate its acceptance tests from `docs/BUILD_PLAN.md`, then build to them.
- When a spec detail is ambiguous, choose the simplest option, record it in `reports/Mx.md` under "Decisions", and continue. Stop only for safety questions or missing rig facts.
- Prefer clear code over clever code. Docstrings state units and frames for every array.
