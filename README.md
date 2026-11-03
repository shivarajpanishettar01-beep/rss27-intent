# Build kit for rss27-intent

How to use this with Claude Code (cloud sessions):

1. Create an empty GitHub repo (e.g. `rss27-intent`) and copy this whole folder into it: `CLAUDE.md`, `docs/`, `prompts/`, `reference/`. Commit.
2. Fill in what you know in `docs/RIG_FACTS.md` (at least the ROS 2 distro and the force/torque source). Leave the rest `unknown`.
3. Open a Claude Code cloud session on the repo and paste `prompts/00_kickoff.md`. Read its reply, answer its questions, then say "go".
4. Run one milestone per session: paste `prompts/M0.md`, then `M1.md`, and so on. After each, review `reports/Mx.md` and the diff before merging.
5. When a milestone needs the lab (`LAB_RUN_Mx.md`), run those commands yourself on the lab machine and paste the results into the next session.
6. Before any number goes into the paper, run `prompts/99_audit.md`.

Order and dates: `docs/BUILD_PLAN.md`. M0–M5 and M7 are needed for the 4 Dec preliminary checks; M8–M10 are for Stage 2.

What cloud sessions can't do: reach the robot, the lab PC or (usually) a GPU. Claude Code writes and tests the code against synthetic data, mock hardware and Docker images; you run training and anything touching the rig. If a cloud session can't download Open-H or the GR00T-H checkpoint because of network limits, run those steps from the lab.

Files:
- `CLAUDE.md`: project memory and hard rules (Claude Code reads it automatically)
- `docs/SPEC.md`: what every module computes
- `docs/INTERFACES.md`: topics, dataset fields, config schema
- `docs/BUILD_PLAN.md`: milestones, dates, acceptance tests
- `docs/SAFETY.md`: rules for any code that can move the robot
- `docs/RIG_FACTS.md`: known vs unknown rig facts (you edit this)
- `prompts/`: one paste-ready prompt per milestone, plus kickoff and audit
- `reference/rss_predictions.py`: the paper's numbers; tests must reproduce them
