# BUILD_PLAN.md — milestones, order, acceptance tests

The order follows the deadlines: **RSS Stage 1 on 4 Dec 2026**, which needs preliminary checks 1–4, then the **Stage 2 charter period from 26 Feb to 16 Apr 2027**, which needs closed loop.

Legend: **[cloud]** can be built and tested fully in a Claude Code cloud session. **[lab]** needs the lab PC, the robot, GPUs or private data; Claude Code writes the code, the human runs it.

| Milestone | Target date | Where |
|---|---|---|
| M0 Scaffold + theory | Nov 5 | cloud |
| M1 Synthetic hold-world | Nov 7 | cloud |
| M2 Data layer | Nov 10 | cloud (+ lab for Open-H access if blocked) |
| M3 Metrics + baselines B1–B6 | Nov 12 | cloud |
| M4 Leakage + prelim-check scripts | Nov 16 | cloud code, lab run |
| M5 ROS 2 protocol stack (msgs, scheduler, teleop, recorder, converter) | Nov 18 | cloud code (ROS docker), lab test |
| — Pilot cued session on the robot (human) | Nov 19–23 | lab |
| M6 Intervention sampler + arm training configs | Nov 25 | cloud code, lab GPU |
| M7 When-head | Nov 28 | cloud |
| — Insert preliminary results into the paper, submit | Dec 4 | — |
| M8 Policy server + open-loop eval on cued data | Jan 2027 | cloud code, lab |
| M9 Control stack (mock hardware / URSim only) | Jan–Feb 2027 | cloud code (docker), lab sim |
| M10 Closed-loop harness + paper output scripts | Feb 2027 | cloud code, lab |

---

## M0 — Scaffold + theory [cloud]

Build: repo layout from `CLAUDE.md`; `pyproject.toml`; ruff, mypy and pytest config; CI workflow (GitHub Actions: lint, type-check, tests); `usintent/theory/` per SPEC §2; `scripts/make_fig_teaser_c.py`.
Acceptance:
- All golden values in SPEC §2 reproduced (±0.02), and closed forms match Monte Carlo.
- `make_fig_teaser_c.py` writes a PNG/PDF plus a JSON of values that match `reference/rss_predictions.py`.
- CI is green.

## M1 — Synthetic hold-world [cloud]

Build: `usintent/sim/holdworld.py` per SPEC §3, writing LeRobot-format episodes plus the per-hold sidecar table.
Acceptance:
- On 10k holds, the departure-containing chunk fraction equals π̄ (±0.01) for both distributions.
- The threshold-crossing rate matches π̄_τ for the generator's τ.
- The leak flag changes nothing in the image when off (test: a classifier on frames predicts `ends_within(H)` at chance).
- Deterministic given a seed.

## M2 — Data layer [cloud]

Build: loaders, chunk sampler with metadata, phase segmentation (both modes), session/day splits, ground-truth passthrough (SPEC §4), segmentation-sensitivity script.
Acceptance:
- Passthrough error < 1e-6 on synthetic data and on one real Open-H episode.
- Split leakage test fails loudly if a session crosses splits.
- On synthetic data, segmentation recovers ground-truth holds with F1 > 0.95.
- Prints Open-H robotic-ultrasound episode counts (to confirm the paper's numbers).

## M3 — Metrics + baselines [cloud]

Build: SPEC §5 metrics and SPEC §6 baselines B1–B6.
Acceptance:
- Each metric has a synthetic test with a known answer.
- On synthetic cued data: B4 timing error = `bound_median` (±0.05 s); B1 hold drift = 0; B2 error during sweeps is lower than B1's.
- B3 detection lag is reported.
- B5 recovers synthetic out-of-plane motion within the tolerance of its calibration.
- B6 forces zero output whenever the scheduler mode is HOLD.

## M4 — Leakage + preliminary-check scripts [cloud code, lab run]

Build: SPEC §7 leakage classifier; `make_table_phase_ratio.py` (prelim check 1); `make_fig_hedging.py` in "existing model" mode, which takes a fine-tuned GR00T-H checkpoint's open-loop predictions on Open-H holds and compares the measured departure-like rate with the training-data departure fraction (prelim check 2).
Acceptance:
- Leakage on synthetic data: no improvement with the leak off (permutation p > 0.2); clear improvement with it on (p < 0.01).
- Scripts run end to end on synthetic data in the cloud and are ready for the lab run on Open-H plus the existing checkpoint.
- Outputs carry the git and config hashes.

## M5 — ROS 2 protocol stack [cloud code, lab test]

Build: `us_intent_msgs`, `us_intent_scheduler`, `us_intent_teleop`, `us_intent_recorder` + bag→LeRobot converter (SPEC §12, INTERFACES). Use a `ros:<distro>` Docker image in the cloud; the distro comes from `configs/rig.yaml` (ask the human if null).
Acceptance:
- Scheduler sampling is reproducible from its seed, and the empirical D histogram matches the configured distribution (KS test).
- Teleop publishes key events, and motion commands are gated on key press (tested against a mock robot node).
- Converter round trip: a synthetic bag (generated from holdworld) → LeRobot → passthrough check passes; B-mode bytes are identical; timestamps are monotonic.
- A dry-run launch prints the session config and waits for confirmation.

## M6 — Intervention + training configs [cloud code, lab GPU]

Build: SPEC §8 sampler; arm configs (SPEC §10) for GR00T-H (A1, A2, A3, A5, A6) and LeRobot DP (DP-1, DP-5); launch scripts that train N seeds per arm and log seed, git hash and data hash; language-check script; sampler-control inference flags.
Acceptance:
- Realized departure fraction within ±0.01 of target in one epoch.
- Every arm config validates against the installed GR00T/LeRobot (a dry-run with 10 steps on synthetic data succeeds).
- Intent-blind arms assert no mode input.
- Documentation states exactly where the mode embedding enters GR00T-H's action head.

## M7 — When-head [cloud]

Build: SPEC §9 (features, survival model, commit rule, image share, confidence implementations with runtime report).
Acceptance:
- Synthetic, leak off: share_img < 0.05, and timing error equals `bound_median` (±0.1 s).
- Synthetic, leak on: share_img > 0.2, and timing error below the bound.
- Censoring handled (test with truncated episodes).
- The confidence-map runtime per frame is reported.

## M8 — Policy server + open-loop evaluation [cloud code, lab]

Build: `us_intent_policy_server` with intent gating; offline evaluation harness that runs every arm × seed × sampler setting on held-out cued sessions and writes a results table; `make_fig_rate_vs_e.py`; intervention analysis.
Acceptance:
- Gating tests: blind arms never receive the scheduler mode.
- The harness reproduces B4 = bound on cued data.
- Every policy result is preceded by a passing passthrough check.

## M9 — Control stack, simulation only [cloud code (docker), lab sim]

Build: `us_intent_control` per SPEC §12 and `docs/SAFETY.md`: hybrid force/impedance controller in the tool frame, force set-point switched by m̂, CBF/limit filter with intervention logging, watchdog, fault latch. Mock-hardware and URSim launch files; a real-robot launch file that is gated by flag and confirmation.
Acceptance:
- Refuses to start with any null safety value.
- Fault latch triggers on stale inputs, NaN, or a lost F/T stream (tests).
- The filter never outputs a command exceeding limits (property-based tests).
- Step responses logged in URSim.
- **No physical-robot run by Claude.**

## M10 — Closed-loop harness + paper outputs [cloud code, lab]

Build: trial runner (fixed station sequences, scheduler-drawn holds, pairing across arms), closed-loop metrics, cluster bootstrap, `make_table_closed_loop.py`, and the remaining `make_*` scripts.
Acceptance:
- On a mock robot with a synthetic policy, the timing errors of the scripted policies (memoryless hedger, calibrated imitator, median rule) reproduce Fig. 1c within Monte-Carlo error.
