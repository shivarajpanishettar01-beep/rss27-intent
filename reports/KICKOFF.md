# Kickoff notes (session 1, 2026-11-04)

## (a) Project in five lines
1. RSS 2027 paper: part of *when* a sonographer leaves a probe hold is not observable from images/state, so imitation-learned policies must hedge.
2. Theory (Eqs. 1–3): intent-blind timing error is bounded below by E|D − med(D)|; blind policies emit "departure-like" motion in a fraction π̄ ≈ E[min(H,D)]/E[D] of hold chunks.
3. Test: a cued-hold protocol on a UR7e (scheduler draws D from U[2,10] or 2+Exp(4)), GR00T-H and Diffusion-Policy arms that differ in what they observe (A1–A7, DP-1/5), and classical baselines B1–B6.
4. A leakage classifier asks whether observations predict hold end beyond elapsed time; a two-source when-head (image hazard + internal clock) splits the share that is visible.
5. Everything is first validated on a synthetic hold-world with known answers, then run on Open-H and our own cued recordings by the human in the lab.

## (b) Plan per milestone
| M | What I build | Cloud-completable? |
|---|---|---|
| M0 | Theory closed forms + MC, `cdf`/Laplace on distributions, `provenance.py`, Fig. 1c script (PNG/PDF/JSON vs reference), CI installs ML extras | yes |
| M1 | `sim/holdworld.py`: episodes in the dataset schema + per-hold sidecar, min-jerk kinematics, speckle B-mode, optional adequacy leak, LeRobot v3.0 writer | yes |
| M2 | Loaders (in-memory + LeRobot v3.0), chunk sampler with metadata, cued + threshold segmentation, session/day splits, passthrough, sensitivity script | yes, except the real Open-H episode (HF blocked → lab) |
| M3 | Open-loop metrics, vessel segmentation, B1–B6 (B3 IMM with detection lag, B5 speckle decorrelation) | yes |
| M4 | Leakage classifier + permutation test; `make_table_phase_ratio.py`, `make_fig_hedging.py`; `LAB_RUN_M4.md` | code yes; Open-H + checkpoint run → lab |
| M5 | Pure-Python protocol logic in `usintent` + thin rclpy nodes, msgs, bag→LeRobot converter, dry-run launch; `LAB_RUN_M5.md` | logic yes; ROS build blocked (distro null, no Docker daemon) |
| M6 | Intervention sampler, arm configs, training launchers, language check; `LAB_RUN_M6.md` | sampler yes; GR00T blocked (GitHub unreachable), LeRobot DP from PyPI source |
| M7 | When-head: features, Karamalis + proxy confidence (runtime measured), survival model, commit rule, image share | yes |
| M8 | Policy server logic + node, offline eval harness, `make_fig_rate_vs_e.py`, intervention analysis | code yes; real eval → lab |
| M9 | Controller + CBF/limit filter + watchdog/fault latch (pure Python, tested on a mock plant); launch files gated per SAFETY.md | code yes; URSim → lab |
| M10 | Trial runner, closed-loop metrics, cluster bootstrap, remaining `make_*` scripts | yes on mock |

## (c) Ambiguities and unknown rig facts (default decision or "needs human")
| # | Item | Decision |
|---|---|---|
| 1 | `ros_distro` null | **needs human** (blocks the M5 colcon build; pure-Python logic built anyway) |
| 2 | B-mode topic/type/rate/timestamp source, F/T source/topic/frame, TCP pose topic, camera presence | **needs human**; stay null in `configs/rig.yaml`, read via `require()` |
| 3 | `departure_threshold_m`, `tau_s`, `force_band_n` (metrics.yaml) | **needs human** (measure). Metrics take them as explicit arguments; synthetic tests pass the generator's own synthetic values |
| 4 | Segmentation thresholds `v_hold`, `T_hold_min`, `T_onset` | **needs human** for Open-H; synthetic tests pass explicit values |
| 5 | When-head `bin_width_s` (= policy period, rig rate null), `coupling_threshold` | **needs human**; synthetic tests pass explicit values |
| 6 | All `safety.yaml` limits | **needs human** (never defaulted) |
| 7 | GR00T-H checkpoint N1.6 vs N1.7; GPU | **needs human** |
| 8 | D definition in the synthetic world | holdworld samples D (start→key) from p_D and sets cue = key − R. On the rig the scheduler samples the *cue* offset, so real D = sampled_D + R: analysis must use the empirical D from key times. **Flag for human**: the paper's p_D predictions should be stated for D (incl. reaction) or for sampled_D. |
| 9 | Action space | 5-D tool-frame displacement (x, y, rx, ry, rz), as in `ActionChunk.msg`; z is owned by the force controller. A chunk is the cumulative displacement from the chunk-start pose, in the chunk-start tool frame |
| 10 | LIFT exits vs a 5-D action | In holdworld, every exit starts with the same tangential min-jerk ramp at key time; LIFT exits additionally raise z, drop the wrench and darken the image (decoupling). So τ is shared by exit types |
| 11 | Back-tracking holds | sidecar gets a `backtrack` bool column (needed to exclude them from the direction test) |
| 12 | `pibar_tau` general form | For D < H the chunk count is E[min(D, H−τ)], not E[max(0, min(H,D)−τ)]. Identical when D ≥ H a.s. (both protocol distributions, H = 1 s). Kept the spec form; noted |
| 13 | B3 Markov transition matrix "from config" (no key exists) | constructor argument; scripts estimate it from training-split mode sequences |
| 14 | B5 calibration σ | new key `rig.yaml: calibration.speckle_sigma_m: null` (rig calibration); synthetic uses the generator's σ |
| 15 | Leakage image embedding ("small pretrained CNN, pin it") | Pretrained weights are not downloadable here. Implemented: parameter-free pooled-intensity embedding (cloud default) and torchvision ResNet-18 ImageNet path for the lab. **Human to confirm** which one the paper uses |
| 16 | Split unit "session and day" | grouping unit = day (so a session can never cross); both session and day asserted |
| 17 | DP arms have `seeds: 3` but SPEC asks ≥5 for H2 arms | kept 3 (config is the human's); **flag** |
| 18 | Push target | harness branch `claude/kind-edison-79i4pq` (linear history) and, per your prompt, `build/Mx` |

## External packages and how APIs are confirmed
- **LeRobot**: `lerobot==0.6.1` from PyPI (reachable). Source read from the installed wheel; API quoted in reports.
- **Isaac-GR00T**: GitHub and HuggingFace are blocked by this session's network policy → cannot read the source → GR00T adapter is **blocked** in the cloud; I will not guess its API.
- **UR ROS 2 driver, ROS 2**: no ROS install, no Docker daemon, distro unknown → nodes written against `rclpy` only in thin wrappers, logic tested in pure Python; colcon build is a lab step.
- **micrus**: source not provided → B-mode topic/type stay config values; tests use mock publishers.
- **Open-H**: HuggingFace blocked → counts and the real passthrough episode are lab steps.
