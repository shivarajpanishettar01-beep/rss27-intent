# AUDIT (prompts/99_audit.md) — 2026-11-04, after M0–M10 (code state at the M10 commit)

Findings ranked by severity. **Nothing was fixed as part of this audit**; each finding has a proposed fix for you to approve.

## Summary
- **No experimental number exists yet.** The only committed paper output is `figures/teaser_c.{json,png,pdf}` (theory, Monte Carlo + closed forms). Every other number in `reports/` is a SYNTHETIC code check or a cloud runtime measurement.
- Passthrough is enforced on every policy-scoring path that decodes actions. Two paths can score without it (finding 4).
- Splits: the split code exists and is tested, but **no script enforces it** on its train/test inputs (finding 1).
- 67 config values are still null (list below). The code reads all of them through `require()`, with three conventions that deserve a decision (findings 2, 3, 5).

## Findings

### HIGH
1. **Train/test session disjointness is not checked by the scripts.** `run_offline_eval.py --train/--test`, `make_fig_hedging.py --train-dataset/--eval-dataset`, `run_whenhead.py --train/--test` and `run_train_arm.py` accept any two datasets. Nothing verifies that no session or day appears in both, so CLAUDE.md rule 5 relies on the operator.
   *Proposed fix*: in each script, build `episode_split_table` for train ∪ test with the split label and call `check_no_leakage` before anything else. Also give `data.yaml: splits.assignment_file` a real file and load splits only through it.

### MEDIUM
2. **Null-as-"absent" semantics for two config keys** instead of `MissingRigFact`. `data.yaml: segmentation.contact_force_n` null means "use the motion proxy for contact" (`segmentation.py:42`), and `rig.yaml: topics.camera` null means "no camera" (`recorder.py:74`, `server.py:29`). Both are documented in the YAML comments, but neither is a "known" rig fact.
   *Proposed fix*: replace them with explicit values (`contact_source: force|motion`, `camera: none|<topic>`), so a forgotten value raises.
3. **Control loop period derived from a safety value.** `control_node.py:40` sets the control timer to `watchdog_period_s / 2`. The control rate is a rig/driver fact.
   *Proposed fix*: add `control.yaml: rate_hz: null` and require `1/rate_hz < watchdog_period_s`.
4. **Two scoring paths without a passthrough check.** `policies.evaluate.hold_timing` can be called directly; in the repo it is only called from `offline_eval` (after passthrough) and from tests. `make_fig_rate_vs_e.py --synthetic` scores ground-truth chunks without a passthrough call; the real path reads `eval_chunks.parquet`, which `offline_eval` wrote after passthrough.
   *Proposed fix*: call `passthrough_check` inside `hold_timing` and in the synthetic branch, or record the passthrough result in `eval_chunks.parquet` and have `make_fig_rate_vs_e.py` refuse rows without it.
5. **Algorithm thresholds with defaults that touch real images**: `metrics/vessel.py` `dark_frac=0.4`, `smooth_px=1.5`; `whenhead.features.coupling_indicator` `near_field_frac=0.15`; `confidence.proxy_confidence` `alpha=1.0`; Karamalis α, β, γ; `ROI_DILATE_PX=2`; recorder `max_clock_skew_s=0.5`. These are method parameters, not rig facts, but none has been validated on real frames (SPEC asks for validation of the vessel segmentation against human labels).
   *Proposed fix*: move them to `metrics.yaml` / `whenhead.yaml` and validate on a labelled subset of pilot frames before any image metric is reported.
6. **Teaser figure config hash is stale.** `figures/teaser_c.json` was made at commit 4073b575 with config hash `18174d5f92eaaad1`. `data.yaml` gained keys afterwards (M2), so a clean re-run gives hash `b9daf932d516d80e`. **The values are identical** (re-run from a clean clone of the M9 commit: `values identical: True`).
   *Proposed fix*: regenerate the figure at the final commit before submission, or hash only the config keys a script reads.

### LOW
7. **Synthetic hold-world defaults changed during the build** (station spacing 2 → 5 cm in M3; the when-head tests use attenuation 150). Every synthetic check is tied to an explicit config in its test, but the reports quote numbers from different defaults. *Proposed fix*: none needed for the paper (synthetic never feeds it); noted for traceability.
8. **`test_full_1000_permutations` (slow) has never completed** in this session (stopped because of CPU contention). *Proposed fix*: run `pytest -m slow` once on the lab machine.
9. **ROS code never built or run** (no ROS / Docker daemon in the cloud): all five nodes, the msgs package and the launch files. *Proposed fix*: LAB_RUN_M5 / M8 / M9.

## 1. Number → producer trace
| Where | Number(s) | Produced by | Data | Hash |
|---|---|---|---|---|
| figures/teaser_c.json, reports/M0.md | Fig. 1c values (2.00/2.67/4.75; 2.77/4.00/5.16; π̄ 0.167 …) | `scripts/make_fig_teaser_c.py` | theory | git 4073b575, config 18174d5f92eaaad1 (finding 6) |
| reports/M1.md | π̄ / π̄_τ matches, AUROC 0.512 / 0.995 | `tests/test_holdworld.py` (+ printout) | synthetic | commit dfe46bb |
| reports/M2.md | segmentation F1 0.962 / 0.963 | `tests/test_segmentation.py` setup | synthetic | ca49d72 |
| reports/M3.md | B4 2.003 s; B2 vs B1 errors; B3 lags | `tests/test_baselines.py`, ad-hoc runs of the same code | synthetic | 158a419 |
| reports/M4.md | leakage gains; phase ratio 2.78; hedging 0.1123 | `tests/test_classifier.py`, `make_*` scripts `--source synthetic` | synthetic | 2d25319 |
| reports/M5.md | KS / round-trip checks | tests | synthetic | b72f0bd |
| reports/M6.md | realised fraction checks | tests | synthetic | 41408db |
| reports/M7.md | share, timing, runtime table | `scripts/run_whenhead.py --source synthetic`, `run_confidence_runtime.py --synthetic` | synthetic; runtime = this cloud CPU | 6d89f6a |
| reports/M8.md | B4 1.894 s | `run_offline_eval.py --synthetic` | synthetic | 254961e |
| reports/M9.md | step response | `tests/test_controller.py` | synthetic mock plant | 4b7252d |
| reports/M10.md | closed-loop Fig. 1c table | `make_table_closed_loop.py --synthetic` | synthetic mock robot | 892ba2b |

Synthetic outputs carry `"data_source": "synthetic"` and a warning field, and `provenance.output_dir` refuses to write them under `figures/` or `results/`.

## 2. Re-running make_* from a clean checkout
`make_fig_teaser_c.py` re-run from a clean clone: identical values (finding 6 for the hash). The other `make_*` scripts have no committed real output; their synthetic modes run in CI on every push (`tests/test_prelim_scripts.py`, `test_server.py`, `test_trials.py`).

## 3. Passthrough before every policy evaluation
Enforced in `evaluate_open_loop`, `offline_eval`, `make_table_phase_ratio.py`, `make_fig_hedging.py`, `run_passthrough.py`, `run_convert_bags.py`. Tests prove that a corrupted decode path raises before scoring. Exceptions: finding 4.

## 4. No session in two splits
No real data exists yet. `check_no_leakage` is tested to fail loudly. Enforcement in scripts is missing: finding 1.

## 5. Null config values (67) and rig facts not marked known
- `rig.yaml` (15): ros_distro; topics.bmode, bmode_type, wrench, wrench_frame, camera, tcp_pose; rates_hz.bmode, wrench, policy; latency_s.image_to_command; tool.tcp_offset; calibration.speckle_sigma_m, speckle_rho0, pixel_spacing_m.
- `safety.yaml` (13): every limit and every sanity range.
- `control.yaml` (7): force_setpoint_n, recontact_setpoint_n, admittance_gain_mps_per_n, position_gain_per_s, lift_speed_mps, cmd_topic, cmd_type.
- `data.yaml` (14): segmentation.v_hold_mps, t_hold_min_s, t_onset_s, contact_force_n; splits.assignment_file; openh.repo_ids, pose_key, pos_slice, rot_slice, rot_format, bmode_key, wrench_key, session_key, day_key.
- `metrics.yaml` (3): departure_threshold_m, tau_s, force_band_n.
- `protocol.yaml` (5): stations_per_episode.min/max, sweep_speeds_mps, tilt_twist_range_deg, audio_cue_command.
- `whenhead.yaml` (2): bin_width_s, coupling_threshold.
- `arms/*.yaml` (4 each): checkpoint, schedule.steps/batch_size/lr.

Code that relies on rig facts not marked `known`: everything above goes through `require()` except findings 2 and 3. The UR7e (user-reported) appears only as the `robot_type` string in the LeRobot metadata. The 5-D action / tool-frame conventions are design decisions (M1 report), not rig facts. The wrench sign convention (compressive positive) is assumed in synthetic data and in the controller: **the real sensor's sign must be checked** (RIG_FACTS: F/T source unknown).
