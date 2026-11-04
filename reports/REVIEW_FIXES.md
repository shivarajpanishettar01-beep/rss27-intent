# Review fixes (after reading the Stage-1 extended abstract, 2026-11-04)

Changes made to bring the code in line with the paper (main_clean.pdf), plus the audit's high
and medium findings. Where you had not chosen yet, I took my recommended option; each is listed
under **Decisions** so you can reverse it.

## 1. Intent = operator key event (paper: intent-aware arms are "limited only by latency")
- A5, DP-5 and B6 now receive the **operator's key-event mode** (`action.contact_cmd`: HOLD from
  arrival until the key press, then the exit mode), not the scheduler cue. The cue precedes the
  key by the reaction time, so with the cue A5 could leave before D.
- `EvalContext.intent_mode`, DP-5 training targets, `PolicyServer.set_intent_mode`; the policy
  server subscribes to `/protocol/mode_key` (arrival + key events) instead of `/protocol/mode_cue`.
- Test: between cue and key the scheduler mode is already the exit mode, but the intent input is
  still HOLD.

## 2. The policy's contact command m̂ and "departure-like"
- **Departure-like** = tangential displacement > thr **or** m̂ ∈ {LIFT, RECONTACT}. The action has
  no probe-axis component (the force controller owns z), so a spurious lift during a hold can only
  appear as m̂ = LIFT. m̂ = SWEEP without motion does not move the robot and is not counted.
  Used everywhere: open-loop rates, hold timing, hedging summary, offline eval.
- **DP contact head**: the DP action is now 9-D per step = 5-D displacement + one-hot operator
  mode at that future step; m̂ = argmax of the first step. The threshold heuristic is gone.
- **GR00T contact head**: second action key `contact_cmd` (one-hot), same decoding.

## 3. Scheduler randomises what the paper says it does
`plan_session` now draws station order with back-tracking, the exit mode (SWEEP / LIFT = lift-off
and re-contact timing), the sweep speed level and the probe tilt/twist per station, plus the hold
durations. New `protocol.yaml` keys (all null until you set them): `preset_stations`,
`p_backtrack`, `p_lift`, `sweep_speeds_mps`, `tilt_twist_range_deg`.
`TeleopGate` follows the plan: only the cued key is accepted (a wrong key is logged), and the one
released motion goes to the planned next station with the planned speed and tilt/twist. The
teleop node rebuilds the same plan from the session parameters.

## 4. Threshold and τ measurement (prelim check 4)
`usintent/metrics/calibration.py` + `scripts/run_measure_threshold_tau.py`:
- **thr** = margin (1.5) × the 99.9th percentile of the recorded jitter in hold chunks without a
  departure (zero-motion commands executed by the controller);
- **τ** = time from key press until the recorded displacement crosses thr, overall and per sweep
  speed. `tau_s` = the mean over speed levels, exact for π̄_τ because it is linear in τ.
- Synthetic test: recovers the generator's τ per speed within 2.5 frames.
- The script prints the values; **you** copy them into `metrics.yaml` (code never writes configs).
- Supporting changes: the holds table now carries `sweep_speed_mps`, the recorder logs
  `/teleop/motion_command`, and the hold-world has optional controller jitter.

## 5. Camera, phantom_id, explicit configs, split enforcement (audit high + medium)
- **Probe camera**: Episode field → LeRobot `observation.camera` → MCAP converter (rgb8,
  sample-and-hold at B-mode stamps) → DP (second image key) and GR00T (`camera` video key).
  `rig.yaml topics.camera` must be a topic or `"none"`; null refuses.
- **`phantom_id`** in Episode, LeRobot, `SessionInfo.msg`, the scheduler/teleop nodes, the dry-run
  script and the Open-H mapping. This supports the vascular vs central-line neck phantom.
- **`segmentation.contact_source`**: `force` | `motion`, explicit (was null = motion).
- **`control.yaml: rate_hz`**: the control period now comes from config and must be shorter than
  the watchdog period (was watchdog/2).
- **Passthrough inside `hold_timing`** and in `make_fig_rate_vs_e.py --synthetic`.
- **Train/test disjointness enforced** (`splits.assert_disjoint`) in `run_offline_eval.py`,
  `make_fig_hedging.py` and `run_whenhead.py`. Synthetic session/day ids now include the seed
  (`syn<seed>-s<j>`).

## 6. H3(a) as worded, and closed-loop D
- Leakage: remaining-hold-time **regression** (ridge, gradient boosting; MAE) with the same
  e-binned permutation test, next to the binary test. Both appear in `make_table_leakage.py`.
  Synthetic: leak off p = 1.0; leak on MAE 1.61 → 1.42 s.
- Trial runner: `run_paired(..., reaction_s=...)` makes D = sampled + R, the same D for every arm.
  On the rig, D is the operator's key press in each closed-loop trial.

## 7. GR00T-H is unblocked
Isaac-GR00T turned out to be readable through the session's git proxy (anonymous public clone).
I read commit **c56f541** (model package `gr00t_n1d7`) and implemented
`usintent/policies/gr00t_adapter.py`; the module docstring quotes every API used. Key facts from
the source:
- **Data format.** Fine-tuning needs GR00T-flavoured **LeRobot v2** (`meta/modality.json`, MP4
  videos), not v3. `export_gr00t_dataset` writes it from our v3 archive. B-mode is encoded H.264
  crf 0 in full-range `yuvj420p`, which is **byte-exact for grey frames** (tested).
- **Action representation.** GR00T's EEF/RELATIVE actions need the pose as a state input, which
  would give A1/A2 proprioception. I use our per-step tool-frame displacement as a NON_EEF /
  ABSOLUTE key `tool_step` and compose the chunk in our decode path. A GR00T-path passthrough
  (`gr00t_passthrough`) checks zero error before any GR00T score.
- **Where A5's mode enters.** As a state key `mode` (one-hot of the operator key-event mode). It is
  embedded by the action head's `state_encoder` (a learned per-embodiment MLP) and concatenated
  with the noisy-action tokens (`sa_embs = torch.cat((state_features, action_features), dim=1)`)
  in the flow-matching DiT.
- **State dropout.** GR00T drops state features during training (`state_dropout_prob`: 0.8 model
  default, 0.2 via `launch_finetune`). That would hide A5's mode and A3/A6's proprioception, so
  every arm is launched with `--state-dropout-prob 0.0` (shared schedule).
- **Seed.** `launch_finetune.py` does not expose it (`config.data.seed`, default 42).
  `scripts/gr00t_finetune_seeded.py` sets it and runs GR00T's own launcher unchanged.
- **A7** = an A5-type model whose mode input is the when-head's committed m̂
  (`usintent.whenhead.commit.WhenHeadPolicy`). A7 itself never receives intent, and output is
  zero while HOLD is committed.
- **Instruction-swap check (prelim check 3)**: `policies.evaluate.instruction_swap_effect` +
  `scripts/run_language_check.py` (`--language-mode` export: the instruction encodes the mode).
- **Verified here**: the generated `modality_config.py` executes against GR00T's real
  `register_modality_config`/types (test marked `lab`; run with `GR00T_ROOT=<clone>`: passed).
- **Not verified here**: training and inference. GR00T pins numpy 1.26, torch 2.9 and flash-attn,
  and needs a GPU; that is a lab run.
- The training-data **intervention is DP-only**: GR00T's sharded episode sampler takes no
  per-chunk weights.

## Decisions (reversible)
1. Intent input = operator key event (your item 1, option a).
2. Departure-like includes m̂ ∈ {LIFT, RECONTACT}; SWEEP without motion is not counted.
3. Controller during collection: I could not change this in code. **Recommendation**: collect
   demonstrations with the same M9 controller (after mock + URSim validation) that runs closed
   loop, or state the difference in the paper. This moves controller validation before collection.
4. Neck phantom: `phantom_id` added; the analyses can stratify by it. My suggestion stands: the
   vascular phantom carries H1–H4; the central-line phantom provides natural, uncued holds for the
   when-head's image share and the leakage classifier.
5. GR00T `state_dropout_prob = 0` for all arms; `tool_step` NON_EEF/ABSOLUTE action; A1/A2 get a
   constant `bias` state (GR00T requires a state key).
6. thr = 1.5 × q99.9 of hold jitter (margin and quantile are script flags).

## Needs you
- `protocol.yaml`: preset_stations, p_backtrack, p_lift, sweep_speeds_mps, tilt_twist_range_deg,
  stations_per_episode, audio_cue_command.
- `rig.yaml topics.camera` (topic or "none"); `data.yaml segmentation.contact_source`,
  `openh.phantom_id`; `control.yaml rate_hz`.
- GR00T: the base checkpoint id/path in `configs/arms/A*.yaml: checkpoint` (paper: GR00T-H N1.7).
  Confirm it loads with Isaac-GR00T at c56f541 or tell me the commit you use.
- Whether to report the GR00T intervention at all (DP-only now).

## Test summary
```
180 passed, 2 deselected in 1005.86s (0:16:45)
```
(+ `GR00T_ROOT=<Isaac-GR00T clone> pytest -m lab tests/test_gr00t_adapter.py`: 1 passed.)
check_structure OK; ruff clean; mypy 2.4.1 clean.
