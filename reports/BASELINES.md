# External comparison families: ACT, SmolVLA, pi0.5, US-VLA

Goal: show that the hedging the paper predicts appears in published policy families other than
our GR00T-H and DP arms, trained on the same data the same way. These arms are comparisons,
not new methods: every model is used as released by LeRobot 0.6.1, without changes to its code.

## What was built
| Arm | Family | Observation (vs A1 / DP-1) | Intent | How it is trained | Inference noise |
|---|---|---|---|---|---|
| ACT-1 | LeRobot ACT | same | blind | our loop, DP schedule (`usintent/policies/act_adapter.py`) | none (zero VAE latent) |
| ACT-5 | LeRobot ACT | same + mode one-hot as ACT's env-state token | aware | our loop | none |
| SmolVLA-1 | LeRobot SmolVLA from `lerobot/smolvla_base` | same | blind | `lerobot-train` (`usintent/policies/lerobot_vla_adapter.py`) | seeded / unseeded, 4 / 16 flow steps |
| PI05-1 | LeRobot pi0.5 from `lerobot/pi05_base` | same | blind | `lerobot-train` | seeded / unseeded, 4 / 16 flow steps |
| USVLA-1 | US-VLA (`pi05_ur_usfm_fusion`, VMVLab/US-VLA @ dde576a) | same + TCP pose | blind | its own trainer, own env (`scripts/usvla_run.py`) | seeded / unseeded, 4 / 16 flow steps (per request) |

Wiring: `configs/arms/*.yaml`, `usintent/layout.py ARMS`, `INTENT_AWARE_ARMS += ACT-5`,
`scripts/run_train_arm.py` (family dispatch), `usintent/policies/evaluate.py load_policy`,
`scripts/run_offline_eval.py` (VLA checkpoint + passthrough), SPEC §10, LAYOUT, `pyproject` extra `vla`.
Lab steps: `scripts/LAB_RUN_BASELINES.md`.

## Tests (tests/test_external_baselines.py, 16 tests, all pass on CPU)
- Arm configs: blind arms have no mode input, ACT-5 is in `INTENT_AWARE_ARMS`, every external arm
  sees the same observation set as DP-1; VLA configs refuse to run without `rename_map` and the
  shared schedule.
- ACT: 10-step dry run of the real LeRobot ACT on synthetic hold-world data for ACT-1 and ACT-5;
  predictions are deterministic; intent gating (blind arm refuses a mode, aware arm requires one);
  the mode changes ACT-5's output; ACT and DP train on byte-identical targets; `model.pt`
  round-trips through `load_policy`.
- VLA (no weights needed): decode passthrough exact to float32 (< 1e-6); a one-step shift of the
  action rows **fails** the passthrough; the exported LeRobot v3.0 copy read back with
  **LeRobot's own loader and `delta_timestamps`** reproduces our chunks (max error 2.4e-9 on the
  dry run), modes and images; train-command flags; observation format.
- Training-script dry runs: ACT-1, ACT-5 train 10 steps; SmolVLA-1, PI05-1 export, pass the
  loader passthrough and write the command.
- `lerobot-train` accepted every generated flag and loaded our exported dataset ("Creating
  dataset"); it stopped only when building the model, which needs the `smolvla` / `pi` extras and
  the weight download (lab machine). Checked with `--policy.type` in place of `--policy.path`
  because the base checkpoints cannot be downloaded here.

## US-VLA (USVLA-1)
Code: `usintent/policies/usvla_adapter.py` (our env), `scripts/usvla_run.py` (US-VLA env:
`build-dataset`, `train`, `serve`), `configs/arms/USVLA-1.yaml`, `tests/test_usvla_adapter.py`.
Read from the US-VLA source, not its README alone (see the adapter docstring).

Checks (10 tests):
- Actions = next absolute TCP pose; decoding against the current pose reproduces the
  ground-truth tool-frame chunk, including changing orientation (1e-12); decode passthrough on
  hold-world data 6.9e-9; storing the current instead of the next pose **fails** it.
- **LeRobot v2.1 build with the LeRobot commit US-VLA pins** (0cf8648, with its locked
  `datasets` 3.6.0 / `huggingface-hub` 0.32.3 / `pyarrow` 20.0.0), read back with openpi's
  `delta_timestamps`: actions, states and B-mode frames identical (max error 0); a second call
  only re-checks.
- **Our client against US-VLA's real `WebsocketPolicyServer`** wrapping `SeededPolicy`
  around a stand-in policy: metadata check (refuses a server for another arm/seed/run),
  `num_steps` reaches the policy, fixed noise (1, horizon, 32) is identical across calls and
  gives identical chunks.
- Not executable here (needs JAX/openpi and weights): `build_config`, norm stats, their trainer,
  `create_trained_policy`. Every config field `build_config` replaces was checked to exist in
  their frozen dataclasses (`TrainConfig`, `LeRobotURDataConfig`, `Pi0Config`,
  `CosineDecaySchedule`).

Decisions:
8. **US-VLA gets the TCP pose.** It predicts absolute TCP poses (its design), which needs the
   current pose; removing it would change the model. Pose carries no intent. Report it as the
   one input difference from A1.
9. **No contact output.** Executed as SWEEP (contact kept by our force controller, tangential
   motion as predicted), the controller's actual behaviour for a pose-only policy; its
   probe-axis prediction is dropped by the same `policy_target` as every other arm.
10. **Side camera blank.** US-VLA has three image slots, all masked True (no missing-camera
   path in `ur_policy.py`); our probe camera goes to `wrist_rgb`, `base_rgb` is black. If the rig
   has no camera, `wrist_rgb` is black too. State it.
11. **Schedule:** their cosine schedule (1000 warm-up steps, flat afterwards since
   decay_lr = peak_lr) with our shared lr as both; LoRA on gemma-2b / 300m as released.
   If the shared step budget is small, the 1000 warm-up steps are a large fraction: decide.
12. **Ablations not added:** the README's no-ultrasound / no-USFM configs are not in the
   released `config.py`. Building them ourselves would make them our variants, not theirs.

## Decisions (simplest option; CLAUDE.md working style)
1. **ACT-5 has no B-mode history.** LeRobot's `ACTConfig.__post_init__` rejects `n_obs_steps != 1`.
   ACT-5 is therefore "A1 + mode", not "A2 + mode" like A5/DP-5. Its intent effect is still
   measurable against ACT-1 (same observation, mode added).
2. **ACT trains in our loop, the VLAs in `lerobot-train`.** ACT is small and has no language or
   tokenizer, so it reuses DP's targets, normalisation and Adam loop (differences from DP are the
   model only). SmolVLA and pi0.5 depend on LeRobot's tokenizer and processor pipelines and on
   released weights, so the defensible baseline is the authors' own fine-tuning recipe.
   Consequence: the VLAs use LeRobot's policy preset (AdamW + warm-up + cosine, decay set to span
   the run) with our shared peak lr, steps, batch size and seed, not plain Adam. State this in
   the paper.
3. **VLA arms are intent-blind only.** Giving a VLA the mode would mean either a language-encoded
   mode (already covered by the GR00T language check) or changing the model. The comparison the
   paper needs is "blind published families hedge as predicted".
4. **VLA action = GR00T action representation** (per-frame tool step + next-mode one-hot), so the
   VLA chunk is decoded by the same composition as the GR00T arm and checked by the same rule-4
   logic, now also through LeRobot's loader.
5. **Constant zero state of size 2 for blind VLA arms.** Found by the loader test: LeRobot stores a
   shape-(1,) feature as a scalar and returns it with no feature axis, which the VLA state
   projections would mis-read. pi0.5 requires a state input.
6. **ACT backbone from scratch** (`pretrained_backbone: null`), matching DP's
   `pretrained_backbone_weights=None`. LeRobot's ACT default is ImageNet ResNet-18; switch both
   or neither.
7. **Arm names** `ACT-1/ACT-5/SmolVLA-1/PI05-1` follow the DP naming (suffix = which A-arm it
   mirrors).

## Open questions / needs a human
- `rename_map` for both VLA checkpoints: read each base checkpoint's `config.json` at the lab.
- Shared `schedule` values (still null for every arm).
- pi0.5 GPU memory: full fine-tune vs `train_expert_only` / bf16 must be decided on the lab GPU and
  reported.
- **Pre-existing, not changed here:** `scripts/run_train_arm.py` builds DP specs without
  `camera_shape`, so DP-1/DP-5 (and now ACT) train **without the probe camera** even though their
  configs list `camera`, while the GR00T arms use it when `rig.yaml topics.camera` is set. For a
  fair comparison, either pass the camera to DP/ACT or drop it from every arm. Needs a decision.
- ForceU-VLA cannot be added: github.com/VMVLab/ForceU-VLA (commit 026842f, 2026-08-16) holds
  only a README, a figure and videos, no code. Position against it in the text; ask the authors.
- US-VLA: download `USFM_latest.pth` (their README) and check that `pi05_base` loads from
  `gs://openpi-assets` on the lab machine; first-run check in LAB_RUN_BASELINES.md §3b.
