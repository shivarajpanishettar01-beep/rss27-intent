# LAB_RUN_M4 — preliminary checks 1–2 (+ leakage) on Open-H with the existing GR00T-H checkpoint

Lab machine, no robot motion. Prerequisite: `scripts/LAB_RUN_M2.md` steps 1–4 done
(`configs/data.yaml: openh` mapping filled, passthrough passes on one real episode).

```bash
pip install -e ".[dev,ml,lerobot]"          # Python >= 3.12 for lerobot 0.6.1
export USINTENT_DATA=/path/to/data USINTENT_RESULTS=/path/to/results
OPENH=$USINTENT_DATA/openh/<vascular_subset>   # LeRobot v3.0 folder
REPO=<repo_id of that subset>
```

## 0. Fill configs (human; never guessed by code)
- `configs/data.yaml: segmentation.v_hold_mps, t_hold_min_s, t_onset_s` (and optionally `contact_force_n`).
- `configs/metrics.yaml: departure_threshold_m` (zero-motion baseline + measured controller jitter) and `tau_s`.
- `configs/whenhead.yaml: coupling_threshold` (uint8 near-field level separating coupled/uncoupled columns).
Every script refuses to run (`MissingRigFact`) while one of these is null.

## 1. Export the checkpoint's open-loop predictions (human-written inference call)
I could not read Isaac-GR00T's source from the cloud (GitHub is blocked), so I did not write the
GR00T inference call. Use your existing inference code and write the file through our format:

```python
import numpy as np
from usintent.config import load
from usintent.data.chunks import build_sampler
from usintent.data.loaders import load_openh
from usintent.data.passthrough import passthrough_check
from usintent.policies.evaluate import save_predictions

cfg = load("data")
eps = load_openh("<OPENH>", cfg["openh"], repo_id="<REPO>")
sampler = build_sampler(eps, "openh", cfg)
passthrough_check(sampler)                      # must pass before anything is scored
rows = sampler.index                            # or a subset, e.g. sampler.index.iloc[::3]
chunks, m_hat = [], []
for ep_idx, start in zip(rows.episode_index, rows.start):
    # >>> YOUR GR00T-H inference for the observation at (ep_idx, start) <<<
    # must return the chunk as (n, 5): x, y (m), rx, ry, rz (rad), cumulative from the
    # chunk-start pose, in the chunk-start tool frame (same convention as usintent.data.chunks)
    chunk, mh = your_gr00t_predict(ep_idx, start)
    chunks.append(chunk); m_hat.append(mh)
save_predictions("$USINTENT_RESULTS/predictions/gr00t_existing.parquet", rows,
                 np.stack(chunks), m_hat, {"model": "gr00t_existing", "checkpoint": "<path>",
                 "arm_id": "existing", "seed": "<seed>"})
```
**Check the frame conversion of your model's action output against `usintent.data.chunks`
before trusting any number** (run your decoder on the recorded actions and compare with
`policy_target(sampler.chunks6(rows))`; error must be < 1e-6).
Runtime: depends on your GPU; ~1 forward pass per chunk start.

## 2. Prelim check 1 — phase ratio to B2
```bash
python scripts/make_table_phase_ratio.py --source openh --dataset $OPENH --repo-id $REPO \
    --predictions $USINTENT_RESULTS/predictions/gr00t_existing.parquet
```
Expected output: `figures/phase_ratio_gr00t_existing.{json,csv,md}`; a table with one row per
phase (approach, contact_onset, hold, sweep), policy error, B2 error, ratio, ratio without
transition windows, and a "B2≈0" flag (expected on holds). Runtime: minutes (B2 is computed in Python).

## 3. Prelim check 2 — hedging (existing-model mode)
Training data = the data the checkpoint was fine-tuned on; evaluation = held-out Open-H holds
(split by session/day, `usintent.data.splits`).
```bash
python scripts/make_fig_hedging.py --source openh --train-dataset $OPENH_TRAIN \
    --eval-dataset $OPENH_EVAL --repo-id $REPO \
    --predictions $USINTENT_RESULTS/predictions/gr00t_existing.parquet
```
Expected output: `figures/hedging_gr00t_existing.{json,png,pdf}` with the training departure
fraction, the predicted π̄_τ (empirical D of the training holds) and the measured
departure-like rate with a hold-cluster bootstrap CI. Runtime: minutes.

## 4. Leakage on Open-H (H3a)
```bash
python scripts/make_table_leakage.py --source openh --dataset $OPENH --repo-id $REPO
```
Expected: `figures/leakage_openh.{json,md}`; log-loss/AUROC for E, E+ctx, E+ctx+obs (logistic
regression, gradient boosting) and a 1000-permutation p-value. Uses the pooled-intensity image
embedding unless you decide otherwise (see reports/M4.md). Runtime: ~10–60 min (1000 CV refits).

Paste the JSON files back into the next session.

## 5. Prelim check 3 — instruction swap on the existing GR00T-H checkpoint
With your existing inference code, wrap it as `predict(ckpt, instruction, episode_index, start)
-> (n, 5) chunk` and call `usintent.policies.evaluate.instruction_swap_effect(predict, rows,
base_instruction, [other instructions], (ckpt_seed_a, ckpt_seed_b))`. If you have only one seed,
the seed-noise term is missing: report the raw swap effect and say so. For our own arms use
`scripts/run_language_check.py` (needs `--language-mode` runs, LAB_RUN_M6).

## 6. Check 4 helpers (first cued episodes)
```bash
python scripts/run_measure_threshold_tau.py --dataset $USINTENT_DATA/cued/<pilot>   # thr, tau
python scripts/make_table_leakage.py --source cued --dataset $USINTENT_DATA/cued/<pilot>
```
