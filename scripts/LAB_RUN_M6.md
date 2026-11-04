# LAB_RUN_M6 — training runs (lab GPU)

## Status of the two policy families
- **LeRobot Diffusion Policy (DP-1, DP-5): ready.** `usintent/policies/dp_adapter.py` uses the
  installed `lerobot==0.6.1` DiffusionPolicy; a 10-step dry run on synthetic data passes in CI.
- **GR00T-H (A1, A2, A3, A5, A6): blocked.** The cloud session could not reach GitHub or
  HuggingFace, and Isaac-GR00T is not on PyPI, so I could not read its fine-tuning entry point,
  modality JSON or action head. Next session: clone Isaac-GR00T at the commit you used for the
  existing checkpoint into the repo (or paste the relevant files). I will then write the adapter
  and document where the A5 mode embedding enters the action head.

## 0. Decisions needed first
- `configs/arms/*.yaml: schedule.steps, batch_size, lr`: one shared schedule for every arm (SPEC 10).
- Which arm carries the intervention (SPEC 8 says "a trained policy"; I suggest A1, or DP-1 while GR00T is blocked) and the targets: 0.5× and 2× the natural π̄ of the training set.
- DP seeds: the configs say 3, SPEC 10 asks ≥ 5 for arms that test H2.

## 1. Time one run to get the GPU-hour estimate
```bash
pip install -e ".[dev,ml,lerobot,bags]"
python scripts/run_train_arm.py --arm DP-1 --seed 0 --dry-run      # sanity, CPU ok, < 1 min
# after filling the schedule, time 1 % of a real run:
time python scripts/run_train_arm.py --arm DP-1 --seed 0 --train $USINTENT_DATA/cued/<train>
```
GPU hours = (number of runs) × (steps) × (seconds per step) / 3600. I could not measure seconds
per step (no GPU here), so I do not give a number.

## 2. Number of runs (with the configs as they are)
| block | arms | seeds | runs |
|---|---|---|---|
| GR00T-H observation arms (blocked) | A1 A2 A3 A5 A6 | 5 | 25 |
| DP arms | DP-1 DP-5 | 3 | 6 |
| intervention (one arm, 2 targets) | e.g. DP-1 | 5 | 10 |
| **total** | | | **41** (16 runnable now: DP + DP intervention) |

## 3. Commands (DP, runnable now)
```bash
for s in 0 1 2; do
  python scripts/run_train_arm.py --arm DP-1 --seed $s --train $USINTENT_DATA/cued/<train>
  python scripts/run_train_arm.py --arm DP-5 --seed $s --train $USINTENT_DATA/cued/<train>
done
P=<natural pi-bar printed by run.json of a natural run>   # or compute: see reports/M6.md
for s in 0 1 2 3 4; do
  python scripts/run_train_arm.py --arm DP-1 --seed $s --train ... --target-departure-fraction $(python -c "print(0.5*$P)")
  python scripts/run_train_arm.py --arm DP-1 --seed $s --train ... --target-departure-fraction $(python -c "print(2*$P)")
done
```
Each run writes `results/train/<arm>[_p<target>]/seed<k>/{model.pt,run.json}` with the seed,
git hash, config hash, data hash and (for intervention runs) the realised departure fraction of
one epoch (must be within ±0.01 of the target). Paste the run.json files back.
