# LAB_RUN_M6 — training runs (lab GPU)

Both policy families are implemented. GR00T-H: `usintent/policies/gr00t_adapter.py`, read from
Isaac-GR00T commit c56f541 (reports/REVIEW_FIXES.md §7). DP: LeRobot 0.6.1.

## 0. Decisions needed first
- `configs/arms/*.yaml: schedule.steps, batch_size, lr` (one shared schedule) and, for A*,
  `checkpoint` (the GR00T-H N1.7 base checkpoint id/path).
- `rig.yaml topics.camera` ("none" if no probe camera); `data.yaml segmentation.v_hold_mps` (A6).

## 1. Isaac-GR00T environment (once)
```bash
git clone https://github.com/NVIDIA/Isaac-GR00T && cd Isaac-GR00T && git checkout c56f541
# follow its README to create the uv environment (Python 3.12, torch 2.9, flash-attn)
uv pip install -e <path-to-rss27-intent>       # usintent inside the GR00T env
GR00T_ROOT=$PWD pytest <path-to-rss27-intent>/tests/test_gr00t_adapter.py -m lab   # config check
```
Also check that `python gr00t/experiment/launch_finetune.py --help` lists `--state-dropout-prob`.

## 2. GR00T-H arms (A1, A2, A3, A5, A6; A7 = A5-type model + when-head)
```bash
for arm in A1 A2 A3 A5 A6; do for s in 0 1 2 3 4; do
  python scripts/run_train_arm.py --arm $arm --seed $s --train $USINTENT_DATA/cued/<train> \
      --gr00t-root <Isaac-GR00T> --launch
done; done
python scripts/run_train_arm.py --arm A5 --seed 0 --train ... --gr00t-root ... --language-mode --launch
python scripts/run_train_arm.py --arm A5 --seed 1 --train ... --gr00t-root ... --language-mode --launch
```
Each run exports the GR00T LeRobot-v2 copy once (`results/gr00t/data/<arm>_<datahash>`), checks
the GR00T passthrough (must print `max_err` < 1e-6), writes `usintent_spec.json` and `run.json`,
and runs `scripts/gr00t_finetune_seeded.py` (GR00T's launch_finetune with `config.data.seed`).
**First run: open one exported video and one parquet in GR00T's own loader** (their
`gr00t/eval/open_loop_eval.py --dataset-path <export> --embodiment-tag NEW_EMBODIMENT ...`) to
confirm the export loads.

## 3. DP arms and the intervention (DP only)
```bash
for s in 0 1 2; do
  python scripts/run_train_arm.py --arm DP-1 --seed $s --train ...
  python scripts/run_train_arm.py --arm DP-5 --seed $s --train ...
done
for s in 0 1 2 3 4; do for p in <0.5*pibar> <2*pibar>; do
  python scripts/run_train_arm.py --arm DP-1 --seed $s --train ... --target-departure-fraction $p
done; done
```

## 4. Run count and GPU hours
| block | runs |
|---|---|
| GR00T A1 A2 A3 A5 A6 × 5 seeds | 25 |
| GR00T A5 language variant × 2 seeds | 2 |
| DP-1, DP-5 × 3 seeds | 6 |
| DP-1 intervention, 2 targets × 5 seeds | 10 |
| **total** | **43** |
GPU hours = runs × steps × (s/step) / 3600; time 100 steps of one GR00T and one DP run first.
