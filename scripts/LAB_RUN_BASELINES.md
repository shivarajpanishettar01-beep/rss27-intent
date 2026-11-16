# LAB_RUN_BASELINES — external comparison families (lab GPU)

Arms: ACT-1, ACT-5 (LeRobot ACT, trained in our loop like DP), SmolVLA-1, PI05-1 (released
LeRobot base checkpoints fine-tuned with `lerobot-train`) and USVLA-1 (US-VLA, its own code and env). Design and decisions:
`reports/BASELINES.md`. Same data, splits, chunk target and seeds as the GR00T and DP arms.

## 0. Decisions needed first
- `configs/arms/{ACT-1,ACT-5,SmolVLA-1,PI05-1}.yaml: schedule.steps, batch_size, lr`: copy the
  shared schedule used for A*/DP-* (SPEC 10). The VLA arms use this as the **peak** lr of
  LeRobot's policy preset (AdamW + cosine decaying over the run).
- `SmolVLA-1.yaml` / `PI05-1.yaml: rename_map` (no default). Read the base checkpoint's
  `config.json` (`input_features`) and map our keys to its image keys, e.g.
  `{"observation.images.bmode": "<checkpoint image key 1>", "observation.images.camera": "<key 2>"}`.
  If the checkpoint has more image keys than we provide, add `policy.empty_cameras: <missing count>`
  to `lerobot_overrides`.
- pi0.5 memory: full fine-tuning of the 2B PaliGemma backbone needs a large GPU. If it does not
  fit, add to `PI05-1.yaml lerobot_overrides` e.g. `{policy.dtype: bfloat16,
  policy.gradient_checkpointing: true}` or `{policy.train_expert_only: true}` and **report the
  choice in the paper** (it changes what is being compared).

## 1. Environment (once)
```bash
pip install -e ".[ml,lerobot,vla,dev]"        # Python >= 3.12; adds lerobot[training,smolvla,pi]
huggingface-cli login                          # pi0.5 base weights may need an accepted licence
pytest tests/test_external_baselines.py        # must pass before any run
```

## 2. ACT arms (same loop and schedule as DP)
```bash
for s in 0 1 2; do
  python scripts/run_train_arm.py --arm ACT-1 --seed $s --train $USINTENT_DATA/cued/<train>
  python scripts/run_train_arm.py --arm ACT-5 --seed $s --train $USINTENT_DATA/cued/<train>
done
```
`pretrained_backbone: null` trains ResNet-18 from scratch, matching our DP arms. Using ImageNet
weights (LeRobot's ACT default) is a one-line config change; if you change it, change it for DP
too or report the difference.

## 3. VLA arms
```bash
for arm in SmolVLA-1 PI05-1; do for s in 0 1 2; do
  python scripts/run_train_arm.py --arm $arm --seed $s --train $USINTENT_DATA/cued/<train> --launch
done; done
```
Each run exports the LeRobot v3.0 training copy once (`results/lerobot_vla/data/<arm>_<datahash>`),
runs the passthrough through LeRobot's own loader (must print `max_err` < 1e-5,
`padded_chunks` 0, `mode_mismatches` 0), writes `usintent_spec.json` + `run.json` with the exact
`lerobot-train` command, then launches it. Checkpoints land in
`results/lerobot_vla/train/<arm>/seed<k>/lerobot/checkpoints/last/pretrained_model`.

**First run only:** before launching all seeds, run one seed with `--steps=200` added to
`lerobot_overrides` and confirm (a) the loss falls, (b) `lerobot-train` printed no
"missing image key" warning, (c) `python -c "from usintent.policies.evaluate import load_policy;
p = load_policy('SmolVLA-1', '<checkpoint dir>', integration_steps=4)"` loads. Remove the override.

## 3b. US-VLA arm (its own environment)
Once:
```bash
git clone https://github.com/VMVLab/US-VLA && cd US-VLA && git checkout dde576a   # = configs/arms/USVLA-1.yaml usvla_commit
GIT_LFS_SKIP_SMUDGE=1 uv sync
# USFM weights -> ./checkpoints/usfm/USFM_latest.pth (their README, "USFM Pretrained Weights")
pip install --no-deps ./packages/openpi-client && pip install msgpack websockets dm-tree   # into OUR env (client)
```
Per seed (from our repo, our env):
```bash
for s in 0 1 2; do
  python scripts/run_train_arm.py --arm USVLA-1 --seed $s --train $USINTENT_DATA/cued/<train> \
      --usvla-root <US-VLA> --launch
done
```
This writes the npz export + decode passthrough (must print `max_err` < 1e-5), `run.json`, then in
`<US-VLA>` runs `uv run python <repo>/scripts/usvla_run.py build-dataset` (LeRobot v2.1 at
`$HF_LEROBOT_HOME/usintent/USVLA-1_<hash>`, read back as openpi does: must print `max_err` 0,
`mismatches` 0) and `... train` (norm stats + their trainer; checkpoints under
`results/usvla/train/USVLA-1/seed<k>/openpi_checkpoints`).
**First run only:** seed 0 with `schedule.steps` = 200 first; confirm the loss falls and
`norm stats ->` printed a path, then the real budget.

Evaluation, one run at a time (the server checks it serves the run being scored):
```bash
cd <US-VLA> && uv run python <repo>/scripts/usvla_run.py serve --run-json <run dir>/run.json --port 8000 &
python scripts/run_offline_eval.py --test ... --runs <run dir> --usvla-server localhost:8000
```

## 4. Offline evaluation
```bash
python scripts/run_offline_eval.py --test $USINTENT_DATA/cued/<test> \
    --runs results/train/ACT-1/seed* results/train/ACT-5/seed* \
           results/lerobot_vla/train/SmolVLA-1/seed* results/lerobot_vla/train/PI05-1/seed*
```
VLA runs re-run the decode passthrough on the test episodes before scoring (rule 4). ACT has one
sampler setting (deterministic by construction); VLAs use the same 2 x 2 grid as DP.

## 5. What to report back
- `run.json` of every seed (passthrough numbers, command, git hash).
- The eval table rows for these arms; whether any arm's hold-departure rate falls outside the
  seed-noise band of DP-1 (that is the comparison the paper makes).
- Anything you had to set in `lerobot_overrides` and why.
