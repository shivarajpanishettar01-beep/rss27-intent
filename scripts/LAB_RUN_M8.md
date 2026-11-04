# LAB_RUN_M8 — offline evaluation on held-out cued sessions (lab; no robot motion)

Prerequisites: cued sessions converted (LAB_RUN_M5), split by day (`usintent.data.splits`),
trained runs from LAB_RUN_M6, `metrics.yaml: departure_threshold_m, tau_s` and
`data.yaml: segmentation.v_hold_mps` filled.

```bash
python scripts/run_offline_eval.py --train $USINTENT_DATA/cued/<train> --test $USINTENT_DATA/cued/<test> \
    --runs $USINTENT_RESULTS/train/*/seed*
python scripts/make_fig_rate_vs_e.py --eval-chunks $USINTENT_RESULTS/eval/eval_chunks.parquet
python scripts/run_intervention_analysis.py $USINTENT_RESULTS/eval/intervention_rows.json
```
Expected: `results/eval/eval_table.{csv,json}` with one row per baseline and per arm × seed ×
sampler setting (deterministic noise on/off × 4/16 steps), each with a passthrough error < 1e-6;
B4's timing error should match E|D − med_train(D)| of the test holds. `figures/rate_vs_e.*`;
`intervention_rows_analysis.json` (slope of departure-like rate on p, permutation interval).
Runtime: DP inference ≈ (chunks / stride) × arms × seeds × 4 settings forward passes; time 100
first. GR00T arms stay blocked until the adapter exists.

ROS policy server (after LAB_RUN_M5 builds): `ros2 run us_intent_policy_server policy_server_node
--ros-args -p arm:=DP-1 -p checkpoint:=<model.pt> -p log:=/tmp/policy.jsonl` with micrus and the
robot driver streaming (robot idle; this node publishes chunks only). Needs `rates_hz.policy`.
