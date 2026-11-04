# LAB_RUN_M2 — Open-H inventory and passthrough (lab machine, no robot)

Cloud sessions cannot reach HuggingFace, so these steps run in the lab. No GPU needed.

```bash
pip install -e ".[dev,ml,lerobot]"
export USINTENT_DATA=/path/to/data          # Open-H goes under $USINTENT_DATA/openh/
```

1. Download the Open-H robotic-ultrasound subsets into `$USINTENT_DATA/openh/<name>/` (LeRobot v3.0 folders; use the official Open-H instructions; I could not read them from the cloud).
2. Counts (confirms the paper's 1,629 real vascular / 4,752 simulated liver):
   ```bash
   python scripts/run_openh_counts.py $USINTENT_DATA/openh/*
   ```
   Expected: JSON with `total_episodes`, `total_frames`, `fps`, and the feature list per subset; also `results/openh_counts.json`. Runtime: seconds.
3. Paste the `features` lists back to me, or fill `configs/data.yaml: openh` yourself (pose key, slices, rotation format, B-mode key, wrench key, session/day keys). **Open-H may have no session/day metadata; then splits by session are impossible and you need to decide what unit to split by.**
4. Passthrough on one real episode (acceptance M2):
   ```bash
   python scripts/run_passthrough.py --openh $USINTENT_DATA/openh/<vascular> --repo-id <repo_id> --episodes 1
   ```
   Expected: `"passed": true`, `max_trans_err_m < 1e-6`. Runtime: < 1 min.
5. Segmentation sensitivity (after filling `segmentation.v_hold_mps`, `t_hold_min_s`, `t_onset_s`):
   ```bash
   python scripts/run_segmentation_sensitivity.py --openh $USINTENT_DATA/openh/<vascular> --repo-id <repo_id>
   ```
   Writes `results/segmentation_sensitivity/openh.json`. Runtime: minutes.
