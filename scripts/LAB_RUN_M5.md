# LAB_RUN_M5 — first dry run of the protocol stack on the lab PC (NO robot motion)

Nothing here commands the robot. Keep the physical e-stop within reach anyway (docs/SAFETY.md).

## 0. Facts I need from you first (all `null` now)
- `configs/rig.yaml: ros_distro` → `echo $ROS_DISTRO` on the control PC.
- `configs/rig.yaml: topics.bmode, topics.bmode_type` → `ros2 topic list -t` with micrus running.
  The converter supports only `sensor_msgs/msg/Image` with `mono8`; if micrus publishes another
  type, send me its `.msg` definition (or the micrus source) and I extend the converter.
- `topics.wrench`, `topics.tcp_pose` (and `topics.camera` if a probe camera exists).
- `configs/protocol.yaml: stations_per_episode.min/max, sweep_speeds_mps, audio_cue_command`.

## 1. Build (ROS 2 environment)
```bash
cd <repo> && pip install -e ".[dev,ml,lerobot,bags]"
cd ros2_ws && colcon build && source install/setup.bash
```
Expected: 6 packages built. This build has **never been run** (no ROS / Docker daemon in the cloud
session); paste any error back to me. Runtime: ~1–3 min.

## 2. Plan-only dry run (no ROS)
```bash
python scripts/run_protocol_dry_run.py --session-id pilot-dry --seed 1 --distribution uniform
```
Expected: JSON with the session info and per-hold `sampled_d_s`, then a prompt; type `yes`.

## 3. ROS dry run (scheduler without audio + teleop; no motion executor running)
```bash
ros2 launch us_intent_scheduler protocol_dry_run.launch.py session_id:=pilot-dry seed:=1 \
    distribution:=uniform n_stations:=4
```
Expected: the scheduler prints the plan and waits for `yes`; after `yes`, `ros2 topic echo
/protocol/session` shows the latched SessionInfo. Pressing `s` in the teleop terminal publishes a
`/protocol/mode_key` event and one `/teleop/target_station`. To simulate arrival:
`ros2 topic pub --once /teleop/arrived std_msgs/msg/Int32 "{data: 1}"`; the scheduler then emits a
`/protocol/mode_cue` after the sampled hold. If stdin does not reach the nodes under
`ros2 launch`, run them in two terminals with `ros2 run` instead.

## 4. Recording + conversion (micrus running, robot idle)
```bash
ros2 run us_intent_recorder recorder_node bags/pilot-dry-ep0     # Ctrl-C after ~30 s
python scripts/run_convert_bags.py --day <YYYY-MM-DD> --out $USINTENT_DATA/cued/pilot-dry bags/pilot-dry-ep0/*.mcap
```
Expected: JSON with episodes/holds and `passthrough.max_trans_err_m < 1e-6`. If it fails with
"two clocks", micrus stamps frames with a device clock: tell me which (RIG_FACTS "timestamp source").
Paste all outputs back.
