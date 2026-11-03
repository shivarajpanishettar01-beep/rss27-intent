# INTERFACES.md — topics, data fields, config schema

Anything marked `?` is unknown and must be read from config (see `docs/RIG_FACTS.md`).

## ROS 2 topics

| Topic | Type | Rate | Producer | Notes |
|---|---|---|---|---|
| B-mode (name ?) | ? (from micrus) | ? | micrus | Do not rename; remap in launch files |
| F/T wrench (name ?) | `geometry_msgs/WrenchStamped` (confirm) | ? | UR driver or sensor driver | Frame ? |
| Probe camera (name ?) | `sensor_msgs/Image` / `CompressedImage` | ? | camera driver | Only if a camera is present |
| `/joint_states` | `sensor_msgs/JointState` | driver rate | UR driver | |
| TCP pose (name ?) | `geometry_msgs/PoseStamped` (confirm) | ? | UR driver / tf | |
| `/protocol/mode_cue` | custom `ModeEvent` | event | scheduler | cue time, next mode, station index, hold id, sampled D |
| `/protocol/mode_key` | custom `ModeEvent` | event | teleop | key time, mode, station index, hold id |
| `/protocol/session` | custom `SessionInfo` | latched | scheduler | seed, hold distribution, block id, operator id |
| `/policy/chunk` | custom `ActionChunk` | policy rate | policy server | displacements (tool frame), horizon, stamp, arm id |
| `/policy/contact_cmd` | custom `ModeEvent` | policy rate | policy server | m̂ |
| `/control/filter_event` | custom `FilterEvent` | event | control | raw cmd, filtered cmd, which constraint, stamp |
| `/control/fault` | `std_msgs/String` (latched) | event | control | fault reason |

Custom messages live in `ros2_ws/src/us_intent_msgs`.

## Dataset fields (LeRobot v3.0 episode)

- `observation.bmode` (lossless), `observation.camera` (if present), `observation.state` (TCP pose in base + joint positions), `observation.wrench` (tool frame).
- `action` (TCP displacement over one step, tool frame; the chunk is built by the loader), `action.contact_cmd` (logged operator mode).
- Per-step metadata: `mode`, `station_index`, `hold_id`, `episode_time`, `session_id`, `day`, `operator_id`.
- Per-hold table (sidecar parquet): `hold_id, start_time, cue_time, key_time, D (start to key), reaction_time (key − cue), exit_type, sampled_D, distribution, block_id`.

## Config files (`configs/`)

- `rig.yaml`: topic names, frames, rates, latency (all `null` until measured).
- `safety.yaml`: max force, max speed, max acceleration, max normal displacement, workspace box, `max_input_age_s`, watchdog period (all `null`; see `SAFETY.md`).
- `data.yaml`: segmentation thresholds (`v_hold`, `T_hold_min`, `T_onset`, `w`), chunk `H`, history `k`, split assignment.
- `protocol.yaml`: hold distributions (`uniform: [2, 10]`, `shifted_exp: {shift: 2, mean: 4}`), station-count range, sweep speeds, tilt/twist range (null until set), block design.
- `arms/*.yaml`: one per arm (inputs, intent access, seeds, schedule).
- `metrics.yaml`: departure threshold `thr` (null until measured), `p_view`, force band (null until set).
- `whenhead.yaml`: bin width, MLP sizes, coupling threshold, confidence implementation.
