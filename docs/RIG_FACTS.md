# RIG_FACTS.md — what is known about the rig

Status values:
- `known`: may be used in code.
- `user-reported`: the user said so; treat as known for software structure, but measure before relying on it numerically.
- `unknown`: must come from config; the code refuses to run until filled.

**The human edits this file. Claude Code reads it and never upgrades a status itself.**

| Item | Value | Status | How to get it |
|---|---|---|---|
| Robot arm | Universal Robots UR7e | user-reported | — |
| ROS 2 distro | ? | unknown | `echo $ROS_DISTRO` on the control PC |
| UR driver | Universal_Robots_ROS2_Driver (version ?) | unknown | `ros2 pkg xml ur_robot_driver` |
| Ultrasound system | Telemed, model ? | unknown | Device label / SDK |
| Probe | linear; model, footprint, centre frequency ? | unknown | Datasheet |
| B-mode source | user's `micrus` ROS 2 node | user-reported | — |
| B-mode topic, msg type, encoding | ? | unknown | `ros2 topic list -t` with micrus running |
| B-mode frame rate | ? | unknown | `ros2 topic hz` + inspect micrus code |
| B-mode timestamp source (device vs arrival) | ? | unknown | Inspect micrus code |
| Lossless storage of B-mode | ? | unknown | Inspect micrus / recorder config |
| Force/torque source | wrist F/T sensor **or** UR tool-flange sensing **or** joint-torque estimate ? | unknown | Check hardware. UR e-Series arms include tool-flange F/T sensing; confirm for this UR7e, then measure noise and latency with known weights |
| F/T topic and frame | ? | unknown | `ros2 topic list -t` |
| Probe-mounted camera | present? model, topic ? | unknown | Check hardware |
| Probe tool offset (TCP) | ? | unknown | Calibration |
| Probe-to-image calibration | ? | unknown | Calibration phantom |
| End-to-end latency (image to command) | ? | unknown | LED/motion event visible in both streams |
| Phantom | vascular; model, vessel depth/diameter, stiffness ? | unknown | Datasheet + indentation test |
| Teleop | keyboard teleop with preset waypoints | user-reported | — |
| Dataset format | LeRobot v3.0 | user-reported | — |
| Base VLA | GR00T-H (checkpoint N1.6 or N1.7 ?) | unknown | Check which checkpoint was fine-tuned |
| GPU for training | ? | unknown | — |
| Max contact force, speed, workspace | ? | unknown | Institutional rules + measurement; **never invented** |
| Public data | Open-H-Embodiment robotic-ultrasound subsets (counts in paper: 1,629 real vascular, 4,752 simulated liver: confirm) | user-reported | Count from dataset metadata |
