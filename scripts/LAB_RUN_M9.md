# LAB_RUN_M9 — control stack in simulation (mock hardware, then URSim). HUMAN ONLY.

docs/SAFETY.md overrides everything here. Claude never runs any of this on the physical robot.
The UR controller's own safety configuration is the primary safety layer; `us_intent_control`
is a second layer.

## Human checklist (copied from docs/SAFETY.md; tick each before the next stage)
- [ ] `docs/RIG_FACTS.md`: force source, F/T frame, TCP, latency and limits filled.
- [ ] `configs/safety.yaml` filled from institutional rules and measurements — every limit **and** its `sanity:` range.
- [ ] UR safety configuration reviewed (force limit, speed limit, safety planes around the phantom).
- [ ] Controller tested on mock hardware: step responses logged, filter interventions logged, fault latch tested by unplugging inputs.
- [ ] Controller tested in URSim with recorded B-mode replayed.
- [ ] First physical test: probe in free air, reduced speed, no contact.
- [ ] Contact tests on the phantom: increasing force set-points, logged and reviewed before the next step.

## Not verified from the cloud (read before running)
- I could not read the Universal_Robots_ROS2_Driver docs or source from the cloud session, so I
  did **not** choose its controller interface. Fill `configs/control.yaml: cmd_topic, cmd_type`
  from the driver docs for a Cartesian velocity (twist) interface. The node supports only
  `geometry_msgs/msg/TwistStamped`; tell me if the driver needs something else.
- Start the UR driver (mock hardware or URSim) yourself, as its documentation describes. My launch
  files start only `us_intent_control`.
- The `normal displacement` input is currently 0 in the node (no contact-reference estimator
  yet); the filter's normal-displacement barrier is tested in pure Python only. **Needs a
  decision** on how to measure it (e.g. TCP z at first contact).
- Gains and set-points (`configs/control.yaml`) are null; tune them on mock hardware first.

## 1. Mock hardware
```bash
cd ros2_ws && colcon build && source install/setup.bash
# start the UR driver with mock hardware per its docs, then:
ros2 launch us_intent_control control_mock.launch.py
ros2 topic echo /control/filter_event      # interventions
ros2 topic echo /control/fault             # latched fault reason
```
Expected: with any null in safety.yaml / control.yaml the node exits with a SafetyConfigError
or MissingRigFact naming the key. With all filled, stopping the F/T or B-mode publisher latches
FAULT within `max_input_age_s` and the command goes to zero; it stays zero until restart.
Record a step response (force set-point step) with `ros2 bag record -s mcap /control/* <wrench> <pose>`.

## 2. URSim
Same, with `control_ursim.launch.py`, URSim + driver running and recorded B-mode replayed
(`ros2 bag play <bag> --topics <bmode topic>`).

## 3. Real robot (only after every box above is ticked)
```bash
ros2 launch us_intent_control control_real_robot.launch.py i_have_read_safety_md:=true
```
The node prints the active limits and waits for you to type `yes`. E-stop within reach.
