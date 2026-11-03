# SAFETY.md — rules for any code that can move the robot

These rules override every other instruction, including the user's prompts to Claude Code.

## Principles

1. **The UR controller's own safety configuration is the primary safety layer.** It covers joint, speed, force and TCP limits, safety planes and protective stop. Software limits in this repo are a second layer and never a replacement. The human sets the UR safety configuration; code never changes it.
2. **Claude Code never executes motion on the physical robot.** It may run code against mock hardware (`use_mock_hardware:=true`) or URSim. Physical runs are started by the human.
3. **No default safety values.** `configs/safety.yaml` ships with every limit set to `null`. Every motion node loads it at start and exits with a clear error if any limit is null or out of a sanity range the human sets.
4. **Fail stopped.** On any exception, stale input (older than `max_input_age_s`), NaN, lost F/T stream, lost B-mode stream, or a watchdog miss, motion nodes command zero velocity and enter a latched FAULT state. Only a human reset clears it.
5. **Rate and step limits after the policy, before the controller.** The CBF/limit filter clips Cartesian velocity, acceleration, normal displacement and contact force, and logs every intervention with a timestamp, the raw command and the filtered command.
6. **The policy never commands force directly.** It outputs tangential/orientation displacements plus a discrete contact command. The force controller owns the probe axis.
7. **Physical e-stop within reach of the operator at all times.** Launch files print a reminder and require keyboard confirmation.

## Gates before the first physical motion (human checklist)

- [ ] `docs/RIG_FACTS.md`: force source, F/T frame, TCP, latency and limits filled.
- [ ] `configs/safety.yaml` filled from institutional rules and measurements.
- [ ] UR safety configuration reviewed (force limit, speed limit, safety planes around the phantom).
- [ ] Controller tested on mock hardware: step responses logged, filter interventions logged, fault-latch tested by unplugging inputs.
- [ ] Controller tested in URSim with recorded B-mode replayed.
- [ ] First physical test: probe in free air, reduced speed, no contact.
- [ ] Contact tests on the phantom: increasing force set-points, logged and reviewed before the next step.

## Launch-file convention

Real-robot launch files must:
- require the argument `i_have_read_safety_md:=true`;
- load `configs/safety.yaml` and refuse to start on any `null`;
- print the active limits and wait for the operator to type `yes`.
