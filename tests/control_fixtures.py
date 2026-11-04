"""SYNTHETIC test-only safety and control values. Never copy into configs/ (SAFETY.md: the human
sets every limit from institutional rules and measurements)."""
TEST_SAFETY = {
    "max_contact_force_n": 8.0, "max_tcp_speed_mps": 0.05, "max_tcp_accel_mps2": 0.5,
    "max_normal_displacement_m": 0.01, "workspace_box_base_m": [-0.2, 0.2, -0.2, 0.2, 0.0, 0.3],
    "max_input_age_s": 0.1, "watchdog_period_s": 0.02,
    "sanity": {"max_contact_force_n": [1, 20], "max_tcp_speed_mps": [0.001, 0.25],
               "max_tcp_accel_mps2": [0.01, 2.0], "max_normal_displacement_m": [0.001, 0.03],
               "max_input_age_s": [0.01, 0.5], "watchdog_period_s": [0.001, 0.1]},
}
TEST_CONTROL = {"force_setpoint_n": 4.0, "recontact_setpoint_n": 2.0,
                "admittance_gain_mps_per_n": 0.004, "position_gain_per_s": 0.0, "lift_speed_mps": 0.01}
