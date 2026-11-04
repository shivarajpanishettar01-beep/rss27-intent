"""SYNTHETIC protocol values for tests only. Never copy into configs/protocol.yaml (the human sets
the real protocol before collection)."""
from usintent.config import load

SYN_PROTOCOL_EXTRA = {"stations_per_episode": {"min": 3, "max": 6}, "preset_stations": 6,
                      "p_backtrack": 0.2, "p_lift": 0.2, "sweep_speeds_mps": [0.01, 0.02, 0.03],
                      "tilt_twist_range_deg": [10.0, 15.0]}


def syn_protocol(**overrides):
    return {**load("protocol"), **SYN_PROTOCOL_EXTRA, **overrides}
