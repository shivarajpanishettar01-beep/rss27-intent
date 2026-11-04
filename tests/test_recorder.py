"""M5 converter round trip: SYNTHETIC holdworld -> MCAP bag -> LeRobot -> passthrough."""
import numpy as np
import pytest

from usintent.config import MissingRigFact, load
from usintent.data.chunks import ChunkSampler
from usintent.data.loaders import read_lerobot, write_lerobot
from usintent.data.passthrough import passthrough_check
from usintent.protocol.recorder import (
    BagTopics,
    ConversionError,
    convert_bag,
    record_command,
    write_synthetic_bag,
)
from usintent.sim.holdworld import HoldWorldConfig, generate

T = BagTopics("/syn/bmode", "/syn/wrench", "/syn/tcp_pose")   # synthetic topic names
T0 = 1_762_000_000_123_456_789
SESSION = {"session_id": "syn-s0", "seed": 5, "hold_distribution": "uniform", "block_id": 0,
           "operator_id": "syn-op0"}


@pytest.fixture(scope="module")
def eps():
    return generate(HoldWorldConfig(image_shape=(16, 16), n_stations=(3, 4), tool_yaw_rad=0.4,
                                    seed=12), 3)


def test_round_trip(tmp_path, eps) -> None:
    conv = []
    for i, ep in enumerate(eps):
        bag = write_synthetic_bag(ep, tmp_path / f"e{i}.mcap", T, T0 + i * 10**11, SESSION)
        conv.append(convert_bag(bag, T, i, "syn-d0"))
    for ep, c in zip(eps, conv):
        assert np.array_equal(c.bmode, ep.bmode)                         # bytes identical
        assert np.all(np.diff(c.time) > 0) and np.max(np.abs(c.time - ep.time)) < 1e-8
        assert np.max(np.abs(c.state[:, :6] - ep.state[:, :6])) < 1e-9
        cols = ["hold_id", "station_index", "start_time", "cue_time", "key_time", "D", "exit_type"]
        assert np.allclose(c.holds[cols[2:6]].to_numpy(float), ep.holds[cols[2:6]].to_numpy(float),
                           atol=1e-8)
        assert list(c.holds["exit_type"]) == list(ep.holds["exit_type"])
        assert np.array_equal(c.hold_id, ep.hold_id)
        last_key = ep.holds["key_time"].max()
        before = ep.time < last_key
        assert np.array_equal(c.mode[before], ep.mode[before])
    back = read_lerobot(write_lerobot(conv, tmp_path / "ds"))
    res = passthrough_check(ChunkSampler(back, 1.0))
    assert res["max_trans_err_m"] < 1e-6 and res["max_rot_err_rad"] < 1e-6
    assert all(np.array_equal(a.bmode, b.bmode) for a, b in zip(conv, back))


def test_two_clocks_detected(tmp_path, eps) -> None:
    from mcap_ros2.writer import Writer

    from usintent.protocol.recorder import MSGDEFS

    ep = eps[0]
    bag = write_synthetic_bag(ep, tmp_path / "ok.mcap", T, T0, SESSION)
    assert convert_bag(bag, T, 0, "d")
    # re-write with B-mode header stamps on a device clock 2 s off the bag clock
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    bad = tmp_path / "bad.mcap"
    with open(bag, "rb") as src, open(bad, "wb") as f:
        w = Writer(f)
        sch = {k: w.register_msgdef(k, v) for k, v in MSGDEFS.items()}
        reader = make_reader(src, decoder_factories=[DecoderFactory()])
        for schema, channel, message, decoded in reader.iter_decoded_messages():
            w.write_message(channel.topic, sch[schema.name], _shift(channel.topic, decoded, 2.0),
                            log_time=message.log_time)
        w.finish()
    with pytest.raises(ConversionError, match="two clocks"):
        convert_bag(bad, T, 0, "d")


def _shift(topic, msg, seconds):
    """Rebuild a decoded message as a dict, shifting B-mode header stamps."""
    def todict(x):
        if hasattr(x, "__slots__"):
            return {k: todict(getattr(x, k)) for k in x.__slots__}
        return x
    d = todict(msg)
    if topic == T.bmode:
        d["header"]["stamp"]["sec"] += int(seconds)
    return d


def test_record_profile_needs_rig_topics() -> None:
    with pytest.raises(MissingRigFact):
        record_command(load("rig"), "out")
    rig = {"topics": {"bmode": "/b", "wrench": "/w", "tcp_pose": "/p", "camera": None}}
    cmd = record_command(rig, "out")
    assert cmd[:5] == ["ros2", "bag", "record", "-s", "mcap"]
    assert {"/b", "/w", "/p", "/protocol/mode_cue", "/protocol/mode_key", "/protocol/session",
            "/joint_states"} <= set(cmd)
