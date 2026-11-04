"""Recording profile and bag (MCAP) -> LeRobot v3.0 converter. SPEC 12, INTERFACES. M5.

Bag reading/writing uses the pure-Python ``mcap`` + ``mcap-ros2-support`` packages (CDR
decoding from the message definitions stored in the bag), so the converter runs without a ROS
install. One bag = one episode (the recorder is started and stopped per episode).

Converter rules:
- Master clock: B-mode header stamps. TCP pose is interpolated to those stamps (linear position,
  slerp orientation), wrench linearly. Stamps are integer ns; episode time = (stamp - first
  B-mode stamp) * 1e-9 s.
- Clock checks: header stamps strictly increasing per topic, and |header stamp - bag log time| <=
  ``max_clock_skew_s`` for every message (catches a device clock that is not the ROS clock).
- B-mode stored losslessly (mono8 bytes copied). Supported B-mode type: sensor_msgs/msg/Image with
  encoding mono8. The micrus message type is unknown (docs/RIG_FACTS.md): any other type raises.
- Protocol events: /protocol/mode_key with source "arrival" starts a hold; /protocol/mode_cue is
  the scheduler cue; /protocol/mode_key with source "operator_key" is the key press (ends D).
- Per-step actions are derived from the interpolated TCP poses (usintent.data.chunks).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation, Slerp

from usintent.config import require
from usintent.data.chunks import actions_from_poses
from usintent.data.loaders import HOLD_COLUMNS, Episode

PROTOCOL_TOPICS = ("/protocol/mode_cue", "/protocol/mode_key", "/protocol/session",
                   "/policy/chunk", "/policy/contact_cmd", "/control/filter_event",
                   "/control/fault")

_SEP = "=" * 80 + "\n"
_TIME = "MSG: builtin_interfaces/Time\nint32 sec\nuint32 nanosec\n"
_HEADER = "MSG: std_msgs/Header\nbuiltin_interfaces/Time stamp\nstring frame_id\n"
MSGDEFS = {
    "sensor_msgs/msg/Image": ("std_msgs/Header header\nuint32 height\nuint32 width\n"
                              "string encoding\nuint8 is_bigendian\nuint32 step\nuint8[] data\n"
                              + _SEP + _HEADER + _SEP + _TIME),
    "geometry_msgs/msg/PoseStamped": (
        "std_msgs/Header header\ngeometry_msgs/Pose pose\n" + _SEP + _HEADER + _SEP + _TIME + _SEP
        + "MSG: geometry_msgs/Pose\ngeometry_msgs/Point position\n"
          "geometry_msgs/Quaternion orientation\n" + _SEP
        + "MSG: geometry_msgs/Point\nfloat64 x\nfloat64 y\nfloat64 z\n" + _SEP
        + "MSG: geometry_msgs/Quaternion\nfloat64 x\nfloat64 y\nfloat64 z\nfloat64 w\n"),
    "geometry_msgs/msg/WrenchStamped": (
        "std_msgs/Header header\ngeometry_msgs/Wrench wrench\n" + _SEP + _HEADER + _SEP + _TIME
        + _SEP + "MSG: geometry_msgs/Wrench\ngeometry_msgs/Vector3 force\n"
                 "geometry_msgs/Vector3 torque\n" + _SEP
        + "MSG: geometry_msgs/Vector3\nfloat64 x\nfloat64 y\nfloat64 z\n"),
    # us_intent_msgs: identical to ros2_ws/src/us_intent_msgs/msg/*.msg
    "us_intent_msgs/msg/ModeEvent": ("builtin_interfaces/Time stamp\nstring mode\nstring source\n"
                                     "int32 station_index\nint32 hold_id\n"
                                     "float64 sampled_hold_s\n" + _SEP + _TIME),
    "us_intent_msgs/msg/SessionInfo": ("builtin_interfaces/Time stamp\nstring session_id\n"
                                       "int64 seed\nstring hold_distribution\nint32 block_id\n"
                                       "string operator_id\n" + _SEP + _TIME),
}


class ConversionError(RuntimeError):
    """The bag violates a converter rule (clock, type, missing topic)."""


def record_topics(rig: dict[str, Any]) -> list[str]:
    """Every topic of docs/INTERFACES.md that the recorder must log (rig topics from config)."""
    topics = [require(rig, "topics.bmode"), require(rig, "topics.wrench"),
              require(rig, "topics.tcp_pose"), "/joint_states"]
    cam = rig.get("topics", {}).get("camera")
    if cam:
        topics.append(cam)
    return [str(t) for t in topics] + list(PROTOCOL_TOPICS)


def record_command(rig: dict[str, Any], out_dir: Path) -> list[str]:
    """``ros2 bag record`` command line for the MCAP profile (storage plugin flag ``-s mcap``)."""
    return ["ros2", "bag", "record", "-s", "mcap", "-o", str(out_dir), *record_topics(rig)]


# ------------------------------------------------------------------------------- helpers


def _stamp(ns: int) -> dict[str, int]:
    return {"sec": int(ns // 1_000_000_000), "nanosec": int(ns % 1_000_000_000)}


def _ns(stamp: Any) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


@dataclass(frozen=True)
class BagTopics:
    bmode: str
    wrench: str
    tcp_pose: str

    @classmethod
    def from_rig(cls, rig: dict[str, Any]) -> BagTopics:
        if require(rig, "topics.bmode_type") != "sensor_msgs/msg/Image":
            raise ConversionError("only sensor_msgs/msg/Image (mono8) B-mode is supported; "
                                  "inspect micrus and extend the converter")
        return cls(require(rig, "topics.bmode"), require(rig, "topics.wrench"),
                   require(rig, "topics.tcp_pose"))


# ------------------------------------------------------------------------------- writer


def write_synthetic_bag(ep: Episode, path: Path, topics: BagTopics, t0_ns: int,
                        session: dict[str, Any]) -> Path:
    """Write a SYNTHETIC hold-world episode as an MCAP bag with the real topic layout."""
    from mcap_ros2.writer import Writer

    if ep.bmode is None:
        raise ValueError("episode needs B-mode frames")
    path = Path(path)
    stamps = t0_ns + np.round(ep.time * 1e9).astype(np.int64)
    quat = Rotation.from_rotvec(ep.state[:, 3:6]).as_quat()   # x, y, z, w
    events: list[tuple[int, str, dict[str, Any]]] = []
    for _, h in ep.holds.iterrows():
        def ev(t: float, mode: str, source: str, h: Any = h) -> dict[str, Any]:
            return {"stamp": _stamp(t0_ns + round(t * 1e9)), "mode": mode, "source": source,
                    "station_index": int(h["station_index"]), "hold_id": int(h["hold_id"]),
                    "sampled_hold_s": float(h["sampled_D"]) if source == "scheduler_cue"
                    else float("nan")}
        events.append((t0_ns + round(h["start_time"] * 1e9), "/protocol/mode_key",
                       ev(h["start_time"], "HOLD", "arrival")))
        if np.isfinite(h["cue_time"]):
            events.append((t0_ns + round(h["cue_time"] * 1e9), "/protocol/mode_cue",
                           ev(h["cue_time"], h["exit_type"], "scheduler_cue")))
        if np.isfinite(h["key_time"]):
            events.append((t0_ns + round(h["key_time"] * 1e9), "/protocol/mode_key",
                           ev(h["key_time"], h["exit_type"], "operator_key")))
    with open(path, "wb") as f:
        w = Writer(f)
        sch = {k: w.register_msgdef(k, v) for k, v in MSGDEFS.items()}
        w.write_message("/protocol/session", sch["us_intent_msgs/msg/SessionInfo"],
                        {"stamp": _stamp(int(stamps[0])), **session}, log_time=int(stamps[0]))
        rows, cols = ep.bmode.shape[1:]
        merged: list[tuple[int, int, str, str, dict[str, Any]]] = []
        for k, ns in enumerate(stamps.tolist()):
            hdr = {"stamp": _stamp(ns), "frame_id": "probe"}
            merged.append((ns, 0, topics.tcp_pose, "geometry_msgs/msg/PoseStamped", {
                "header": {**hdr, "frame_id": "base"},
                "pose": {"position": dict(zip("xyz", map(float, ep.state[k, :3]))),
                         "orientation": dict(zip("xyzw", map(float, quat[k])))}}))
            merged.append((ns, 1, topics.wrench, "geometry_msgs/msg/WrenchStamped", {
                "header": {**hdr, "frame_id": "tool"},
                "wrench": {"force": dict(zip("xyz", map(float, ep.wrench[k, :3]))),
                           "torque": dict(zip("xyz", map(float, ep.wrench[k, 3:])))}}))
            merged.append((ns, 2, topics.bmode, "sensor_msgs/msg/Image", {
                "header": hdr, "height": rows, "width": cols, "encoding": "mono8",
                "is_bigendian": 0, "step": cols, "data": ep.bmode[k].tobytes()}))
        merged += [(ns, 3, topic, "us_intent_msgs/msg/ModeEvent", msg) for ns, topic, msg in events]
        for ns, _, topic, typ, msg in sorted(merged, key=lambda x: (x[0], x[1])):
            w.write_message(topic, sch[typ], msg, log_time=ns)
        w.finish()  # type: ignore[no-untyped-call]
    return path


# ------------------------------------------------------------------------------- converter


def _read(path: Path) -> dict[str, list[tuple[int, int, Any, str]]]:
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    out: dict[str, list[tuple[int, int, Any, str]]] = {}
    with open(path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for schema, channel, message, decoded in reader.iter_decoded_messages():
            out.setdefault(channel.topic, []).append(
                (message.log_time, 0, decoded, schema.name if schema else ""))
    return out


def convert_bag(path: Path, topics: BagTopics, episode_index: int, day: str,
                max_clock_skew_s: float = 0.5, fps: float | None = None) -> Episode:
    """Convert one MCAP bag (one episode) to an :class:`Episode` (see module docstring)."""
    msgs = _read(Path(path))
    for t in (topics.bmode, topics.tcp_pose, topics.wrench, "/protocol/session"):
        if t not in msgs:
            raise ConversionError(f"topic {t} missing from {path}")
    for topic, lst in msgs.items():
        hdr_ns = [(_ns(m.header.stamp) if hasattr(m, "header") else _ns(m.stamp), log)
                  for log, _, m, _ in lst]
        st = np.array([h for h, _ in hdr_ns], dtype=np.int64)
        if topic in (topics.bmode, topics.tcp_pose, topics.wrench) and np.any(np.diff(st) <= 0):
            raise ConversionError(f"{topic}: header stamps not strictly increasing")
        skew = np.abs(st - np.array([lg for _, lg in hdr_ns], dtype=np.int64)) * 1e-9
        if np.any(skew > max_clock_skew_s):
            raise ConversionError(f"{topic}: header stamps differ from bag time by up to "
                                  f"{skew.max():.3f} s (two clocks?)")
    imgs = msgs[topics.bmode]
    if any(s != "sensor_msgs/msg/Image" or m.encoding != "mono8" for _, _, m, s in imgs):
        raise ConversionError("B-mode must be sensor_msgs/msg/Image mono8")
    b_ns = np.array([_ns(m.header.stamp) for _, _, m, _ in imgs], dtype=np.int64)
    t0 = int(b_ns[0])
    time = np.asarray((b_ns - t0) * 1e-9, dtype=np.float64)
    bmode = np.stack([np.frombuffer(m.data, np.uint8).reshape(m.height, m.width)
                      for _, _, m, _ in imgs])
    poses = msgs[topics.tcp_pose]
    p_t = np.array([_ns(m.header.stamp) - t0 for _, _, m, _ in poses]) * 1e-9
    pos = np.array([[m.pose.position.x, m.pose.position.y, m.pose.position.z]
                    for _, _, m, _ in poses])
    rot = Rotation.from_quat([[m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z,
                               m.pose.orientation.w] for _, _, m, _ in poses])
    if time[0] < p_t[0] or time[-1] > p_t[-1]:
        raise ConversionError("B-mode frames outside the TCP pose time range")
    state = np.zeros((len(time), 12))
    state[:, :3] = np.column_stack([np.interp(time, p_t, pos[:, i]) for i in range(3)])
    state[:, 3:6] = Slerp(p_t, rot)(time).as_rotvec()
    wr = msgs[topics.wrench]
    w_t = np.array([_ns(m.header.stamp) - t0 for _, _, m, _ in wr]) * 1e-9
    w_v = np.array([[m.wrench.force.x, m.wrench.force.y, m.wrench.force.z, m.wrench.torque.x,
                     m.wrench.torque.y, m.wrench.torque.z] for _, _, m, _ in wr])
    wrench = np.column_stack([np.interp(time, w_t, w_v[:, i]) for i in range(6)])
    sess = msgs["/protocol/session"][0][2]
    holds, mode, cmd, st_idx, hold_id = _events_to_holds(msgs, t0, time, episode_index, sess)
    rate = fps if fps is not None else float(np.round(1.0 / np.median(np.diff(time))))
    ep = Episode(
        episode_index=episode_index, fps=rate, time=time, state=state,
        action=actions_from_poses(state[:, :6]), wrench=wrench, mode=mode, contact_cmd=cmd,
        station_index=st_idx, hold_id=hold_id, session_id=str(sess.session_id), day=day,
        operator_id=str(sess.operator_id), holds=holds, bmode=bmode)
    ep.validate()
    return ep


def _events_to_holds(msgs: dict[str, list[tuple[int, int, Any, str]]], t0: int,
                     time: NDArray[np.float64], episode_index: int, sess: Any
                     ) -> tuple[pd.DataFrame, NDArray[np.str_], NDArray[np.str_],
                                NDArray[np.int64], NDArray[np.int64]]:
    keys = [m for _, _, m, _ in msgs.get("/protocol/mode_key", [])]
    cues = {int(m.hold_id): m for _, _, m, _ in msgs.get("/protocol/mode_cue", [])}
    arrivals = [m for m in keys if m.source == "arrival"]
    presses = {int(m.hold_id): m for m in keys if m.source == "operator_key"}
    n = len(time)
    mode = np.full(n, "RECONTACT", dtype="<U9")
    cmd = np.full(n, "RECONTACT", dtype="<U9")
    st_idx = np.full(n, -1, dtype=np.int64)
    hold_id = np.full(n, -1, dtype=np.int64)
    t_end = float(time[-1])
    rows = []
    def sec(m: Any) -> float:
        return (_ns(m.stamp) - t0) * 1e-9

    starts = sorted(sec(a) for a in arrivals)
    for a in arrivals:
        hid, start = int(a.hold_id), sec(a)
        cue, key = cues.get(hid), presses.get(hid)
        cue_t = sec(cue) if cue is not None else np.nan
        key_t = sec(key) if key is not None else np.nan
        nxt = [s for s in starts if s > start]
        seg_end = nxt[0] if nxt else np.inf
        exit_type = str(key.mode) if key is not None else (str(cue.mode) if cue is not None else "")
        in_hold = (time >= start) & (time < (key_t if np.isfinite(key_t) else np.inf))
        hold_id[in_hold] = hid
        st_idx[in_hold] = int(a.station_index)
        cmd[in_hold] = "HOLD"
        mode[(time >= start) & (time < (cue_t if np.isfinite(cue_t) else np.inf))] = "HOLD"
        if cue is not None and np.isfinite(cue_t):
            mode[(time >= cue_t) & (time < seg_end)] = str(cue.mode)
        if np.isfinite(key_t):
            cmd[(time >= key_t) & (time < seg_end)] = exit_type
        rows.append({
            "hold_id": hid, "episode_index": episode_index, "station_index": int(a.station_index),
            "start_time": start, "cue_time": cue_t, "key_time": key_t,
            "D": key_t - start, "reaction_time": key_t - cue_t, "exit_type": exit_type,
            "backtrack": False, "sampled_D": float(cue.sampled_hold_s) if cue is not None else np.nan,
            "distribution": str(sess.hold_distribution), "block_id": int(sess.block_id),
            "censored": not np.isfinite(key_t) or key_t > t_end,
        })
    holds = pd.DataFrame(rows, columns=HOLD_COLUMNS).sort_values("start_time").reset_index(drop=True)
    # a hold is "backtrack" when the operator leaves it towards a lower station index; unknown
    # (False) for the last hold of the bag, whose next station is not recorded
    stations = holds["station_index"].to_numpy()
    if len(holds):
        holds["backtrack"] = np.concatenate([stations[1:] < stations[:-1], [False]])
    return holds, mode, cmd, st_idx, hold_id
