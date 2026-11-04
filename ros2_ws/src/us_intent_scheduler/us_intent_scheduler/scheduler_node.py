"""Samples stations and hold durations, plays audio cues, publishes /protocol/mode_cue. SPEC 12, M5.

Thin ROS 2 wrapper: all logic is in usintent.protocol.scheduler. Parameters:
  session_id, seed, distribution (uniform | shifted_exp), block_id, operator_id, n_episodes,
  episode_index, dry_run (bool: no audio; still prints the plan and waits for 'yes').
Subscribes /protocol/mode_key (arrival events start a hold); publishes /protocol/mode_cue and the
latched /protocol/session. Audio: configs/protocol.yaml ``audio_cue_command`` (null refuses
unless dry_run). Not tested in a ROS environment from the cloud session (see reports/M5.md).
"""
import math
import shlex
import subprocess


def main() -> None:
    import rclpy
    from builtin_interfaces.msg import Time
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from us_intent_msgs.msg import ModeEvent, SessionInfo

    from usintent.config import load, require
    from usintent.protocol.scheduler import CueClock, plan_session
    from usintent.protocol.session import confirm

    def to_time(t: float) -> Time:
        return Time(sec=int(t), nanosec=round((t - int(t)) * 1e9) % 1_000_000_000)

    class SchedulerNode(Node):
        def __init__(self) -> None:
            super().__init__("us_intent_scheduler")
            p = {k: self.declare_parameter(k, v).value for k, v in {
                "session_id": "", "seed": -1, "distribution": "", "block_id": 0,
                "operator_id": "", "n_episodes": 1, "episode_index": 0, "dry_run": True}.items()}
            if not p["session_id"] or p["seed"] < 0 or not p["distribution"]:
                raise SystemExit("session_id, seed and distribution parameters are required")
            proto = load("protocol")
            self.audio = None if p["dry_run"] else shlex.split(require(proto, "audio_cue_command"))
            plan = plan_session(proto, p["session_id"], int(p["seed"]), p["distribution"],
                                int(p["block_id"]), p["operator_id"], int(p["n_episodes"]))
            ep = plan.episodes[int(p["episode_index"])]
            confirm({**plan.info(), "episode": ep.episode_index, "dry_run": p["dry_run"],
                     "holds": [h.__dict__ for h in ep.holds]})
            self.clock_logic = CueClock(ep)
            latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.pub_session = self.create_publisher(SessionInfo, "/protocol/session", latched)
            self.pub_cue = self.create_publisher(ModeEvent, "/protocol/mode_cue", 10)
            self.create_subscription(ModeEvent, "/protocol/mode_key", self.on_key, 10)
            info = SessionInfo(stamp=self.get_clock().now().to_msg(), **plan.info())
            self.pub_session.publish(info)
            self.create_timer(0.01, self.tick)

        def now_s(self) -> float:
            return self.get_clock().now().nanoseconds * 1e-9

        def on_key(self, msg: "ModeEvent") -> None:
            if msg.source == "arrival":
                self.clock_logic.arrived(msg.stamp.sec + msg.stamp.nanosec * 1e-9, msg.station_index)

        def tick(self) -> None:
            cue = self.clock_logic.tick(self.now_s())
            if cue is None:
                return
            self.pub_cue.publish(ModeEvent(
                stamp=to_time(cue["stamp"]), mode=cue["mode"], source=cue["source"],
                station_index=cue["station_index"], hold_id=cue["hold_id"],
                sampled_hold_s=cue["sampled_hold_s"] if math.isfinite(cue["sampled_hold_s"]) else math.nan))
            self.get_logger().info(f"cue {cue}")
            if self.audio:
                subprocess.Popen(self.audio)

    rclpy.init()
    node = SchedulerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
