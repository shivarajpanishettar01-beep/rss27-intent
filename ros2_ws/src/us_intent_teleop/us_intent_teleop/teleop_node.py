"""Keyboard teleop over preset stations; key press publishes /protocol/mode_key; motion gated on key. M5.

Thin ROS 2 wrapper around usintent.protocol.teleop.TeleopGate. It rebuilds the scheduler's plan
from the same session parameters (session_id, seed, distribution, block_id, operator_id,
phantom_id, n_episodes, episode_index), so both nodes agree on station order, exit modes,
speeds and tilt/twist. Keys: 's' = SWEEP, 'l' = LIFT (terminal, raw mode); only the cued key is
accepted. After it, the node publishes ONE target station index on /teleop/target_station
(std_msgs/Int32) and the full command (speed, tilt, twist) as JSON on /teleop/motion_command
(std_msgs/String); whatever executes motion (the user's preset-waypoint
teleop, the mock robot, or the M9 controller with all docs/SAFETY.md gates) reports arrival on
/teleop/arrived (std_msgs/Int32), which this node republishes as a mode_key event with
source "arrival". This node never commands the robot directly.
Not tested in a ROS environment from the cloud session (see reports/M5.md).
"""
import select
import sys
import termios
import tty


def main() -> None:
    import json

    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Int32, String
    from us_intent_msgs.msg import ModeEvent

    from usintent.config import load, require
    from usintent.protocol.scheduler import plan_session
    from usintent.protocol.teleop import TeleopGate

    class TeleopNode(Node):
        def __init__(self) -> None:
            super().__init__("us_intent_teleop")
            p = {k: self.declare_parameter(k, v).value for k, v in {
                "session_id": "", "seed": -1, "distribution": "", "block_id": 0,
                "operator_id": "", "phantom_id": "", "n_episodes": 1, "episode_index": 0}.items()}
            if not p["session_id"] or p["seed"] < 0 or not p["distribution"] or not p["phantom_id"]:
                raise SystemExit("session_id, seed, distribution and phantom_id are required")
            proto = load("protocol")
            plan = plan_session(proto, p["session_id"], int(p["seed"]), p["distribution"],
                                int(p["block_id"]), p["operator_id"], p["phantom_id"],
                                int(p["n_episodes"]))
            self.gate = TeleopGate(plan=plan.episodes[int(p["episode_index"])],
                                   n_stations=int(require(proto, "preset_stations")))
            self.pub_motion = self.create_publisher(String, "/teleop/motion_command", 10)
            self.pub_key = self.create_publisher(ModeEvent, "/protocol/mode_key", 10)
            self.pub_target = self.create_publisher(Int32, "/teleop/target_station", 10)
            self.create_subscription(Int32, "/teleop/arrived", self.on_arrived, 10)
            self.create_timer(0.01, self.poll_keyboard)
            self.publish_event(self.gate.arrived_initial())

        def publish_event(self, ev: dict) -> None:
            t = self.get_clock().now().to_msg()
            self.pub_key.publish(ModeEvent(stamp=t, mode=ev["mode"], source=ev["source"],
                                           station_index=ev["station_index"],
                                           hold_id=ev["hold_id"], sampled_hold_s=float("nan")))

        def poll_keyboard(self) -> None:
            if not select.select([sys.stdin], [], [], 0)[0]:
                return
            ch = sys.stdin.read(1)
            ev = self.gate.key(self.get_clock().now().nanoseconds * 1e-9, ch)
            if ev is None:
                return
            self.publish_event(ev)
            cmd = self.gate.motion_command(ev["stamp"])
            self.pub_target.publish(Int32(data=cmd["target_station"]))
            self.pub_motion.publish(String(data=json.dumps(cmd)))

        def on_arrived(self, msg: "Int32") -> None:
            self.publish_event(self.gate.arrived(self.get_clock().now().nanoseconds * 1e-9, msg.data))

    old = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    rclpy.init()
    node = TeleopNode()
    try:
        rclpy.spin(node)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
