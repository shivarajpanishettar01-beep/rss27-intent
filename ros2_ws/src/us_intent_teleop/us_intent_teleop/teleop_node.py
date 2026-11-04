"""Keyboard teleop over preset stations; key press publishes /protocol/mode_key; motion gated on key. M5.

Thin ROS 2 wrapper around usintent.protocol.teleop.TeleopGate. Keys: 's' = SWEEP, 'l' = LIFT
(from the terminal, raw mode). After a key press it publishes ONE target station index on
/teleop/target_station (std_msgs/Int32); whatever executes motion (the user's preset-waypoint
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
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Int32
    from us_intent_msgs.msg import ModeEvent

    from usintent.protocol.teleop import TeleopGate

    class TeleopNode(Node):
        def __init__(self) -> None:
            super().__init__("us_intent_teleop")
            n = int(self.declare_parameter("n_stations", 0).value)
            if n <= 0:
                raise SystemExit("parameter n_stations is required")
            self.gate = TeleopGate(n_stations=n)
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
            cmd = self.gate.motion_command(ev["stamp"], self.gate.station + 1)
            self.pub_target.publish(Int32(data=cmd["target_station"]))

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
