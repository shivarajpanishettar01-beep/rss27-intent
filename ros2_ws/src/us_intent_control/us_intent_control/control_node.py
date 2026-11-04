"""Hybrid force/impedance controller + CBF filter. SAFETY-CRITICAL: docs/SAFETY.md. Mock/URSim only. M9.

Thin rclpy wrapper around usintent.control. Before creating any publisher it runs
usintent.control.safety.preflight: configs/safety.yaml must be complete and inside its sanity
ranges; with real_robot:=true it also requires i_have_read_safety_md:=true and a typed "yes".
configs/control.yaml (gains, set-points, the UR driver command topic/type) must be filled; the
UR driver controller interface was NOT read from its docs in the cloud session, so cmd_topic and
cmd_type are human-filled config values, and only geometry_msgs/msg/TwistStamped is supported.
On any fault (stale/NaN inputs, lost F/T or B-mode, watchdog miss, exception) it publishes zero
velocity, latches FAULT on /control/fault and waits for a human reset (service-free: restart
the node after typing RESET in its terminal). Claude never runs this against the real robot.
"""


def main() -> None:
    import numpy as np
    import rclpy
    from geometry_msgs.msg import PoseStamped, TwistStamped, WrenchStamped
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from sensor_msgs.msg import Image
    from std_msgs.msg import String
    from us_intent_msgs.msg import ActionChunk, FilterEvent, ModeEvent

    from usintent.config import load, require
    from usintent.control.controller import ControlLoop, ControlParams
    from usintent.control.safety import preflight

    class ControlNode(Node):
        def __init__(self) -> None:
            super().__init__("us_intent_control")
            real = bool(self.declare_parameter("real_robot", False).value)
            flag = bool(self.declare_parameter("i_have_read_safety_md", False).value)
            limits = preflight(load("safety"), flag, real)
            ctl = load("control")
            if require(ctl, "cmd_type") != "geometry_msgs/msg/TwistStamped":
                raise SystemExit("only geometry_msgs/msg/TwistStamped commands are supported")
            rig = load("rig")
            self.loop = ControlLoop(ControlParams.from_config(ctl), limits)
            self.dt = limits.watchdog_period_s / 2.0
            self.pose, self.force, self.disp = None, 0.0, 0.0
            self.chunk, self.m_hat = np.zeros((1, 5)), "HOLD"
            self.pub_cmd = self.create_publisher(TwistStamped, require(ctl, "cmd_topic"), 1)
            latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.pub_fault = self.create_publisher(String, "/control/fault", latched)
            self.pub_ev = self.create_publisher(FilterEvent, "/control/filter_event", 50)
            self.create_subscription(PoseStamped, require(rig, "topics.tcp_pose"), self.on_pose, 10)
            self.create_subscription(WrenchStamped, require(rig, "topics.wrench"), self.on_ft, 10)
            self.create_subscription(Image, require(rig, "topics.bmode"), self.on_img, 10)
            self.create_subscription(ActionChunk, "/policy/chunk", self.on_chunk, 10)
            self.create_subscription(ModeEvent, "/policy/contact_cmd", self.on_cmd, 10)
            self.n_events = 0
            self.create_timer(self.dt, self.tick)

        def now(self) -> float:
            return self.get_clock().now().nanoseconds * 1e-9

        def on_pose(self, m: "PoseStamped") -> None:
            from scipy.spatial.transform import Rotation

            q = m.pose.orientation
            p = [m.pose.position.x, m.pose.position.y, m.pose.position.z]
            self.pose = np.array(p + list(Rotation.from_quat([q.x, q.y, q.z, q.w]).as_rotvec()))
            self.loop.latch.stamp("pose", self.now(), self.pose)

        def on_ft(self, m: "WrenchStamped") -> None:
            self.force = float(m.wrench.force.z)
            self.loop.latch.stamp("ft", self.now(), [self.force])

        def on_img(self, m: "Image") -> None:
            self.loop.latch.stamp("bmode", self.now())

        def on_chunk(self, m: "ActionChunk") -> None:
            c = np.asarray(m.displacements, dtype=np.float64).reshape(-1, 5)
            self.chunk = c
            self.loop.latch.stamp("policy", self.now(), c)

        def on_cmd(self, m: "ModeEvent") -> None:
            self.m_hat = m.mode

        def tick(self) -> None:
            t = self.now()
            pose = self.pose if self.pose is not None else np.full(6, np.nan)
            cmd = self.loop.step(t, pose, self.force, self.disp, self.m_hat, self.chunk, self.dt)
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z = map(float, cmd[:3])
            msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z = map(float, cmd[3:])
            self.pub_cmd.publish(msg)
            for ev in self.loop.filter.events[self.n_events:]:
                self.pub_ev.publish(FilterEvent(stamp=msg.header.stamp, constraint=ev.constraint,
                                                raw_command=list(ev.raw_command),
                                                filtered_command=list(ev.filtered_command)))
            self.n_events = len(self.loop.filter.events)
            if self.loop.latch.faulted:
                self.pub_fault.publish(String(data=self.loop.latch.reason))

    rclpy.init()
    node = ControlNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
