"""Loads an arm checkpoint; intent gating per arm; publishes /policy/chunk and /policy/contact_cmd. M8.

Thin rclpy wrapper around usintent.policies.server. Parameters: arm (e.g. DP-1), checkpoint
(path to model.pt), log (JSONL path). The policy rate is configs/rig.yaml rates_hz.policy (null
refuses). The scheduler-mode subscription exists only for intent-aware arms
(usintent.policies.server.subscriptions). This node never commands the robot: the M9 control
node consumes /policy/chunk under docs/SAFETY.md. Not run in a ROS environment from the cloud.
"""
from pathlib import Path


def main() -> None:
    import numpy as np
    import rclpy
    from geometry_msgs.msg import PoseStamped, WrenchStamped
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from us_intent_msgs.msg import ActionChunk, ModeEvent

    from usintent.config import load, require
    from usintent.policies.evaluate import load_policy
    from usintent.policies.server import MODE_TOPIC, PolicyServer, subscriptions

    class PolicyServerNode(Node):
        def __init__(self) -> None:
            super().__init__("us_intent_policy_server")
            arm = self.declare_parameter("arm", "").value
            ckpt = self.declare_parameter("checkpoint", "").value
            log = self.declare_parameter("log", "").value
            rig = load("rig")
            rate = float(require(rig, "rates_hz.policy"))
            k = int(load("data")["history_frames_k"])
            self.server = PolicyServer(load_policy(arm, Path(ckpt)), Path(log) if log else None)
            self.k, self.frames, self.states, self.wrenches, self.times = k, [], [], [], []
            topics = subscriptions(arm, rig["topics"])
            self.create_subscription(Image, require(rig, "topics.bmode"), self.on_img, 10)
            self.create_subscription(PoseStamped, require(rig, "topics.tcp_pose"), self.on_pose, 10)
            self.create_subscription(WrenchStamped, require(rig, "topics.wrench"), self.on_wrench, 10)
            if MODE_TOPIC in topics:
                self.create_subscription(ModeEvent, MODE_TOPIC,
                                         lambda m: self.server.set_intent_mode(m.mode), 10)
            self.pub_chunk = self.create_publisher(ActionChunk, "/policy/chunk", 10)
            self.pub_cmd = self.create_publisher(ModeEvent, "/policy/contact_cmd", 10)
            self.dt = 1.0 / rate
            self.state, self.wrench = np.zeros(12), np.zeros(6)
            self.create_timer(self.dt, self.tick)

        def on_img(self, m: "Image") -> None:
            self.frames.append(np.frombuffer(m.data, np.uint8).reshape(m.height, m.width))
            self.states.append(self.state.copy())
            self.wrenches.append(self.wrench.copy())
            self.times.append(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)
            for buf in (self.frames, self.states, self.wrenches, self.times):
                del buf[:-self.k]

        def on_pose(self, m: "PoseStamped") -> None:
            from scipy.spatial.transform import Rotation

            q = m.pose.orientation
            self.state[:3] = [m.pose.position.x, m.pose.position.y, m.pose.position.z]
            self.state[3:6] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_rotvec()

        def on_wrench(self, m: "WrenchStamped") -> None:
            f, t = m.wrench.force, m.wrench.torque
            self.wrench[:] = [f.x, f.y, f.z, t.x, t.y, t.z]

        def tick(self) -> None:
            if len(self.frames) < self.k:
                return
            hist = {"bmode": np.stack(self.frames), "state": np.stack(self.states),
                    "wrench": np.stack(self.wrenches), "time": np.array(self.times),
                    "dt": np.array([self.dt])}
            now = self.get_clock().now()
            chunk, m_hat = self.server.step(now.nanoseconds * 1e-9, hist)
            self.pub_chunk.publish(ActionChunk(stamp=now.to_msg(), arm_id=self.server.policy.arm_id,
                                               horizon_s=chunk.shape[0] * self.dt, dt_s=self.dt,
                                               displacements=chunk.ravel().tolist()))
            self.pub_cmd.publish(ModeEvent(stamp=now.to_msg(), mode=m_hat, source="policy",
                                           station_index=-1, hold_id=-1, sampled_hold_s=float("nan")))

    rclpy.init()
    node = PolicyServerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
