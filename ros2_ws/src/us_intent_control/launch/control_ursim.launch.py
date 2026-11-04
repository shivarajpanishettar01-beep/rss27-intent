"""us_intent_control against URSim (simulated controller; start URSim and the UR driver per their
documentation). Recorded B-mode must be replayed (ros2 bag play) so the B-mode watchdog is fed."""
from launch import LaunchDescription
from launch.actions import LogInfo
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        LogInfo(msg="URSIM: simulated robot only. Replay recorded B-mode with ros2 bag play."),
        Node(package="us_intent_control", executable="control_node", output="screen",
             emulate_tty=True, parameters=[{"real_robot": False}]),
    ])
