"""us_intent_control against MOCK hardware (start the UR driver with mock hardware yourself,
following its documentation). Never moves a physical robot. docs/SAFETY.md applies."""
from launch import LaunchDescription
from launch.actions import LogInfo
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        LogInfo(msg="MOCK HARDWARE: us_intent_control with real_robot:=false."),
        Node(package="us_intent_control", executable="control_node", output="screen",
             emulate_tty=True, parameters=[{"real_robot": False}]),
    ])
