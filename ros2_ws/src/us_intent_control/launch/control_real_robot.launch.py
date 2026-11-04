"""REAL ROBOT. Started only by the human operator (docs/SAFETY.md). Requires
i_have_read_safety_md:=true; the node loads configs/safety.yaml, refuses any null or
out-of-range limit, prints the active limits and waits for the operator to type 'yes'.
Keep the physical e-stop within reach. Claude never runs this file.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _check(context, *args, **kwargs):
    if LaunchConfiguration("i_have_read_safety_md").perform(context) != "true":
        raise RuntimeError("Refusing to start: pass i_have_read_safety_md:=true after reading "
                           "docs/SAFETY.md and completing its human checklist.")
    return [LogInfo(msg="REAL ROBOT. Physical e-stop within reach. Confirm the limits in the "
                        "control node terminal by typing 'yes'."),
            Node(package="us_intent_control", executable="control_node", output="screen",
                 emulate_tty=True, parameters=[{"real_robot": True, "i_have_read_safety_md": True}])]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument("i_have_read_safety_md", default_value="false"),
        OpaqueFunction(function=_check),
    ])
