"""Dry run of the protocol stack: scheduler (no audio) + teleop + recorder profile print.

  ros2 launch us_intent_scheduler protocol_dry_run.launch.py session_id:=dry seed:=1 \
      distribution:=uniform n_stations:=4
The scheduler prints the session plan and waits for 'yes' in its terminal (emulate_tty).
No robot motion: nothing in this launch commands the robot.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    args = ["session_id", "seed", "distribution", "n_stations"]
    return LaunchDescription([
        *[DeclareLaunchArgument(a) for a in args],
        LogInfo(msg="DRY RUN: no robot motion. Keep the physical e-stop within reach anyway."),
        Node(package="us_intent_scheduler", executable="scheduler_node", output="screen",
             emulate_tty=True,
             parameters=[{"session_id": LaunchConfiguration("session_id"),
                          "seed": LaunchConfiguration("seed"),
                          "distribution": LaunchConfiguration("distribution"),
                          "dry_run": True}]),
        Node(package="us_intent_teleop", executable="teleop_node", output="screen",
             emulate_tty=True, parameters=[{"n_stations": LaunchConfiguration("n_stations")}]),
    ])
