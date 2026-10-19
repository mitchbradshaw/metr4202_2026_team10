"""Start only the frontier explorer node.

    ros2 launch exploration exploration_only.launch.py

Values come from team10_bringup/config/params.yaml, then <config>.yaml on top.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    args = [
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              description='Use /clock from Gazebo'),
        DeclareLaunchArgument('config', default_value='sim',
                              description='sim | rig: overrides loaded on top of params.yaml'),
    ]

    # Each inner list is joined into one file path. team10_bringup is found at
    # runtime; it already depends on this package, so no package.xml entry.
    params_files = [
        [FindPackageShare('team10_bringup'), '/config/params.yaml'],
        [FindPackageShare('team10_bringup'), '/config/', LaunchConfiguration('config'), '.yaml'],
    ]

    explorer_node = Node(
        package='exploration',
        executable='frontier_explorer',
        name='frontier_explorer',
        output='screen',
        parameters=params_files + [{
            'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool),
        }],
    )

    return LaunchDescription(args + [explorer_node])
