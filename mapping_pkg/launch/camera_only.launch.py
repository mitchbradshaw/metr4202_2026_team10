"""Start only the ArUco camera node and the marker mapping node.

    ros2 launch mapping_pkg camera_only.launch.py
    ros2 launch mapping_pkg camera_only.launch.py config:=rig use_sim_time:=false

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
                              description='Use /clock from Gazebo or `ros2 bag play --clock`'),
        DeclareLaunchArgument('config', default_value='sim',
                              description='sim | rig: overrides loaded on top of params.yaml'),
    ]

    # Each inner list is joined into one file path. team10_bringup is found at
    # runtime; it already depends on this package, so no package.xml entry.
    params_files = [
        [FindPackageShare('team10_bringup'), '/config/params.yaml'],
        [FindPackageShare('team10_bringup'), '/config/', LaunchConfiguration('config'), '.yaml'],
    ]
    # ParameterValue(..., value_type=bool) makes launch pass a real bool
    # instead of a string, which declare_parameter would otherwise reject.
    use_sim_time = {
        'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool),
    }

    camera_node = Node(
        package='mapping_pkg',
        executable='camera_node',
        name='camera_node',
        output='screen',
        parameters=params_files + [use_sim_time],
    )

    map_markers_node = Node(
        package='mapping_pkg',
        executable='map_markers_node',
        name='map_markers_node',
        output='screen',
        parameters=params_files + [use_sim_time],
    )

    return LaunchDescription(args + [camera_node, map_markers_node])
