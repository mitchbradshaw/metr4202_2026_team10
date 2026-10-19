"""Start Nav2 + SLAM, the camera nodes and the explorer. No Gazebo here, so this
runs on top of whatever world is already up, or on the real robot.

    ros2 launch team10_bringup stack.launch.py
    ros2 launch team10_bringup stack.launch.py use_sim_time:=false config:=rig

Our nodes read config/params.yaml, then config/<config>.yaml on top of it.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    # Not called params_file: launch arguments are shared with included files,
    # and Nav2 would pick up a params_file meant for our nodes.
    config = LaunchConfiguration('config')

    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('turtlebot3_navigation2'), '/launch/navigation2.launch.py'
        ]),
        launch_arguments={'use_sim_time': use_sim_time, 'slam': 'True'}.items(),
    )

    camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('mapping_pkg'), '/launch/camera_only.launch.py'
        ]),
        launch_arguments={'use_sim_time': use_sim_time, 'config': config}.items(),
    )

    explorer_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('exploration'), '/launch/exploration_only.launch.py'
        ]),
        launch_arguments={'use_sim_time': use_sim_time, 'config': config}.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              description='true in Gazebo, false on the real robot'),
        DeclareLaunchArgument('config', default_value='sim',
                              description='sim | rig: overrides loaded on top of config/params.yaml'),

        nav2_launch,
        camera_launch,
        # give SLAM a few seconds to publish the first map before we look for frontiers
        TimerAction(period=10.0, actions=[explorer_launch]),
    ])
