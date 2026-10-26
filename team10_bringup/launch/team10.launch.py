"""Start the camera nodes and the explorer. No Gazebo here, so this
runs on top of whatever world is already up, or on the real robot.
SLAM and Nav2 are expected to be running already (started separately).
The defaults (use_sim_time false, config rig) are for the real robot.

    ros2 launch team10_bringup team10.launch.py
    ros2 launch team10_bringup team10.launch.py use_sim_time:=true config:=sim

Our nodes read config/params.yaml, then config/<config>.yaml on top of it.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    config = LaunchConfiguration('config')

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
        DeclareLaunchArgument('use_sim_time', default_value='false',
                              description='true in Gazebo, false on the real robot'),
        DeclareLaunchArgument('config', default_value='rig',
                              description='sim | rig: overrides loaded after params.yaml'),

        camera_launch,
        explorer_launch,
    ])
