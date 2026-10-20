"""Start the camera nodes and the explorer. No Gazebo here, so this
runs on top of whatever world is already up, or on the real robot. 
SLAM and Nav2 are expected to be running already (started separately).
use_sim_time default set to false, launch with argument true if running a sim.

    ros2 launch team10_bringup team10.launch.py
    ros2 launch team10_bringup team10.launch.py use_sim_time:=false \
        use_compressed:=true image_topic:=/camera/image_raw/compressed
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')


    camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('mapping_pkg'), '/launch/camera_only.launch.py'
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'image_topic': LaunchConfiguration('image_topic'),
            'use_compressed': LaunchConfiguration('use_compressed'),
        }.items(),
    )

    explorer_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('exploration'), '/launch/exploration_only.launch.py'
        ]),
        launch_arguments={'use_sim_time': use_sim_time}.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false',
                              description='true in Gazebo, false on the real robot'),
        DeclareLaunchArgument('image_topic', default_value='/camera/image_raw'),
        DeclareLaunchArgument('use_compressed', default_value='false',
                              description='true when image_topic is a CompressedImage'),

        camera_launch,
        explorer_launch,
    ])
