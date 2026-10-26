from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, TimerAction)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    world = LaunchConfiguration('world')
    gui = LaunchConfiguration('gui')
    use_sim_time = LaunchConfiguration('use_sim_time')

    world_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('metr4202_test_worlds'), '/launch/', world, '.launch.py'
        ]),
        launch_arguments={'gui': gui}.items(),
    )

    # Nav2 + SLAM, camera and explorer all live in stack.launch.py so the same
    # file can be used with any world and on the real robot
    stack_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('team10_bringup'), '/launch/stack.launch.py'
        ]),
        launch_arguments={'use_sim_time': use_sim_time}.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument('world', default_value='marker_gauntlet',
                              description='horseshoe | marker_gauntlet | slam_stress'),
        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument('use_sim_time', default_value='True'),

        GroupAction([world_launch]),
        # Gazebo needs about 10 s to spawn the robot before Nav2 can see /odom
        TimerAction(period=10.0, actions=[GroupAction([stack_launch])]),
    ])
