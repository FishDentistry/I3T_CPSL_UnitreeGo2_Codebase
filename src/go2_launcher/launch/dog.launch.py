from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
import os

def generate_launch_description():
    base_urdf_file = os.path.join(
        get_package_share_directory('dog_utilities'),
        'urdf',
        'go2_with_realsense.urdf'
    )
    arm_urdf_file = os.path.join(
        get_package_share_directory('dog_utilities'),
        'urdf',
        'go2_with_realsense_and_arm.urdf'
    )

    collect_realsense = LaunchConfiguration('collect_realsense')
    internal_board_ip = LaunchConfiguration('internal_board_ip')
    launch_arm = LaunchConfiguration('launch_arm')
    arm_command_enabled = LaunchConfiguration('arm_command_enabled')
    arm_trajectory_command_mode = LaunchConfiguration(
        'arm_trajectory_command_mode'
    )
    arm_trajectory_goal_tolerance_radians = LaunchConfiguration(
        'arm_trajectory_goal_tolerance_radians'
    )
    grasp_execution_enabled = LaunchConfiguration(
        'grasp_execution_enabled'
    )

    arm_launch_file = os.path.join(
        get_package_share_directory('unitree_arm_control'),
        'launch',
        'd1_arm.launch.py'
    )
    moveit_launch_file = os.path.join(
        get_package_share_directory('unitree_arm_control'),
        'launch',
        'd1_moveit.launch.py'
    )

    return LaunchDescription([

        DeclareLaunchArgument('collect_realsense', default_value='true'),

        DeclareLaunchArgument(
            'launch_arm',
            default_value='false',
            description='Launch the D1 arm wrapper and MoveIt.'
        ),

        DeclareLaunchArgument(
            'arm_command_enabled',
            default_value='false',
            description=(
                'Allow the D1 arm wrapper to publish physical commands.'
            )
        ),

        DeclareLaunchArgument(
            'arm_trajectory_command_mode',
            default_value='1',
            description=(
                'D1 command mode used for MoveIt trajectory samples.'
            )
        ),

        DeclareLaunchArgument(
            'arm_trajectory_goal_tolerance_radians',
            default_value='0.01',
            description=(
                'Final per-joint tolerance for D1 MoveIt trajectories.'
            )
        ),

        DeclareLaunchArgument(
            'grasp_execution_enabled',
            default_value='false',
            description=(
                'Allow semantic grasp commands to execute the guarded '
                'grasp, hold, release, and retreat sequence without lifting.'
            )
        ),

        DeclareLaunchArgument(
            'internal_board_ip',
            default_value='192.168.123.161',
            description=(
                'IP address of the robot internal board. The default is its '
                'Ethernet address.'
            )
        ),

        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{
                'robot_description': open(base_urdf_file).read()
            }],
            output='screen',
            condition=UnlessCondition(launch_arm)
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{
                'robot_description': open(arm_urdf_file).read()
            }],
            condition=IfCondition(launch_arm),
            output='screen'
        ),
        Node(
            package='dog_utilities',
            executable='transformUpdater',
            name='dog_transform_updater',
            output='screen'
        ),
        Node(
            package='dog_utilities',
            executable='lidarScanRelay',
            name='lidar_scan_relay',
            output='screen'
        ),
        Node(
            package='dog_utilities',
            executable='cmdVelTranslator',
            name='cmd_vel_translator',
            output='screen',
            parameters=[{
                'internal_board_ip': internal_board_ip
            }]
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(arm_launch_file),
            condition=IfCondition(launch_arm),
            launch_arguments={
                'commanding_enabled': arm_command_enabled,
                'trajectory_command_mode': arm_trajectory_command_mode,
                'trajectory_goal_tolerance_radians': (
                    arm_trajectory_goal_tolerance_radians
                ),
            }.items()
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(moveit_launch_file),
            condition=IfCondition(launch_arm),
            launch_arguments={
                'use_rviz': 'false',
                'allow_trajectory_execution': 'true',
                'grasp_execution_enabled': grasp_execution_enabled,
            }.items()
        ),
        Node(
            package='intel_realsense_functions',
            executable='getCameraFrames',
            name='get_intel_realsense_frames',
            output='screen',
            condition=IfCondition(collect_realsense)
        ),
        Node(
            package='pointcloud_to_laserscan',
            executable='pointcloud_to_laserscan_node',
            name='pc_to_scan',
            output='screen',
            remappings=[
                ('cloud_in', '/onboard_lidar_point_cloud2'),
                ('scan', '/onboard_lidar_scan')
            ],
            parameters=[{
                'target_frame': '',  # or your robot base frame
                'transform_tolerance': 0.01,
                'min_height': -0.1,
                'max_height': 0.1,
                'angle_min': -3.14,
                'angle_max': 3.14,
                'angle_increment': 0.008,
                'scan_time': 0.1,
                'range_min': 1.5,
                'range_max': 10.0,
                'use_inf': True,
                'inf_epsilon': 1.0
            }]
        ),
    ])
