from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
import os

def generate_launch_description():
    urdf_file = os.path.join(
        get_package_share_directory('dog_utilities'),
        'urdf',
        'go2_with_realsense.urdf'
    )

    collect_realsense = LaunchConfiguration('collect_realsense')
    internal_board_ip = LaunchConfiguration('internal_board_ip')
    launch_arm = LaunchConfiguration('launch_arm')

    arm_launch_file = os.path.join(
        get_package_share_directory('unitree_arm_control'),
        'launch',
        'd1_arm.launch.py'
    )

    return LaunchDescription([

        DeclareLaunchArgument('collect_realsense', default_value='true'),

        DeclareLaunchArgument(
            'launch_arm',
            default_value='false',
            description='Launch the D1 arm wrapper in passive mode.'
        ),
        
        DeclareLaunchArgument(
            'internal_board_ip',
            default_value='192.168.123.161',
            description='IP address of the robot internal board (string). Default is default Ethernet IP of internal board.'
        ),

        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{'robot_description': open(urdf_file).read()}],
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
            condition=IfCondition(launch_arm)
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
