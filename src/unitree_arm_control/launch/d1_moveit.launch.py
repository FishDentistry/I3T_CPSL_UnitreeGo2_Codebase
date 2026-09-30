"""Launch MoveIt 2 for the mounted Unitree D1 arm."""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _load_text(path):
    with open(path, 'r') as stream:
        return stream.read()


def _load_yaml(path):
    with open(path, 'r') as stream:
        return yaml.safe_load(stream)


def generate_launch_description():
    """Start move_group and an optional RViz MotionPlanning display."""
    package_share = get_package_share_directory('unitree_arm_control')
    dog_share = get_package_share_directory('dog_utilities')
    config_directory = os.path.join(package_share, 'config')

    robot_description = {
        'robot_description': _load_text(os.path.join(
            dog_share,
            'urdf',
            'go2_with_realsense_and_arm.urdf',
        )),
    }
    robot_description_semantic = {
        'robot_description_semantic': _load_text(os.path.join(
            config_directory,
            'd1.srdf',
        )),
    }
    robot_description_kinematics = {
        'robot_description_kinematics': _load_yaml(os.path.join(
            config_directory,
            'kinematics.yaml',
        )),
    }
    robot_description_planning = {
        'robot_description_planning': _load_yaml(os.path.join(
            config_directory,
            'joint_limits.yaml',
        )),
    }

    ompl_pipeline = {
        'move_group': {
            'planning_plugin': 'ompl_interface/OMPLPlanner',
            'request_adapters': (
                'default_planner_request_adapters/'
                'AddTimeOptimalParameterization '
                'default_planner_request_adapters/FixWorkspaceBounds '
                'default_planner_request_adapters/FixStartStateBounds '
                'default_planner_request_adapters/FixStartStateCollision '
                'default_planner_request_adapters/'
                'FixStartStatePathConstraints'
            ),
            'start_state_max_bounds_error': 0.1,
        },
    }
    ompl_pipeline['move_group'].update(_load_yaml(os.path.join(
        config_directory,
        'ompl_planning.yaml',
    )))

    controller_configuration = {
        'moveit_simple_controller_manager': _load_yaml(os.path.join(
            config_directory,
            'moveit_controllers.yaml',
        )),
        'moveit_controller_manager': (
            'moveit_simple_controller_manager/'
            'MoveItSimpleControllerManager'
        ),
    }
    allow_execution = ParameterValue(
        LaunchConfiguration('allow_trajectory_execution'),
        value_type=bool,
    )
    trajectory_execution = {
        'allow_trajectory_execution': allow_execution,
        'moveit_manage_controllers': False,
        'trajectory_execution.allowed_execution_duration_scaling': 2.0,
        'trajectory_execution.allowed_goal_duration_margin': 3.0,
        'trajectory_execution.allowed_start_tolerance': 0.035,
        'trajectory_execution.controller_connection_timeout': 15.0,
    }
    planning_scene_monitor = {
        'publish_planning_scene': True,
        'publish_geometry_updates': True,
        'publish_state_updates': True,
        'publish_transforms_updates': True,
    }

    move_group = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        name='move_group',
        output='screen',
        parameters=[
            robot_description,
            robot_description_semantic,
            robot_description_kinematics,
            robot_description_planning,
            ompl_pipeline,
            controller_configuration,
            trajectory_execution,
            planning_scene_monitor,
        ],
        condition=IfCondition(LaunchConfiguration('start_move_group')),
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='d1_moveit_rviz',
        output='log',
        arguments=[
            '-d',
            os.path.join(config_directory, 'moveit.rviz'),
        ],
        parameters=[
            robot_description,
            robot_description_semantic,
            robot_description_kinematics,
            robot_description_planning,
            ompl_pipeline,
        ],
        condition=IfCondition(LaunchConfiguration('use_rviz')),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'start_move_group',
            default_value='true',
            description='Start move_group; disable for an RViz-only launch.',
        ),
        DeclareLaunchArgument(
            'use_rviz',
            default_value='true',
            description='Open RViz with the MoveIt MotionPlanning panel.',
        ),
        DeclareLaunchArgument(
            'allow_trajectory_execution',
            default_value='false',
            description=(
                'Allow MoveIt to send plans to the physical D1 controller.'
            ),
        ),
        move_group,
        rviz,
    ])
