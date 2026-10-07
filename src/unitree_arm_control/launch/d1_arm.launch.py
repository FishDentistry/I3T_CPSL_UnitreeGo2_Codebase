"""Launch the Unitree D1 arm wrapper."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Create a passive-by-default D1 arm launch description."""
    command_topic = LaunchConfiguration('command_topic')
    feedback_topic = LaunchConfiguration('feedback_topic')
    joint_states_topic = LaunchConfiguration('joint_states_topic')
    gripper_closed_degrees = LaunchConfiguration(
        'gripper_closed_degrees'
    )
    gripper_open_degrees = LaunchConfiguration('gripper_open_degrees')
    gripper_max_travel_m = LaunchConfiguration('gripper_max_travel_m')
    commanding_enabled = LaunchConfiguration('commanding_enabled')
    require_fresh_feedback = LaunchConfiguration('require_fresh_feedback')
    feedback_timeout_sec = LaunchConfiguration('feedback_timeout_sec')
    enforce_joint_limits = LaunchConfiguration('enforce_joint_limits')
    lay_down_tolerance_degrees = LaunchConfiguration(
        'lay_down_tolerance_degrees'
    )
    lay_down_timeout_sec = LaunchConfiguration('lay_down_timeout_sec')
    lay_down_required_samples = LaunchConfiguration(
        'lay_down_required_samples'
    )
    trajectory_command_rate_hz = LaunchConfiguration(
        'trajectory_command_rate_hz'
    )
    trajectory_command_mode = LaunchConfiguration(
        'trajectory_command_mode'
    )
    trajectory_goal_tolerance_radians = LaunchConfiguration(
        'trajectory_goal_tolerance_radians'
    )
    trajectory_gripper_tolerance_m = LaunchConfiguration(
        'trajectory_gripper_tolerance_m'
    )
    trajectory_goal_timeout_sec = LaunchConfiguration(
        'trajectory_goal_timeout_sec'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'command_topic', default_value='/arm_Command'
        ),
        DeclareLaunchArgument(
            'feedback_topic', default_value='/arm_Feedback'
        ),
        DeclareLaunchArgument(
            'joint_states_topic', default_value='/joint_states'
        ),
        DeclareLaunchArgument(
            'gripper_closed_degrees',
            default_value='0.0',
            description='J6 feedback value representing closed fingers.',
        ),
        DeclareLaunchArgument(
            'gripper_open_degrees',
            default_value='30.0',
            description='J6 feedback value representing open fingers.',
        ),
        DeclareLaunchArgument(
            'gripper_max_travel_m',
            default_value='0.03',
            description='Maximum URDF travel of each gripper finger.',
        ),
        DeclareLaunchArgument(
            'commanding_enabled',
            default_value='false',
            description='Allow services to publish physical arm commands.',
        ),
        DeclareLaunchArgument(
            'require_fresh_feedback',
            default_value='true',
            description='Reject commands when valid arm feedback is stale.',
        ),
        DeclareLaunchArgument(
            'feedback_timeout_sec',
            default_value='2.0',
            description='Maximum feedback age allowed before a command.',
        ),
        DeclareLaunchArgument(
            'enforce_joint_limits',
            default_value='true',
            description='Validate J0-J5 against D1-550 mechanical limits.',
        ),
        DeclareLaunchArgument(
            'lay_down_tolerance_degrees',
            default_value='2.0',
            description='Maximum per-joint lay-down position error.',
        ),
        DeclareLaunchArgument(
            'lay_down_timeout_sec',
            default_value='15.0',
            description='Timeout for each lay-down operation phase.',
        ),
        DeclareLaunchArgument(
            'lay_down_required_samples',
            default_value='3',
            description='Consecutive in-tolerance samples before release.',
        ),
        DeclareLaunchArgument(
            'trajectory_command_rate_hz',
            default_value='10.0',
            description='Rate used to stream interpolated trajectory points.',
        ),
        DeclareLaunchArgument(
            'trajectory_command_mode',
            default_value='1',
            description=(
                'D1 all-joint mode for trajectory samples: 0 is 10 Hz '
                'smoothing and 1 is firmware trajectory smoothing.'
            ),
        ),
        DeclareLaunchArgument(
            'trajectory_goal_tolerance_radians',
            default_value='0.01',
            description='Default final tolerance for arm joints.',
        ),
        DeclareLaunchArgument(
            'trajectory_gripper_tolerance_m',
            default_value='0.005',
            description='Default final tolerance for the gripper joint.',
        ),
        DeclareLaunchArgument(
            'trajectory_goal_timeout_sec',
            default_value='3.0',
            description='Time allowed for final feedback convergence.',
        ),
        Node(
            package='unitree_arm_control',
            executable='d1_arm_controller',
            name='d1_arm_controller',
            output='screen',
            parameters=[{
                'command_topic': command_topic,
                'feedback_topic': feedback_topic,
                'joint_states_topic': joint_states_topic,
                'gripper_closed_degrees': gripper_closed_degrees,
                'gripper_open_degrees': gripper_open_degrees,
                'gripper_max_travel_m': gripper_max_travel_m,
                'commanding_enabled': commanding_enabled,
                'require_fresh_feedback': require_fresh_feedback,
                'feedback_timeout_sec': feedback_timeout_sec,
                'enforce_joint_limits': enforce_joint_limits,
                'lay_down_tolerance_degrees': (
                    lay_down_tolerance_degrees
                ),
                'lay_down_timeout_sec': lay_down_timeout_sec,
                'lay_down_required_samples': lay_down_required_samples,
                'trajectory_command_rate_hz': (
                    trajectory_command_rate_hz
                ),
                'trajectory_command_mode': trajectory_command_mode,
                'trajectory_goal_tolerance_radians': (
                    trajectory_goal_tolerance_radians
                ),
                'trajectory_gripper_tolerance_m': (
                    trajectory_gripper_tolerance_m
                ),
                'trajectory_goal_timeout_sec': (
                    trajectory_goal_timeout_sec
                ),
            }],
        ),
    ])
