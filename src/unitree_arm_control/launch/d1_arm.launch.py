"""Launch the Unitree D1 arm wrapper."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Create a passive-by-default D1 arm launch description."""
    command_topic = LaunchConfiguration('command_topic')
    feedback_topic = LaunchConfiguration('feedback_topic')
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

    return LaunchDescription([
        DeclareLaunchArgument(
            'command_topic', default_value='/arm_Command'
        ),
        DeclareLaunchArgument(
            'feedback_topic', default_value='/arm_Feedback'
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
        Node(
            package='unitree_arm_control',
            executable='d1_arm_controller',
            name='d1_arm_controller',
            output='screen',
            parameters=[{
                'command_topic': command_topic,
                'feedback_topic': feedback_topic,
                'commanding_enabled': commanding_enabled,
                'require_fresh_feedback': require_fresh_feedback,
                'feedback_timeout_sec': feedback_timeout_sec,
                'enforce_joint_limits': enforce_joint_limits,
                'lay_down_tolerance_degrees': (
                    lay_down_tolerance_degrees
                ),
                'lay_down_timeout_sec': lay_down_timeout_sec,
                'lay_down_required_samples': lay_down_required_samples,
            }],
        ),
    ])
