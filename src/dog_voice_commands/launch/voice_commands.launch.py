"""Launch the owner-verified voice command node.

Parameters are loaded in this order, later sources overriding earlier ones:

1. The packaged defaults (``config/voice_commands.yaml``), or ``params_file``.
2. The per-machine settings file
   ``~/.local/share/dog_voice_commands/params.yaml``, if it exists. This is
   where a robot's microphone and tuned threshold belong, so they persist
   across launches without editing the repository.
3. Launch arguments, but only those given a non-empty value.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

USER_PARAMS_FILE = os.path.expanduser(
    '~/.local/share/dog_voice_commands/params.yaml')


def _as_bool(value):
    return value.strip().lower() in ('1', 'true', 'yes', 'on')


def _make_node(context):
    params_file = LaunchConfiguration('params_file').perform(context)
    parameters = [params_file]
    if os.path.isfile(USER_PARAMS_FILE):
        parameters.append(USER_PARAMS_FILE)

    overrides = {}
    audio_device = LaunchConfiguration('audio_device').perform(context)
    if audio_device:
        overrides['audio_device'] = audio_device
    for name in ('dry_run', 'speaker_verification'):
        value = LaunchConfiguration(name).perform(context)
        if value:
            overrides[name] = _as_bool(value)
    if overrides:
        parameters.append(overrides)

    return [Node(
        package='dog_voice_commands',
        executable='voice_command_node',
        name='voice_command_node',
        # The namespace must be set explicitly, even to ''. Without it,
        # launch_ros writes the override dict above under the '/**' wildcard
        # key, and rclpy applies wildcard parameters *before* the
        # 'voice_command_node' entries in the YAML files, so the files would
        # silently win over launch arguments such as dry_run:=true.
        namespace='',
        output='screen',
        parameters=parameters,
    )]


def generate_launch_description():
    default_params = os.path.join(
        get_package_share_directory('dog_voice_commands'),
        'config', 'voice_commands.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file', default_value=default_params,
            description='Base YAML file with voice_command_node parameters.'),
        DeclareLaunchArgument(
            'audio_device', default_value='',
            description='Microphone index or name substring. Empty = use '
                        'the parameter files.'),
        DeclareLaunchArgument(
            'dry_run', default_value='',
            description='true/false. Empty = use the parameter files.'),
        DeclareLaunchArgument(
            'speaker_verification', default_value='',
            description='true/false. Empty = use the parameter files.'),
        OpaqueFunction(function=_make_node),
    ])
