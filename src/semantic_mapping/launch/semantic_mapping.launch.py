# Copyright 2026 CPSL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Launch Grounding DINO and semantic mapping without the camera node."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import EnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Create the detector and semantic-mapper launch description."""
    model_config_path = LaunchConfiguration('model_config_path')
    model_checkpoint_path = LaunchConfiguration('model_checkpoint_path')
    device = LaunchConfiguration('device')
    detection_rate_hz = LaunchConfiguration('detection_rate_hz')
    camera_frame = LaunchConfiguration('camera_frame')
    map_frame = LaunchConfiguration('map_frame')
    detections_topic = LaunchConfiguration('detections_topic')
    minimum_confidence = LaunchConfiguration('minimum_confidence')
    confirmation_observations = LaunchConfiguration(
        'confirmation_observations'
    )

    default_dino_root = [
        EnvironmentVariable('HOME'),
        '/.local/share/go2_groundingdino/GroundingDINO',
    ]

    return LaunchDescription([
        DeclareLaunchArgument(
            'model_config_path',
            default_value=default_dino_root + [
                '/groundingdino/config/GroundingDINO_SwinT_OGC.py'
            ],
            description='Grounding DINO model configuration file.',
        ),
        DeclareLaunchArgument(
            'model_checkpoint_path',
            default_value=default_dino_root + [
                '/dino_weights/groundingdino_swint_ogc.pth'
            ],
            description='Grounding DINO checkpoint file.',
        ),
        DeclareLaunchArgument(
            'device',
            default_value='auto',
            description='Grounding DINO Torch device.',
        ),
        DeclareLaunchArgument(
            'detection_rate_hz',
            default_value='5.0',
            description='Maximum detector scheduling frequency.',
        ),
        DeclareLaunchArgument(
            'camera_frame',
            default_value='front_camera',
            description='Optical TF frame used by Grounding DINO.',
        ),
        DeclareLaunchArgument(
            'map_frame',
            default_value='map',
            description='Shared global frame for detections and the map.',
        ),
        DeclareLaunchArgument(
            'detections_topic',
            default_value='/grounding_dino/detection_array',
            description='Structured topic connecting the two nodes.',
        ),
        DeclareLaunchArgument(
            'minimum_confidence',
            default_value='0.60',
            description='Minimum score accepted by semantic mapping.',
        ),
        DeclareLaunchArgument(
            'confirmation_observations',
            default_value='3',
            description='Observations required to confirm an object.',
        ),
        Node(
            package='intel_realsense_functions',
            executable='groundingDinoNode',
            name='grounding_dino_detector',
            output='screen',
            parameters=[{
                'model_config_path': model_config_path,
                'model_checkpoint_path': model_checkpoint_path,
                'device': device,
                'detection_rate_hz': detection_rate_hz,
                'camera_frame': camera_frame,
                'map_frame': map_frame,
                'structured_detections_topic': detections_topic,
            }],
        ),
        Node(
            package='semantic_mapping',
            executable='semanticMappingNode',
            name='semantic_mapping',
            output='screen',
            parameters=[{
                'detections_topic': detections_topic,
                'map_frame': map_frame,
                'minimum_confidence': minimum_confidence,
                'confirmation_observations': confirmation_observations,
            }],
        ),
    ])
