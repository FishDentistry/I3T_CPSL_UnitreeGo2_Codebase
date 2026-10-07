#!/usr/bin/env python3
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

"""Detect text-specified objects and localize them with aligned depth."""

from concurrent.futures import ThreadPoolExecutor
import json
import threading
import time

import cv2
from cv_bridge import CvBridge, CvBridgeError
from intel_realsense_interfaces.msg import GroundedDetection
from intel_realsense_interfaces.msg import GroundedDetectionArray
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from std_msgs.msg import String
from tf2_ros import Buffer
from tf2_ros import TransformException
from tf2_ros import TransformListener

from intel_realsense_functions.detection_geometry import (
    camera_point_from_depth,
)
from intel_realsense_functions.detection_geometry import (
    parse_detection_targets,
)
from intel_realsense_functions.detection_geometry import transform_point
from intel_realsense_functions.grounding_dino_backend import (
    GroundingDinoBackend,
)
from intel_realsense_functions.latest_frame_buffer import LatestFrameBuffer


class GroundingDinoNode(Node):
    """Run Grounding DINO against the latest aligned RealSense frames."""

    def __init__(self):
        super().__init__('grounding_dino_detector')

        self.declare_parameter('rgb_topic', '/realsense_rgb_image')
        self.declare_parameter('depth_topic', '/realsense_depth_image')
        self.declare_parameter(
            'camera_info_topic', '/depth_camera_intrinsics'
        )
        self.declare_parameter('targets_topic', '/detection_targets')
        self.declare_parameter(
            'detections_topic', '/grounding_dino/detections'
        )
        self.declare_parameter(
            'structured_detections_topic',
            '/grounding_dino/detection_array',
        )
        self.declare_parameter(
            'annotated_image_topic',
            '/grounding_dino/annotated_image',
        )
        self.declare_parameter(
            'camera_frame', 'camera_color_optical_frame'
        )
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('model_config_path', '')
        self.declare_parameter('model_checkpoint_path', '')
        self.declare_parameter('device', 'auto')
        self.declare_parameter('box_threshold', 0.35)
        self.declare_parameter('text_threshold', 0.25)
        self.declare_parameter('detection_rate_hz', 5.0)
        self.declare_parameter('maximum_frame_age_sec', 1.0)
        self.declare_parameter('maximum_pair_offset_sec', 0.25)
        self.declare_parameter('minimum_depth_m', 0.15)
        self.declare_parameter('maximum_depth_m', 6.0)
        self.declare_parameter('depth_center_fraction', 0.5)

        config_path = self.get_parameter('model_config_path').value
        checkpoint_path = self.get_parameter(
            'model_checkpoint_path'
        ).value
        device = self.get_parameter('device').value
        self._validate_parameters()

        self.get_logger().info('Loading Grounding DINO model')
        self._detector = GroundingDinoBackend(
            config_path,
            checkpoint_path,
            device,
        )
        self.get_logger().info(
            'Grounding DINO loaded on {}'.format(self._detector.device)
        )

        self._input_bridge = CvBridge()
        self._output_bridge = CvBridge()
        self._targets = []
        self._targets_revision = 0
        self._state_lock = threading.Lock()
        self._frame_buffer = LatestFrameBuffer()
        self._inference_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix='grounding-dino',
        )
        self._shutting_down = False
        self._warning_times = {}
        self._warning_lock = threading.Lock()

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._detections_publisher = self.create_publisher(
            String,
            self.get_parameter('detections_topic').value,
            10,
        )
        self._structured_detections_publisher = self.create_publisher(
            GroundedDetectionArray,
            self.get_parameter('structured_detections_topic').value,
            10,
        )
        self._annotated_publisher = self.create_publisher(
            Image,
            self.get_parameter('annotated_image_topic').value,
            2,
        )

        self.create_subscription(
            String,
            self.get_parameter('targets_topic').value,
            self._targets_callback,
            10,
        )
        self.create_subscription(
            Image,
            self.get_parameter('rgb_topic').value,
            self._rgb_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            self.get_parameter('depth_topic').value,
            self._depth_callback,
            qos_profile_sensor_data,
        )
        camera_info_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            CameraInfo,
            self.get_parameter('camera_info_topic').value,
            self._camera_info_callback,
            camera_info_qos,
        )

        rate = float(self.get_parameter('detection_rate_hz').value)
        self._processing_timer = self.create_timer(
            1.0 / rate, self._process_latest_frame
        )
        self.get_logger().info(
            'Waiting for targets on {}'.format(
                self.get_parameter('targets_topic').value
            )
        )

    def _validate_parameters(self):
        required_paths = (
            'model_config_path',
            'model_checkpoint_path',
        )
        for name in required_paths:
            if not str(self.get_parameter(name).value).strip():
                raise RuntimeError('{} must be provided'.format(name))

        positive_parameters = (
            'detection_rate_hz',
            'maximum_frame_age_sec',
            'maximum_pair_offset_sec',
            'minimum_depth_m',
            'maximum_depth_m',
            'depth_center_fraction',
        )
        for name in positive_parameters:
            if float(self.get_parameter(name).value) <= 0.0:
                raise RuntimeError('{} must be positive'.format(name))
        if (
                float(self.get_parameter('minimum_depth_m').value)
                >= float(self.get_parameter('maximum_depth_m').value)):
            raise RuntimeError(
                'minimum_depth_m must be less than maximum_depth_m'
            )
        center_fraction = float(
            self.get_parameter('depth_center_fraction').value
        )
        if center_fraction > 1.0:
            raise RuntimeError('depth_center_fraction must not exceed 1.0')
        for name in ('box_threshold', 'text_threshold'):
            threshold = float(self.get_parameter(name).value)
            if not 0.0 <= threshold <= 1.0:
                raise RuntimeError('{} must be in [0, 1]'.format(name))
        for name in (
                'camera_frame',
                'map_frame',
                'structured_detections_topic'):
            if not str(self.get_parameter(name).value).strip():
                raise RuntimeError('{} must not be empty'.format(name))

    def _targets_callback(self, message):
        try:
            targets = parse_detection_targets(message.data)
        except ValueError as error:
            self.get_logger().warning(
                'Rejected detection targets: {}'.format(error)
            )
            return
        if len(targets) > 20:
            self.get_logger().warning(
                'Rejected detection targets: at most 20 are allowed'
            )
            return
        with self._state_lock:
            if targets != self._targets:
                self._targets = targets
                self._targets_revision += 1
        if targets:
            self.get_logger().info(
                'Detection targets updated: {}'.format(', '.join(targets))
            )
        else:
            self.get_logger().info('Detection targets cleared')

    def _rgb_callback(self, message):
        try:
            image = self._input_bridge.imgmsg_to_cv2(
                message, desired_encoding='rgb8'
            )
        except CvBridgeError as error:
            self._warn_throttled('rgb_conversion', str(error))
            return
        self._frame_buffer.update_rgb(
            np.ascontiguousarray(image),
            time.monotonic(),
            message.header.stamp,
        )

    def _depth_callback(self, message):
        try:
            image = self._input_bridge.imgmsg_to_cv2(
                message, desired_encoding='32FC1'
            )
        except CvBridgeError as error:
            self._warn_throttled('depth_conversion', str(error))
            return
        self._frame_buffer.update_depth(
            np.ascontiguousarray(image, dtype=np.float32),
            time.monotonic(),
            message.header.stamp,
        )

    def _camera_info_callback(self, message):
        self._frame_buffer.update_camera_info(message)

    def _warn_throttled(self, key, message, period=5.0):
        now = time.monotonic()
        with self._warning_lock:
            last_time = self._warning_times.get(key)
            if last_time is None or now - last_time >= period:
                self.get_logger().warning(message)
                self._warning_times[key] = now

    def _take_latest_frame(self):
        maximum_age = float(
            self.get_parameter('maximum_frame_age_sec').value
        )
        maximum_offset = float(
            self.get_parameter('maximum_pair_offset_sec').value
        )
        frame, status, details = self._frame_buffer.claim(
            time.monotonic(), maximum_age, maximum_offset
        )
        if status == 'missing':
            self._warn_throttled(
                'missing_camera_data',
                'Waiting for RGB, aligned depth, and camera intrinsics',
            )
        elif status == 'stale':
            rgb_age, depth_age = details
            self._warn_throttled(
                'stale_camera_data',
                'RGB or depth data is stale; skipping detection '
                '(ages: {:.3f}s RGB, {:.3f}s depth; limit: {:.3f}s)'.format(
                    rgb_age, depth_age, maximum_age
                ),
            )
        elif status == 'unpaired':
            self._warn_throttled(
                'unpaired_camera_data',
                'RGB and depth timestamps differ by {:.3f} seconds'.format(
                    details
                ),
            )
        return frame

    @staticmethod
    def _scaled_intrinsics(camera_info, image_width, image_height):
        calibration_width = camera_info.width or image_width
        calibration_height = camera_info.height or image_height
        scale_x = float(image_width) / float(calibration_width)
        scale_y = float(image_height) / float(calibration_height)
        return (
            float(camera_info.k[0]) * scale_x,
            float(camera_info.k[4]) * scale_y,
            float(camera_info.k[2]) * scale_x,
            float(camera_info.k[5]) * scale_y,
        )

    def _latest_map_transform(
            self, camera_frame, map_frame, acquisition_time):
        try:
            return self._tf_buffer.lookup_transform(
                map_frame,
                camera_frame,
                acquisition_time,
                timeout=Duration(seconds=0.05),
            )
        except TransformException as error:
            self._warn_throttled(
                'map_transform',
                'Map coordinates unavailable: {}'.format(error),
            )
            return None

    @staticmethod
    def _map_point(camera_point, transform):
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        return transform_point(
            (
                camera_point['x'],
                camera_point['y'],
                camera_point['z'],
            ),
            (translation.x, translation.y, translation.z),
            (rotation.x, rotation.y, rotation.z, rotation.w),
        )

    @staticmethod
    def _matched_target(label, targets):
        normalized_label = label.lower()
        for target in targets:
            if target in normalized_label or normalized_label in target:
                return target
        return label

    def _process_detection(
            self, detection, depth_image, intrinsics, map_transform,
            image_width, image_height, targets, depth_settings):
        raw_box = detection['bounding_box']
        box = (
            max(0.0, min(float(image_width - 1), raw_box[0])),
            max(0.0, min(float(image_height - 1), raw_box[1])),
            max(0.0, min(float(image_width - 1), raw_box[2])),
            max(0.0, min(float(image_height - 1), raw_box[3])),
        )
        camera_point = camera_point_from_depth(
            depth_image,
            box,
            intrinsics,
            depth_settings['minimum'],
            depth_settings['maximum'],
            depth_settings['center_fraction'],
        )

        result = {
            'label': detection['label'],
            'requested_target': self._matched_target(
                detection['label'], targets
            ),
            'score': detection['score'],
            'bounding_box_pixels': {
                'x_min': int(round(box[0])),
                'y_min': int(round(box[1])),
                'x_max': int(round(box[2])),
                'y_max': int(round(box[3])),
            },
            'camera_coordinates_m': None,
            'map_coordinates_m': None,
            'depth_sample_count': 0,
        }
        if camera_point is None:
            return result

        result['camera_coordinates_m'] = {
            'x': camera_point['x'],
            'y': camera_point['y'],
            'z': camera_point['z'],
        }
        result['depth_sample_count'] = camera_point[
            'depth_sample_count'
        ]
        result['depth_pixel'] = {
            'u': camera_point['pixel_u'],
            'v': camera_point['pixel_v'],
        }
        if map_transform is not None:
            try:
                map_x, map_y, map_z = self._map_point(
                    camera_point, map_transform
                )
            except ValueError as error:
                self._warn_throttled(
                    'map_transform_value',
                    'Map transform is invalid: {}'.format(error),
                )
            else:
                result['map_coordinates_m'] = {
                    'x': map_x,
                    'y': map_y,
                    'z': map_z,
                }
        return result

    def _publish_annotated_image(self, image_rgb, detections, stamp, frame):
        annotated = image_rgb.copy()
        for detection in detections:
            box = detection['bounding_box_pixels']
            start = (box['x_min'], box['y_min'])
            end = (box['x_max'], box['y_max'])
            cv2.rectangle(annotated, start, end, (255, 0, 0), 2)
            text = '{} {:.2f}'.format(
                detection['label'], detection['score']
            )
            text_y = max(15, box['y_min'] - 5)
            cv2.putText(
                annotated,
                text,
                (box['x_min'], text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
                cv2.LINE_AA,
            )
        try:
            message = self._output_bridge.cv2_to_imgmsg(
                annotated, encoding='rgb8'
            )
        except CvBridgeError as error:
            self._warn_throttled('annotation_conversion', str(error))
            return
        message.header.stamp = stamp
        message.header.frame_id = frame
        self._annotated_publisher.publish(message)

    @staticmethod
    def _structured_detection(detection):
        message = GroundedDetection()
        message.label = detection['label']
        message.requested_target = detection['requested_target']
        message.score = float(detection['score'])

        box = detection['bounding_box_pixels']
        message.x_min = int(box['x_min'])
        message.y_min = int(box['y_min'])
        message.x_max = int(box['x_max'])
        message.y_max = int(box['y_max'])

        camera_position = detection['camera_coordinates_m']
        message.has_camera_position = camera_position is not None
        if camera_position is not None:
            message.camera_position.x = float(camera_position['x'])
            message.camera_position.y = float(camera_position['y'])
            message.camera_position.z = float(camera_position['z'])

        map_position = detection['map_coordinates_m']
        message.has_map_position = map_position is not None
        if map_position is not None:
            message.map_position.x = float(map_position['x'])
            message.map_position.y = float(map_position['y'])
            message.map_position.z = float(map_position['z'])

        message.depth_sample_count = int(detection['depth_sample_count'])
        depth_pixel = detection.get('depth_pixel')
        message.has_depth = depth_pixel is not None
        if depth_pixel is not None:
            message.depth_pixel_u = float(depth_pixel['u'])
            message.depth_pixel_v = float(depth_pixel['v'])
        return message

    def _publish_structured_detections(
            self, stamp, settings, map_transform, detections):
        message = GroundedDetectionArray()
        message.header.stamp = stamp
        message.requested_targets = list(settings['targets'])
        message.camera_frame = settings['camera_frame']
        message.map_frame = settings['map_frame']
        message.map_transform_available = map_transform is not None
        message.detections = [
            self._structured_detection(detection)
            for detection in detections
        ]
        self._structured_detections_publisher.publish(message)

    def _process_latest_frame(self):
        with self._state_lock:
            if self._shutting_down or not self._targets:
                return
            targets = tuple(self._targets)
            targets_revision = self._targets_revision

        frame = self._take_latest_frame()
        if frame is None:
            return

        settings = {
            'targets': targets,
            'targets_revision': targets_revision,
            'box_threshold': float(
                self.get_parameter('box_threshold').value
            ),
            'text_threshold': float(
                self.get_parameter('text_threshold').value
            ),
            'depth': {
                'minimum': float(
                    self.get_parameter('minimum_depth_m').value
                ),
                'maximum': float(
                    self.get_parameter('maximum_depth_m').value
                ),
                'center_fraction': float(
                    self.get_parameter('depth_center_fraction').value
                ),
            },
            'camera_frame': self.get_parameter('camera_frame').value,
            'map_frame': self.get_parameter('map_frame').value,
        }
        try:
            self._inference_executor.submit(
                self._process_claimed_frame, frame, settings
            )
        except RuntimeError:
            self._frame_buffer.release()

    def _process_claimed_frame(self, frame, settings):
        """Run inference in the worker and release the frame claim."""
        try:
            self._run_inference(frame, settings)
        except Exception as error:
            self._warn_throttled(
                'frame_processing',
                'Grounding DINO frame processing failed: {}'.format(error),
            )
        finally:
            self._frame_buffer.release()

    def _run_inference(self, frame, settings):
        """Detect and publish one stable frame snapshot."""
        image_rgb = frame.rgb
        depth_image = frame.depth
        if image_rgb.shape[:2] != depth_image.shape:
            self._warn_throttled(
                'image_shape',
                'RGB and depth dimensions differ; aligned depth is required',
            )
            return

        try:
            raw_detections = self._detector.detect(
                image_rgb,
                settings['targets'],
                settings['box_threshold'],
                settings['text_threshold'],
            )
        except Exception as error:
            self._warn_throttled(
                'model_inference',
                'Grounding DINO inference failed: {}'.format(error),
            )
            return

        image_height, image_width = image_rgb.shape[:2]
        try:
            intrinsics = self._scaled_intrinsics(
                frame.camera_info, image_width, image_height
            )
        except (IndexError, TypeError, ValueError) as error:
            self._warn_throttled(
                'camera_intrinsics',
                'Camera intrinsics are invalid: {}'.format(error),
            )
            return

        camera_frame = settings['camera_frame']
        map_frame = settings['map_frame']
        if frame.stamp_nanoseconds > 0:
            acquisition_time = Time.from_msg(frame.stamp)
        else:
            acquisition_time = Time()
        map_transform = self._latest_map_transform(
            camera_frame, map_frame, acquisition_time
        )
        try:
            detections = [
                self._process_detection(
                    detection,
                    depth_image,
                    intrinsics,
                    map_transform,
                    image_width,
                    image_height,
                    settings['targets'],
                    settings['depth'],
                )
                for detection in raw_detections
            ]
        except ValueError as error:
            self._warn_throttled(
                'depth_geometry',
                'Depth coordinate calculation failed: {}'.format(error),
            )
            return

        with self._state_lock:
            if (
                    self._shutting_down
                    or settings['targets_revision'] != self._targets_revision):
                return

        if frame.stamp_nanoseconds > 0:
            stamp = frame.stamp
        else:
            stamp = self.get_clock().now().to_msg()
        payload = {
            'stamp': {
                'sec': stamp.sec,
                'nanosec': stamp.nanosec,
            },
            'requested_targets': list(settings['targets']),
            'camera_frame': camera_frame,
            'map_frame': map_frame,
            'map_transform_available': map_transform is not None,
            'detections': detections,
        }
        message = String()
        message.data = json.dumps(
            payload, separators=(',', ':'), allow_nan=False
        )
        self._detections_publisher.publish(message)
        self._publish_structured_detections(
            stamp, settings, map_transform, detections
        )
        self._publish_annotated_image(
            image_rgb, detections, stamp, camera_frame
        )

    def destroy_node(self):
        """Stop inference before destroying ROS publishers and resources."""
        with self._state_lock:
            self._shutting_down = True
        if hasattr(self, '_processing_timer'):
            self._processing_timer.cancel()
        if hasattr(self, '_inference_executor'):
            self._inference_executor.shutdown(wait=True)
        return super().destroy_node()


def main(args=None):
    """Run the Grounding DINO RealSense detector."""
    rclpy.init(args=args)
    node = GroundingDinoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
