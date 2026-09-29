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

import json
import time

import cv2
from cv_bridge import CvBridge, CvBridgeError
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
            'annotated_image_topic',
            '/grounding_dino/annotated_image',
        )
        self.declare_parameter('camera_frame', 'front_camera')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('model_config_path', '')
        self.declare_parameter('model_checkpoint_path', '')
        self.declare_parameter('device', 'auto')
        self.declare_parameter('box_threshold', 0.35)
        self.declare_parameter('text_threshold', 0.25)
        self.declare_parameter('detection_rate_hz', 1.0)
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

        self._bridge = CvBridge()
        self._targets = []
        self._latest_rgb = None
        self._latest_depth = None
        self._camera_info = None
        self._latest_rgb_receipt = None
        self._latest_depth_receipt = None
        self._last_processed_rgb_receipt = None
        self._warning_times = {}

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._detections_publisher = self.create_publisher(
            String,
            self.get_parameter('detections_topic').value,
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
        self.create_timer(1.0 / rate, self._process_latest_frame)
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
        for name in ('camera_frame', 'map_frame'):
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
        self._targets = targets
        if targets:
            self.get_logger().info(
                'Detection targets updated: {}'.format(', '.join(targets))
            )
        else:
            self.get_logger().info('Detection targets cleared')

    def _rgb_callback(self, message):
        try:
            image = self._bridge.imgmsg_to_cv2(
                message, desired_encoding='rgb8'
            )
        except CvBridgeError as error:
            self._warn_throttled('rgb_conversion', str(error))
            return
        self._latest_rgb = np.ascontiguousarray(image)
        self._latest_rgb_receipt = time.monotonic()

    def _depth_callback(self, message):
        try:
            image = self._bridge.imgmsg_to_cv2(
                message, desired_encoding='32FC1'
            )
        except CvBridgeError as error:
            self._warn_throttled('depth_conversion', str(error))
            return
        self._latest_depth = np.ascontiguousarray(
            image, dtype=np.float32
        )
        self._latest_depth_receipt = time.monotonic()

    def _camera_info_callback(self, message):
        self._camera_info = message

    def _warn_throttled(self, key, message, period=5.0):
        now = time.monotonic()
        last_time = self._warning_times.get(key)
        if last_time is None or now - last_time >= period:
            self.get_logger().warning(message)
            self._warning_times[key] = now

    def _frames_ready(self):
        if (
                self._latest_rgb is None
                or self._latest_depth is None
                or self._camera_info is None):
            self._warn_throttled(
                'missing_camera_data',
                'Waiting for RGB, aligned depth, and camera intrinsics',
            )
            return False

        now = time.monotonic()
        maximum_age = float(
            self.get_parameter('maximum_frame_age_sec').value
        )
        if (
                now - self._latest_rgb_receipt > maximum_age
                or now - self._latest_depth_receipt > maximum_age):
            self._warn_throttled(
                'stale_camera_data',
                'RGB or depth data is stale; skipping detection',
            )
            return False

        maximum_offset = float(
            self.get_parameter('maximum_pair_offset_sec').value
        )
        frame_offset = abs(
            self._latest_rgb_receipt - self._latest_depth_receipt
        )
        if frame_offset > maximum_offset:
            self._warn_throttled(
                'unpaired_camera_data',
                'RGB and depth receipt times differ by {:.3f} seconds'.format(
                    frame_offset
                ),
            )
            return False
        if self._last_processed_rgb_receipt == self._latest_rgb_receipt:
            return False
        return True

    def _scaled_intrinsics(self, image_width, image_height):
        camera_info = self._camera_info
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

    def _latest_map_transform(self, camera_frame):
        map_frame = self.get_parameter('map_frame').value
        try:
            return self._tf_buffer.lookup_transform(
                map_frame,
                camera_frame,
                Time(),
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
            image_width, image_height):
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
            float(self.get_parameter('minimum_depth_m').value),
            float(self.get_parameter('maximum_depth_m').value),
            float(self.get_parameter('depth_center_fraction').value),
        )

        result = {
            'label': detection['label'],
            'requested_target': self._matched_target(
                detection['label'], self._targets
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
            message = self._bridge.cv2_to_imgmsg(
                annotated, encoding='rgb8'
            )
        except CvBridgeError as error:
            self._warn_throttled('annotation_conversion', str(error))
            return
        message.header.stamp = stamp
        message.header.frame_id = frame
        self._annotated_publisher.publish(message)

    def _process_latest_frame(self):
        if not self._targets or not self._frames_ready():
            return

        image_rgb = self._latest_rgb.copy()
        depth_image = self._latest_depth.copy()
        self._last_processed_rgb_receipt = self._latest_rgb_receipt
        if image_rgb.shape[:2] != depth_image.shape:
            self._warn_throttled(
                'image_shape',
                'RGB and depth dimensions differ; aligned depth is required',
            )
            return

        box_threshold = float(
            self.get_parameter('box_threshold').value
        )
        text_threshold = float(
            self.get_parameter('text_threshold').value
        )
        try:
            raw_detections = self._detector.detect(
                image_rgb,
                self._targets,
                box_threshold,
                text_threshold,
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
                image_width, image_height
            )
        except (IndexError, TypeError, ValueError) as error:
            self._warn_throttled(
                'camera_intrinsics',
                'Camera intrinsics are invalid: {}'.format(error),
            )
            return

        camera_frame = self.get_parameter('camera_frame').value
        map_frame = self.get_parameter('map_frame').value
        map_transform = self._latest_map_transform(camera_frame)
        try:
            detections = [
                self._process_detection(
                    detection,
                    depth_image,
                    intrinsics,
                    map_transform,
                    image_width,
                    image_height,
                )
                for detection in raw_detections
            ]
        except ValueError as error:
            self._warn_throttled(
                'depth_geometry',
                'Depth coordinate calculation failed: {}'.format(error),
            )
            return

        stamp = self.get_clock().now().to_msg()
        payload = {
            'stamp': {
                'sec': stamp.sec,
                'nanosec': stamp.nanosec,
            },
            'requested_targets': list(self._targets),
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
        self._publish_annotated_image(
            image_rgb, detections, stamp, camera_frame
        )


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
