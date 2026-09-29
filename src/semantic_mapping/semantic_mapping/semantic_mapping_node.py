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

"""Build and visualize a stable semantic map from grounded detections."""

import math
import threading

from builtin_interfaces.msg import Time as TimeMessage
from intel_realsense_interfaces.msg import GroundedDetectionArray
from intel_realsense_interfaces.msg import SemanticMap
from intel_realsense_interfaces.msg import SemanticObject
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

from semantic_mapping.semantic_tracker import ACTIVE
from semantic_mapping.semantic_tracker import Observation
from semantic_mapping.semantic_tracker import REMOVED
from semantic_mapping.semantic_tracker import SemanticTracker
from semantic_mapping.semantic_tracker import STALE
from semantic_mapping.semantic_tracker import TrackerConfig


class SemanticMappingNode(Node):
    """Fuse grounded detections into stable semantic object identities."""

    def __init__(self):
        super().__init__('semantic_mapping')

        self.declare_parameter(
            'detections_topic', '/grounding_dino/detection_array'
        )
        self.declare_parameter('semantic_map_topic', '/semantic_map')
        self.declare_parameter('markers_topic', '/semantic_map/markers')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('publish_rate_hz', 1.0)
        self.declare_parameter('minimum_confidence', 0.50)
        self.declare_parameter('confirmation_observations', 3)
        self.declare_parameter('observation_merge_distance_m', 0.20)
        self.declare_parameter('association_distance_m', 0.75)
        self.declare_parameter('position_fusion_alpha', 0.25)
        self.declare_parameter('movement_distance_m', 0.35)
        self.declare_parameter('movement_cluster_distance_m', 0.30)
        self.declare_parameter('movement_confirmation_observations', 3)
        self.declare_parameter('relocation_minimum_age_sec', 5.0)
        self.declare_parameter('relocation_maximum_distance_m', 3.0)
        self.declare_parameter('tentative_timeout_sec', 10.0)
        self.declare_parameter('stale_after_sec', 30.0)
        self.declare_parameter('removed_after_sec', 300.0)
        self.declare_parameter('removed_retention_sec', 300.0)
        self.declare_parameter('include_removed_objects', True)
        self.declare_parameter('marker_scale_m', 0.18)

        self._map_frame = str(self.get_parameter('map_frame').value).strip()
        self._publish_rate = float(
            self.get_parameter('publish_rate_hz').value
        )
        self._marker_scale = float(
            self.get_parameter('marker_scale_m').value
        )
        self._include_removed = bool(
            self.get_parameter('include_removed_objects').value
        )
        self._validate_node_parameters()

        config = TrackerConfig(
            minimum_confidence=float(
                self.get_parameter('minimum_confidence').value
            ),
            confirmation_observations=int(
                self.get_parameter('confirmation_observations').value
            ),
            observation_merge_distance=float(
                self.get_parameter('observation_merge_distance_m').value
            ),
            association_distance=float(
                self.get_parameter('association_distance_m').value
            ),
            position_fusion_alpha=float(
                self.get_parameter('position_fusion_alpha').value
            ),
            movement_distance=float(
                self.get_parameter('movement_distance_m').value
            ),
            movement_cluster_distance=float(
                self.get_parameter('movement_cluster_distance_m').value
            ),
            movement_confirmation_observations=int(
                self.get_parameter(
                    'movement_confirmation_observations'
                ).value
            ),
            relocation_minimum_age=float(
                self.get_parameter('relocation_minimum_age_sec').value
            ),
            relocation_maximum_distance=float(
                self.get_parameter('relocation_maximum_distance_m').value
            ),
            tentative_timeout=float(
                self.get_parameter('tentative_timeout_sec').value
            ),
            stale_after=float(
                self.get_parameter('stale_after_sec').value
            ),
            removed_after=float(
                self.get_parameter('removed_after_sec').value
            ),
            removed_retention=float(
                self.get_parameter('removed_retention_sec').value
            ),
        )
        self._tracker = SemanticTracker(config)
        self._tracker_lock = threading.Lock()
        self._published_marker_ids = set()

        output_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._map_publisher = self.create_publisher(
            SemanticMap,
            self.get_parameter('semantic_map_topic').value,
            output_qos,
        )
        self._marker_publisher = self.create_publisher(
            MarkerArray,
            self.get_parameter('markers_topic').value,
            output_qos,
        )
        self.create_subscription(
            GroundedDetectionArray,
            self.get_parameter('detections_topic').value,
            self._detections_callback,
            10,
        )
        self._publish_timer = self.create_timer(
            1.0 / self._publish_rate, self._publish_current_map
        )
        self._clear_rviz_markers()
        self.get_logger().info(
            'Building semantic map in frame {} from {}'.format(
                self._map_frame,
                self.get_parameter('detections_topic').value,
            )
        )

    def _validate_node_parameters(self):
        if not self._map_frame:
            raise RuntimeError('map_frame must not be empty')
        if self._publish_rate <= 0.0:
            raise RuntimeError('publish_rate_hz must be positive')
        if self._marker_scale <= 0.0:
            raise RuntimeError('marker_scale_m must be positive')
        for name in (
                'detections_topic', 'semantic_map_topic', 'markers_topic'):
            if not str(self.get_parameter(name).value).strip():
                raise RuntimeError('{} must not be empty'.format(name))

    def _now_seconds(self):
        return self.get_clock().now().nanoseconds / 1000000000.0

    @staticmethod
    def _stamp_seconds(stamp):
        return float(stamp.sec) + float(stamp.nanosec) / 1000000000.0

    @staticmethod
    def _time_message(timestamp):
        timestamp = max(0.0, float(timestamp))
        seconds = int(math.floor(timestamp))
        nanoseconds = int(round((timestamp - seconds) * 1000000000.0))
        if nanoseconds >= 1000000000:
            seconds += 1
            nanoseconds -= 1000000000
        message = TimeMessage()
        message.sec = seconds
        message.nanosec = nanoseconds
        return message

    def _detections_callback(self, message):
        if message.map_frame and message.map_frame != self._map_frame:
            self.get_logger().warning(
                'Ignoring detections in frame {}; expected {}'.format(
                    message.map_frame, self._map_frame
                )
            )
            return

        timestamp = self._stamp_seconds(message.header.stamp)
        if timestamp <= 0.0:
            timestamp = self._now_seconds()
        observations = []
        for detection in message.detections:
            if not detection.has_map_position:
                continue
            label = detection.requested_target.strip()
            if not label:
                label = detection.label.strip()
            observations.append(Observation(
                label=label,
                position=(
                    detection.map_position.x,
                    detection.map_position.y,
                    detection.map_position.z,
                ),
                confidence=detection.score,
                timestamp=timestamp,
            ))

        with self._tracker_lock:
            summary = self._tracker.update(observations, timestamp)

        if summary.confirmed or summary.moved or summary.relocated:
            self.get_logger().info(
                'Semantic map update: {} confirmed, {} moved, '
                '{} relocated'.format(
                    summary.confirmed, summary.moved, summary.relocated
                )
            )
        if summary.changed:
            self._publish_current_map()

    @staticmethod
    def _status_value(track):
        if track.state == STALE:
            return SemanticObject.STATUS_STALE
        if track.state == REMOVED:
            return SemanticObject.STATUS_REMOVED
        return SemanticObject.STATUS_ACTIVE

    def _semantic_object_message(self, track):
        message = SemanticObject()
        message.object_id = track.object_id
        message.label = track.label
        message.position.x = float(track.position[0])
        message.position.y = float(track.position[1])
        message.position.z = float(track.position[2])
        message.confidence = float(track.confidence)
        message.observation_count = int(track.observation_count)
        message.first_observed = self._time_message(track.first_observed)
        message.last_observed = self._time_message(track.last_observed)
        message.movement_count = int(track.movement_count)
        message.status = self._status_value(track)
        return message

    def _map_message(self, tracks, revision, stamp):
        message = SemanticMap()
        message.header.stamp = stamp
        message.header.frame_id = self._map_frame
        message.revision = int(revision)
        message.objects = [
            self._semantic_object_message(track) for track in tracks
        ]
        return message

    def _base_marker(self, track, stamp, marker_id, namespace):
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = self._map_frame
        marker.ns = namespace
        marker.id = marker_id
        marker.action = Marker.ADD
        marker.pose.position.x = float(track.position[0])
        marker.pose.position.y = float(track.position[1])
        marker.pose.position.z = float(track.position[2])
        marker.pose.orientation.w = 1.0
        return marker

    def _object_markers(self, track, stamp):
        shape_id = track.numeric_id * 2
        text_id = shape_id + 1

        shape = self._base_marker(
            track, stamp, shape_id, 'semantic_objects'
        )
        shape.type = Marker.SPHERE
        shape.scale.x = self._marker_scale
        shape.scale.y = self._marker_scale
        shape.scale.z = self._marker_scale
        if track.state == ACTIVE:
            shape.color.r = 0.10
            shape.color.g = 0.85
            shape.color.b = 0.25
            shape.color.a = 0.90
        else:
            shape.color.r = 0.65
            shape.color.g = 0.65
            shape.color.b = 0.65
            shape.color.a = 0.55

        text = self._base_marker(
            track, stamp, text_id, 'semantic_labels'
        )
        text.type = Marker.TEXT_VIEW_FACING
        text.pose.position.z += self._marker_scale
        text.scale.z = max(0.10, self._marker_scale * 0.65)
        text.color.r = 1.0
        text.color.g = 1.0 if track.state == ACTIVE else 0.75
        text.color.b = 1.0 if track.state == ACTIVE else 0.20
        text.color.a = 0.95
        text.text = '{} [{}]\n{:.2f}, {} observations'.format(
            track.label,
            track.object_id,
            track.confidence,
            track.observation_count,
        )
        return (shape, text)

    def _delete_marker(self, namespace, marker_id, stamp):
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = self._map_frame
        marker.ns = namespace
        marker.id = marker_id
        marker.action = Marker.DELETE
        return marker

    def _marker_array(self, tracks, stamp):
        markers = []
        visible_ids = set()
        for track in tracks:
            if track.state == REMOVED:
                continue
            object_markers = self._object_markers(track, stamp)
            markers.extend(object_markers)
            visible_ids.update(
                (marker.ns, marker.id) for marker in object_markers
            )

        for namespace, marker_id in sorted(
                self._published_marker_ids - visible_ids):
            markers.append(
                self._delete_marker(namespace, marker_id, stamp)
            )
        self._published_marker_ids = visible_ids

        message = MarkerArray()
        message.markers = markers
        return message

    def _publish_current_map(self):
        now = self._now_seconds()
        with self._tracker_lock:
            self._tracker.refresh(now)
            tracks = self._tracker.snapshot(
                include_removed=self._include_removed
            )
            revision = self._tracker.revision
        stamp = self.get_clock().now().to_msg()
        self._map_publisher.publish(
            self._map_message(tracks, revision, stamp)
        )
        self._marker_publisher.publish(
            self._marker_array(tracks, stamp)
        )

    def _clear_rviz_markers(self):
        marker = Marker()
        marker.action = Marker.DELETEALL
        message = MarkerArray()
        message.markers = [marker]
        self._marker_publisher.publish(message)


def main(args=None):
    """Run the semantic mapping node."""
    rclpy.init(args=args)
    node = SemanticMappingNode()
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
