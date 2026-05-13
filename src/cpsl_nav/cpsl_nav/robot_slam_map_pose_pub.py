#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration

from geometry_msgs.msg import PoseStamped
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer, TransformListener
from tf2_ros import TransformException


class RobotSlamMapPosePublisher(Node):
    def __init__(self):
        super().__init__("robot_slam_pose_publisher")

        # Frames
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")

        self.map_frame = self.get_parameter("map_frame").value
        self.base_frame = self.get_parameter("base_frame").value

        # Publisher
        self.pose_pub = self.create_publisher(
            PoseStamped,
            "/robot_slam_map_pose",
            10
        )

        # TF listener maintains the TF buffer internally
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Trigger publishing whenever a TF message arrives
        self.tf_sub = self.create_subscription(
            TFMessage,
            "/tf",
            self.tf_callback,
            10
        )

        self.get_logger().info(
            f"Publishing latest TF pose {self.map_frame} -> {self.base_frame} "
            f"on /robot_slam_pose whenever /tf updates"
        )

    def tf_callback(self, msg: TFMessage):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                Time(),  # latest available transform
                timeout=Duration(seconds=0.05)
            )

            pose_msg = PoseStamped()
            pose_msg.header.stamp = transform.header.stamp
            pose_msg.header.frame_id = self.map_frame

            pose_msg.pose.position.x = transform.transform.translation.x
            pose_msg.pose.position.y = transform.transform.translation.y
            pose_msg.pose.position.z = transform.transform.translation.z

            pose_msg.pose.orientation = transform.transform.rotation

            self.pose_pub.publish(pose_msg)

        except TransformException as e:
            self.get_logger().warn(
                f"Could not get transform {self.map_frame} -> {self.base_frame}: {e}",
                throttle_duration_sec=2.0
            )


def main(args=None):
    rclpy.init(args=args)

    node = RobotSlamMapPosePublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()