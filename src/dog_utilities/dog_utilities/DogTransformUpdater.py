# Copyright (c) 2024, RoboVerse community
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import json
import logging
import os
import threading
import asyncio

from cv_bridge import CvBridge


#from scripts.go2_constants import ROBOT_CMD, RTC_TOPIC
#from scripts.go2_func import gen_command, gen_mov_command
#from scripts.go2_lidar_decoder import update_meshes_for_cloud2
#from scripts.go2_math import get_robot_joints
#from scripts.go2_camerainfo import load_camera_info
#from scripts.webrtc_driver import Go2Connection

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
#from rclpy.qos_overriding_options import QoSOverridingOptions

from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import Twist, TransformStamped, PoseStamped
from go2_interfaces.msg import Go2State, IMU
from unitree_go.msg import LowState, VoxelMapCompressed, WebRtcReq
from sensor_msgs.msg import PointCloud2, PointField, JointState, Joy
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CameraInfo


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RobotBaseNode(Node):

    def __init__(self):
        super().__init__('dog_transform_updater')

        self.conn = {}
        qos_profile = QoSProfile(depth=10)
     
        best_effort_qos = qos_profile#QoSProfile(
            #reliability=QoSReliabilityPolicy.BEST_EFFORT,
            #history=QoSHistoryPolicy.KEEP_LAST,
            #depth=1
        #)
        

        self.joint_pub = []
        self.go2_state_pub = []
        self.go2_lidar_pub = []
        self.go2_odometry_pub = []
        self.imu_pub = []
        self.img_pub = []
        self.camera_info_pub = []
        self.voxel_pub = []

        if True:
            self.joint_pub.append(self.create_publisher(
                JointState, 'joint_states', qos_profile))
            self.go2_state_pub.append(self.create_publisher(
                Go2State, 'go2_states', qos_profile))
            self.go2_lidar_pub.append(
                self.create_publisher(
                    PointCloud2,
                    'onboard_lidar_point_cloud2',
                    best_effort_qos,))
                    #qos_overriding_options=QoSOverridingOptions.with_default_policies()))
            self.go2_odometry_pub.append(
                self.create_publisher(Odometry, 'odom', qos_profile))
            self.imu_pub.append(self.create_publisher(IMU, 'imu', qos_profile))
            # if self.enable_video:
            #     self.img_pub.append(
            #         self.create_publisher(
            #             Image,
            #             'camera/image_raw',
            #             best_effort_qos,))
            #             #qos_overriding_options=QoSOverridingOptions.with_default_policies()))
            #     self.camera_info_pub.append(
            #         self.create_publisher(
            #             CameraInfo,
            #             'camera/camera_info',
            #             best_effort_qos,))
                        #qos_overriding_options=QoSOverridingOptions.with_default_policies()))
            # if self.publish_raw_voxel:
            #     self.voxel_pub.append(
            #         self.create_publisher(
            #             VoxelMapCompressed,
            #             '/utlidar/voxel_map_compressed',
            #             best_effort_qos))


        self.broadcaster = TransformBroadcaster(self, qos=qos_profile)

        self.bridge = CvBridge()
        #self.camera_info = load_camera_info()

        self.robot_cmd_vel = {}
        self.robot_odom = {}
        self.robot_low_cmd = {}
        self.robot_sport_state = {}
        self.robot_lidar = {}
        self.webrtc_msgs = asyncio.Queue()

        self.joy_state = Joy()

        if True:
            self.create_subscription(
                Twist,
                'cmd_vel_out',
                lambda msg: self.cmd_vel_cb(msg, "0"),
                qos_profile)
            #self.create_subscription(
            #    WebRtcReq,
            #    'webrtc_req',
            #    lambda msg: self.webrtc_req_cb(msg, "0"),
            #    qos_profile)

        self.create_subscription(
            Joy,
            'joy',
            self.joy_cb,
            qos_profile)

        # Support for CycloneDDS (EDU version via ethernet)
        if True:
            self.create_subscription(
                LowState,
                'lowstate',
                self.publish_joint_state_cyclonedds,
                qos_profile)

            self.create_subscription(
                PoseStamped,
                '/utlidar/robot_pose',
                self.publish_body_poss_cyclonedds,
                qos_profile)

            self.create_subscription(
                PointCloud2,
                '/utlidar/cloud',
                self.publish_lidar_cyclonedds,
                qos_profile)

        #self.timer = self.create_timer(0.1, self.timer_callback)
        #self.timer_lidar = self.create_timer(0.5, self.timer_callback_lidar)


    # def cmd_vel_cb(self, msg, robot_num):
    #     x = msg.linear.x
    #     y = msg.linear.y
    #     z = msg.angular.z

    #     # Allow omni-directional movement
    #     if x != 0.0 or y != 0.0 or z != 0.0:
    #         self.robot_cmd_vel[robot_num] = gen_mov_command(
    #             round(x, 2), round(y, 2), round(z, 2))

    # def webrtc_req_cb(self, msg, robot_num):
    #     parameter_str = msg.parameter if msg.parameter else ""
    #     try:
    #         parameter = json.loads(parameter_str)
    #     except ValueError as e:
    #         self.get_logger().error(f"Invalid JSON in WebRTC request: {e}")
    #         parameter = parameter_str
    #     payload = gen_command(msg.api_id, parameter, msg.topic, msg.id)
    #     self.get_logger().info(f"Received WebRTC request: {payload[:50]}")
    #     self.webrtc_msgs.put_nowait(payload)

    def joy_cb(self, msg):
        self.joy_state = msg

    def publish_body_poss_cyclonedds(self, msg):
        odom_trans = TransformStamped()
        odom_trans.header.stamp = self.get_clock().now().to_msg()
        odom_trans.header.frame_id = 'odom'
        odom_trans.child_frame_id = "base_link"
        odom_trans.transform.translation.x = msg.pose.position.x
        odom_trans.transform.translation.y = msg.pose.position.y
        odom_trans.transform.translation.z = msg.pose.position.z + 0.07
        odom_trans.transform.rotation.x = msg.pose.orientation.x
        odom_trans.transform.rotation.y = msg.pose.orientation.y
        odom_trans.transform.rotation.z = msg.pose.orientation.z
        odom_trans.transform.rotation.w = msg.pose.orientation.w
        self.broadcaster.sendTransform(odom_trans)

    def publish_joint_state_cyclonedds(self, msg):
        joint_state = JointState()
        joint_state.header.stamp = self.get_clock().now().to_msg()
        joint_state.name = [
            'FL_hip_joint',
            'FL_thigh_joint',
            'FL_calf_joint',
            'FR_hip_joint',
            'FR_thigh_joint',
            'FR_calf_joint',
            'RL_hip_joint',
            'RL_thigh_joint',
            'RL_calf_joint',
            'RR_hip_joint',
            'RR_thigh_joint',
            'RR_calf_joint',
        ]
        joint_state.position = [
            msg.motor_state[3].q, msg.motor_state[4].q, msg.motor_state[5].q,
            msg.motor_state[0].q, msg.motor_state[1].q, msg.motor_state[2].q,
            msg.motor_state[9].q, msg.motor_state[10].q, msg.motor_state[11].q,
            msg.motor_state[6].q, msg.motor_state[7].q, msg.motor_state[8].q,
        ]
        self.joint_pub[0].publish(joint_state)

    def publish_lidar_cyclonedds(self, msg):
        msg.header = Header(frame_id="radar")
        msg.header.stamp = self.get_clock().now().to_msg()
        self.go2_lidar_pub[0].publish(msg)

    # def joy_cmd(self, robot_num):
    #     if robot_num in self.conn and robot_num in self.robot_cmd_vel and self.robot_cmd_vel[
    #             robot_num] is not None:
    #         self.get_logger().info("Move")
    #         self.conn[robot_num].data_channel.send(
    #             self.robot_cmd_vel[robot_num])
    #         self.robot_cmd_vel[robot_num] = None

    #     if robot_num in self.conn and self.joy_state.buttons and self.joy_state.buttons[1]:
    #         self.get_logger().info("Stand down")
    #         stand_down_cmd = gen_command(ROBOT_CMD["StandDown"])
    #         self.conn[robot_num].data_channel.send(stand_down_cmd)

    #     if robot_num in self.conn and self.joy_state.buttons and self.joy_state.buttons[0]:
    #         self.get_logger().info("Stand up")
    #         stand_up_cmd = gen_command(ROBOT_CMD["StandUp"])
    #         self.conn[robot_num].data_channel.send(stand_up_cmd)
    #         move_cmd = gen_command(ROBOT_CMD['BalanceStand'])
    #         self.conn[robot_num].data_channel.send(move_cmd)


def main(args=None):
    rclpy.init(args=args)

    baseNode = RobotBaseNode()

    rclpy.spin(baseNode)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    baseNode.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
