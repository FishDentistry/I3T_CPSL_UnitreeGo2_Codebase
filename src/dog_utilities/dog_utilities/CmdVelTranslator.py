# import asyncio
# import threading
# import rclpy
# from rclpy.node import Node
# from rclpy.qos import QoSProfile
# from geometry_msgs.msg import Twist
# from scripts.go2_func import gen_mov_command
# from scripts.webrtc_driver import Go2Connection

# class CmdVelTranslator(Node):
#     def __init__(self):
#         super().__init__('cmd_vel_translator')
#         qos_profile = QoSProfile(depth=10)

#         # Declare + read launch-passed parameter (ROS2 Foxy style)
#         self.declare_parameter('internal_board_ip', '192.168.123.255')
#         self.internal_board_ip = (
#             self.get_parameter('internal_board_ip')
#             .get_parameter_value()
#             .string_value
#         )

#         self.webrtcConn = None
#         self.loop = None  # Asyncio event loop reference set later

#         self.get_logger().info(f"cmdVelTranslator using internal_board_ip: {self.internal_board_ip}")

#         self.create_subscription(
#             Twist,
#             'cmd_vel',
#             self.cmd_vel_cb,
#             qos_profile
#         )

#     def cmd_vel_cb(self, msg):
#         if self.webrtcConn is None or self.webrtcConn.data_channel is None:
#             return

#         x = msg.linear.x
#         y = msg.linear.y
#         z = msg.angular.z
#         if x != 0.0 or y != 0.0 or z != 0.0:
#             cmd = gen_mov_command(round(x, 2), round(y, 2), round(z, 2))

#             # Schedule the send on the asyncio loop to avoid thread issues
#             def safe_send():
#                 try:
#                     self.webrtcConn.data_channel.send(cmd)
#                 except Exception as e:
#                     self.get_logger().error(f"Failed to send via data_channel: {e}")

#             if self.loop is not None:
#                 self.loop.call_soon_threadsafe(safe_send)

#     async def run(self):
#         self.get_logger().info("Creating Go2Connection and connecting...")
#         try:
#             self.webrtcConn = Go2Connection(
#                 robot_ip=self.internal_board_ip,
#                 robot_num="1",
#                 token="",
#             )
#             await self.webrtcConn.connect()
#             self.get_logger().info("Successfully connected to the robot!")
#         except Exception as e:
#             self.get_logger().error(f"Connection failed: {e}")

# def main(args=None):
#     rclpy.init(args=args)
#     node = CmdVelTranslator()

#     # Spin ROS in a separate thread
#     def ros_spin():
#         try:
#             rclpy.spin(node)
#         except KeyboardInterrupt:
#             pass
#         finally:
#             rclpy.shutdown()

#     ros_thread = threading.Thread(target=ros_spin, daemon=True)
#     ros_thread.start()

#     # Run asyncio in main thread
#     async def async_main():
#         node.loop = asyncio.get_running_loop()  # Store event loop in node
#         await node.run()

#         # Keep the loop alive to allow aiortc to run background tasks
#         while rclpy.ok():
#             await asyncio.sleep(1)

#     try:
#         asyncio.run(async_main())
#     finally:
#         ros_thread.join()

# if __name__ == '__main__':
#     main()

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import Twist
from std_msgs.msg import String

from scripts.go2_func import gen_mov_command


# The WebRTC connection to the Go2 internal board is owned by the
# go2WebRtcBridge node (Go2WebRtcBridge.py), because the board accepts only
# one client. This node translates cmd_vel into move commands and hands them
# to the bridge on the go2/webrtc_send topic.
DEFAULT_INTERNAL_BOARD_IP = '192.168.123.161'


class CmdVelTranslator(Node):
    def __init__(self):
        super().__init__('cmd_vel_translator')
        qos_profile = QoSProfile(depth=10)

        # Kept only so existing launch files and commands that set it keep
        # working. The robot IP is now configured on go2_webrtc_bridge.
        self.declare_parameter(
            'internal_board_ip',
            DEFAULT_INTERNAL_BOARD_IP
        )
        internal_board_ip = (
            self.get_parameter('internal_board_ip')
            .get_parameter_value()
            .string_value
        )
        if internal_board_ip != DEFAULT_INTERNAL_BOARD_IP:
            self.get_logger().warn(
                "CmdVelTranslator no longer connects to the robot, so its "
                "internal_board_ip parameter has no effect. Set "
                "internal_board_ip on go2_webrtc_bridge instead."
            )

        self.webrtc_pub = self.create_publisher(
            String,
            'go2/webrtc_send',
            qos_profile
        )

        self.get_logger().info(
            "CmdVelTranslator forwarding cmd_vel to go2/webrtc_send"
        )

        self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_vel_cb,
            qos_profile
        )

    def cmd_vel_cb(self, msg):
        x = msg.linear.x
        y = msg.linear.y
        z = msg.angular.z

        if x != 0.0 or y != 0.0 or z != 0.0:
            cmd = gen_mov_command(
                round(x, 2),
                round(y, 2),
                round(z, 2)
            )

            # The bridge sends this on the robot's WebRTC data channel.
            self.webrtc_pub.publish(String(data=cmd))


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelTranslator()

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