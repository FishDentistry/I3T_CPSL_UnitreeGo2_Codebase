"""Single owner of the WebRTC connection to the Go2 internal board.

The internal board accepts one WebRTC client at a time, so every node that
needs to send commands to the robot goes through this bridge instead of
opening its own connection.

Subscriptions (relative names, so they follow the node's namespace):

* ``go2/webrtc_send`` (``std_msgs/String``): a complete, pre-built
  data-channel payload (e.g. from ``gen_mov_command``), sent as-is. Used by
  ``CmdVelTranslator``.
* ``go2/sport_cmd`` (``go2_interfaces/Go2Cmd``): a sport-mode API ID such as
  1009 (Sit). Only IDs listed in ``ROBOT_CMD`` are forwarded. Used by
  ``dog_voice_commands``.
"""

import asyncio
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import String
from go2_interfaces.msg import Go2Cmd

from scripts.go2_constants import ROBOT_CMD
from scripts.go2_func import gen_command
from scripts.webrtc_driver import Go2Connection


class Go2WebRtcBridge(Node):
    def __init__(self):
        super().__init__('go2_webrtc_bridge')
        qos_profile = QoSProfile(depth=10)

        # Default Go2 internal-board Ethernet IP.
        self.declare_parameter(
            'internal_board_ip',
            '192.168.123.161'
        )

        self.internal_board_ip = (
            self.get_parameter('internal_board_ip')
            .get_parameter_value()
            .string_value
        )

        self.webrtcConn = None
        self.loop = None
        self._known_api_ids = {
            api_id: name for name, api_id in ROBOT_CMD.items()
        }

        self.get_logger().info(
            f"Go2WebRtcBridge using internal_board_ip: "
            f"{self.internal_board_ip}"
        )

        self.create_subscription(
            String,
            'go2/webrtc_send',
            self.raw_send_cb,
            qos_profile
        )
        self.create_subscription(
            Go2Cmd,
            'go2/sport_cmd',
            self.sport_cmd_cb,
            qos_profile
        )

    def send(self, payload):
        """Send a payload on the data channel from the asyncio thread."""
        if (
            self.webrtcConn is None
            or self.webrtcConn.data_channel is None
        ):
            return False

        # Schedule the send on the asyncio loop to avoid thread issues.
        def safe_send():
            try:
                self.webrtcConn.data_channel.send(payload)
            except Exception as e:
                self.get_logger().error(
                    f"Failed to send via data_channel: {e}"
                )

        if self.loop is None:
            return False
        self.loop.call_soon_threadsafe(safe_send)
        return True

    def raw_send_cb(self, msg):
        self.send(msg.data)

    def sport_cmd_cb(self, msg):
        name = self._known_api_ids.get(msg.cmd)
        if name is None:
            self.get_logger().warn(
                f"Ignoring unknown sport API ID {msg.cmd}"
            )
            return
        if self.send(gen_command(msg.cmd)):
            self.get_logger().info(
                f"Sent sport command {name} ({msg.cmd})"
            )
        else:
            self.get_logger().warn(
                f"Dropped sport command {name} ({msg.cmd}): "
                f"not connected to the robot"
            )

    async def run(self):
        self.get_logger().info(
            f"Creating Go2Connection to "
            f"{self.internal_board_ip}..."
        )

        try:
            self.webrtcConn = Go2Connection(
                robot_ip=self.internal_board_ip,
                robot_num="1",
                token="",
            )

            await self.webrtcConn.connect()

            self.get_logger().info(
                f"Successfully connected to the robot at "
                f"{self.internal_board_ip}!"
            )

        except Exception as e:
            self.get_logger().error(
                f"Connection to {self.internal_board_ip} failed: {e}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = Go2WebRtcBridge()

    # Spin ROS in a separate thread.
    def ros_spin():
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            if rclpy.ok():
                rclpy.shutdown()

    ros_thread = threading.Thread(
        target=ros_spin,
        daemon=True
    )
    ros_thread.start()

    # Run asyncio in the main thread.
    async def async_main():
        node.loop = asyncio.get_running_loop()
        await node.run()

        # Keep the loop alive to allow aiortc to run background tasks.
        while rclpy.ok():
            await asyncio.sleep(1)

    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()

        ros_thread.join()


if __name__ == '__main__':
    main()
