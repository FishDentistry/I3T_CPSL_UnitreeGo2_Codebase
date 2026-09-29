
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import pyrealsense2 as rs
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image as ImageMsg
from sensor_msgs.msg import CameraInfo
from rclpy.qos import QoSProfile, QoSDurabilityPolicy

class GetCameraFrames(Node):
    def __init__(self):
        super().__init__('Get_camera_frames')

        # ---- CONFIG (use lower bandwidth defaults) ----
        self.color_w, self.color_h, self.color_fps = 640, 480, 15
        self.depth_w, self.depth_h, self.depth_fps = 640, 480, 15
        self.max_poll_ms = 500        # per-try timeout
        self.poll_tries = 6            # total ~3s per timer tick
        self.restart_after_timeouts = 5

        # ---- PIPELINE SETUP ----
        self.ctx = rs.context()
        if len(self.ctx.devices) == 0:
            raise RuntimeError("No RealSense device found")

        self.pipeline = rs.pipeline(self.ctx)
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, self.depth_w, self.depth_h, rs.format.z16, self.depth_fps)
        self.config.enable_stream(rs.stream.color, self.color_w, self.color_h, rs.format.bgr8, self.color_fps)

        self._start_pipeline()

        # ---- ROS PUBS ----
        self.bridge = CvBridge()
        self.color_pub = self.create_publisher(ImageMsg, '/realsense_rgb_image', 10)
        self.depth_pub = self.create_publisher(ImageMsg, '/realsense_depth_image', 10)

        qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.camIntrPub = self.create_publisher(CameraInfo, '/depth_camera_intrinsics', qos)

        # Intrinsics (color stream)
        vprof = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = vprof.get_intrinsics()
        self.cam_info = CameraInfo()
        self.cam_info.width = intr.width
        self.cam_info.height = intr.height
        self.cam_info.k = [intr.fx, 0.0, intr.ppx,
                           0.0, intr.fy, intr.ppy,
                           0.0, 0.0, 1.0]
        self.cam_info.p = [intr.fx, 0.0, intr.ppx, 0.0,
                           0.0, intr.fy, intr.ppy, 0.0,
                           0.0, 0.0, 1.0, 0.0]
        self.cam_info.r = [1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0]
        self.cam_info.d = [0.0]*5
        self.cam_info.distortion_model = 'none'

        # Timers
        self.create_timer(0.1, lambda: self.camIntrPub.publish(self.cam_info))
        self.create_timer(0.1, self.imagePubTimerCallback)

        self.consecutive_timeouts = 0

    def _start_pipeline(self):
        # (Re)start pipeline and related helpers
        self.profile = self.pipeline.start(self.config)

        # Reduce internal frame queues to avoid stale frames
        dev = self.profile.get_device()
        for s in dev.sensors:
            if s.supports(rs.option.frames_queue_size):
                try:
                    s.set_option(rs.option.frames_queue_size, 1)
                except Exception:
                    pass

        self.depth_sensor = dev.first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()
        self.align = rs.align(rs.stream.color)

        # Warmup (non-fatal if some time out)
        for _ in range(10):
            try:
                fs = self.pipeline.wait_for_frames(200)
                if fs: break
            except Exception:
                pass
        self.get_logger().info(f"Depth scale: {self.depth_scale:.6f}. Pipeline started.")

    def _stop_pipeline(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass

    def _restart_pipeline(self, do_hw_reset=False):
        self.get_logger().warn("Restarting RealSense pipeline...")
        self._stop_pipeline()
        if do_hw_reset:
            try:
                self.profile.get_device().hardware_reset()
            except Exception:
                pass
        self._start_pipeline()
        self.consecutive_timeouts = 0

    def _poll_frames(self):
        # Try poll loop to avoid long blocking timeouts
        for _ in range(self.poll_tries):
            try:
                fs = self.pipeline.wait_for_frames(self.max_poll_ms)
            except Exception:
                fs = None
            if fs:
                return fs
        return None

    def imagePubTimerCallback(self):
        frames = self._poll_frames()
        if frames is None:
            self.consecutive_timeouts += 1
            self.get_logger().warn(f"No frames ({self.consecutive_timeouts} consecutive timeouts).")
            if self.consecutive_timeouts >= self.restart_after_timeouts:
                # First soft restart, then hard reset if still bad
                hard = (self.consecutive_timeouts >= 2*self.restart_after_timeouts)
                self._restart_pipeline(do_hw_reset=hard)
            return

        self.consecutive_timeouts = 0
        aligned = self.align.process(frames)
        depth = aligned.get_depth_frame()
        color = aligned.get_color_frame()
        if not depth or not color:
            self.get_logger().warn("Got invalid frames (None). Skipping this tick.")
            return

        depth_np = np.asanyarray(depth.get_data()).astype(np.float32) * self.depth_scale
        color_bgr = np.asanyarray(color.get_data())
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

        depth_np = np.ascontiguousarray(depth_np)
        color_rgb = np.ascontiguousarray(color_rgb)

        try:
            color_msg = self.bridge.cv2_to_imgmsg(color_rgb, encoding='rgb8')
            depth_msg = self.bridge.cv2_to_imgmsg(depth_np, encoding='32FC1')
        except CvBridgeError as e:
            self.get_logger().warn(f"CvBridge error: {e}")
            return

        # The aligned RGB/depth pair represents the same camera acquisition.
        # Keep frame_id empty for compatibility with existing consumers that
        # select the camera TF frame through their own parameters.
        acquisition_stamp = self.get_clock().now().to_msg()
        color_msg.header.stamp = acquisition_stamp
        depth_msg.header.stamp = acquisition_stamp
        self.cam_info.header.stamp = acquisition_stamp

        self.color_pub.publish(color_msg)
        self.depth_pub.publish(depth_msg)

    def destroy_node(self):
        self._stop_pipeline()
        return super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = GetCameraFrames()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

