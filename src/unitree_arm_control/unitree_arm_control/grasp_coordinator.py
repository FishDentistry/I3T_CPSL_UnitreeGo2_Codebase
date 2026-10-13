"""Coordinate guarded semantic-object grasp-and-release motions through MoveIt."""

from collections import deque
import copy
import math
import os
import time

from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseStamped
from intel_realsense_interfaces.msg import GroundedDetectionArray
from intel_realsense_interfaces.msg import SemanticMap
from intel_realsense_interfaces.msg import SemanticObject
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints
from moveit_msgs.msg import JointConstraint
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.msg import OrientationConstraint
from moveit_msgs.msg import PositionConstraint
from moveit_msgs.srv import GetCartesianPath
from moveit_msgs.srv import GetPositionFK
from moveit_msgs.srv import GetPositionIK
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.time import Time
from shape_msgs.msg import SolidPrimitive
from tf2_ros import Buffer
from tf2_ros import TransformListener
from tf2_ros import TransformException
from unitree_arm.msg import GraspCommand
from unitree_arm.msg import GraspStatus
from unitree_arm.msg import JointAngles
from unitree_arm.srv import SetJoint
from unitree_arm.srv import SetArmEnabled
from unitree_arm.srv import StartGraspTeaching
from unitree_arm.srv import CaptureGraspPose
from unitree_arm.srv import SaveGraspTeaching
from unitree_arm.srv import CancelGraspTeaching
from trajectory_msgs.msg import JointTrajectoryPoint

from unitree_arm_control import grasping
from unitree_arm_control import grasp_templates
from unitree_arm_control import protocol
from unitree_arm_control import trajectory


MOVEIT_ERROR_NAMES = {
    MoveItErrorCodes.SUCCESS: 'success',
    MoveItErrorCodes.PLANNING_FAILED: 'planning failed',
    MoveItErrorCodes.INVALID_MOTION_PLAN: 'invalid motion plan',
    MoveItErrorCodes.MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE: (
        'motion plan invalidated by an environment change'
    ),
    MoveItErrorCodes.CONTROL_FAILED: 'controller execution failed',
    MoveItErrorCodes.TIMED_OUT: 'planning or execution timed out',
    MoveItErrorCodes.PREEMPTED: 'request preempted',
    MoveItErrorCodes.START_STATE_IN_COLLISION: 'start state is in collision',
    MoveItErrorCodes.START_STATE_VIOLATES_PATH_CONSTRAINTS: (
        'start state violates path constraints'
    ),
    MoveItErrorCodes.GOAL_IN_COLLISION: 'goal is in collision',
    MoveItErrorCodes.GOAL_VIOLATES_PATH_CONSTRAINTS: (
        'goal violates path constraints'
    ),
    MoveItErrorCodes.GOAL_CONSTRAINTS_VIOLATED: (
        'goal constraints were violated'
    ),
    MoveItErrorCodes.INVALID_GROUP_NAME: 'invalid planning group',
    MoveItErrorCodes.INVALID_GOAL_CONSTRAINTS: 'invalid goal constraints',
    MoveItErrorCodes.INVALID_ROBOT_STATE: 'invalid robot state',
    MoveItErrorCodes.INVALID_LINK_NAME: 'invalid end-effector link',
    MoveItErrorCodes.FRAME_TRANSFORM_FAILURE: 'frame transform failed',
    MoveItErrorCodes.ROBOT_STATE_STALE: 'robot state is stale',
    MoveItErrorCodes.COMMUNICATION_FAILURE: 'communication failure',
    MoveItErrorCodes.NO_IK_SOLUTION: 'no inverse-kinematics solution',
}

# The slowest arm-joint limits in config/joint_limits.yaml. A Cartesian
# retiming fallback uses these for every arm joint, so it cannot exceed the
# configured velocity or acceleration of any one of them.
D1_ARM_JOINT_NAMES = tuple('d1_joint_{}'.format(index) for index in range(6))
D1_ARM_MIN_VELOCITY_LIMIT = 1.05
D1_ARM_MIN_ACCELERATION_LIMIT = 0.5


class D1GraspCoordinator(Node):
    """Run a guarded pre-grasp, grasp, hold, release, and retreat sequence."""

    def __init__(self):
        super().__init__('d1_grasp_coordinator')
        self._declare_parameters()
        self._load_parameters()
        self._validate_parameters()
        self._template_library_error = None
        try:
            self._template_library = grasp_templates.GraspTemplateLibrary(
                self._grasp_templates_path
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            # A damaged optional library must not disable the existing
            # generated-grasp path. Do not overwrite that file on save.
            self._template_library = None
            self._template_library_error = str(error)
            self.get_logger().error(
                'Taught grasps unavailable; generated grasps remain active: '
                + str(error)
            )
        self._teaching = None

        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        status_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._status_publisher = self.create_publisher(
            GraspStatus, self._status_topic, status_qos
        )
        self._pregrasp_publisher = self.create_publisher(
            PoseStamped, self._pregrasp_pose_topic, latched_qos
        )
        self._grasp_publisher = self.create_publisher(
            PoseStamped, self._grasp_pose_topic, latched_qos
        )
        self.create_subscription(
            SemanticMap, self._semantic_map_topic, self._map_callback,
            latched_qos,
        )
        self.create_subscription(
            GroundedDetectionArray,
            self._detections_topic,
            self._detections_callback,
            10,
        )
        self.create_subscription(
            GraspCommand, self._command_topic, self._command_callback, 10
        )
        self.create_subscription(
            JointAngles,
            self._joint_angles_topic,
            self._joint_angles_callback,
            10,
        )
        self.create_service(
            StartGraspTeaching, '/d1_grasp/teach/start',
            self._start_grasp_teaching,
        )
        self.create_service(
            CaptureGraspPose, '/d1_grasp/teach/capture',
            self._capture_grasp_pose,
        )
        self.create_service(
            SaveGraspTeaching, '/d1_grasp/teach/save',
            self._save_grasp_teaching,
        )
        self.create_service(
            CancelGraspTeaching, '/d1_grasp/teach/cancel',
            self._cancel_grasp_teaching,
        )

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._move_group = ActionClient(
            self, MoveGroup, self._move_group_action
        )
        self._execute_trajectory = ActionClient(
            self, ExecuteTrajectory, self._execute_trajectory_action
        )
        self._cartesian_path = self.create_client(
            GetCartesianPath, self._cartesian_path_service
        )
        self._forward_kinematics = self.create_client(
            GetPositionFK, self._forward_kinematics_service
        )
        self._inverse_kinematics = self.create_client(
            GetPositionIK, self._ik_service
        )
        self._set_joint = self.create_client(
            SetJoint, self._gripper_service
        )
        self._set_arm_enabled = self.create_client(
            SetArmEnabled, self._arm_enable_service
        )
        self._arm_trajectory = ActionClient(
            self, FollowJointTrajectory, self._arm_trajectory_action
        )
        self._latest_map = None
        self._latest_detections = None
        self._latest_detection_receipt = 0.0
        self._latest_gripper_angle = None
        self._latest_gripper_receipt = 0.0
        self._latest_arm_angles = None
        self._latest_arm_receipt = 0.0
        self._active = None
        self._taught_search = None
        self._recent_request_ids = deque(maxlen=100)
        self.create_timer(0.1, self._periodic_update)

        mode = 'grasp execution' if self._execution_enabled else 'plan-only'
        self.get_logger().info(
            'D1 grasp coordinator ready on {} in {} mode'.format(
                self._command_topic, mode
            )
        )
        if self._execution_enabled:
            self.get_logger().warning(
                'Physical grasp-and-release execution is enabled. Perception '
                'provides object points, not collision geometry; use only a '
                'manually verified clear volume.'
            )

    def _declare_parameters(self):
        string_parameters = {
            'semantic_map_topic': '/semantic_map',
            'detections_topic': '/grounding_dino/detection_array',
            'command_topic': '/d1_grasp/command',
            'status_topic': '/d1_grasp/status',
            'pregrasp_pose_topic': '/d1_grasp/pregrasp_pose',
            'grasp_pose_topic': '/d1_grasp/grasp_pose',
            'move_group_action': '/move_action',
            'execute_trajectory_action': '/execute_trajectory',
            'cartesian_path_service': '/compute_cartesian_path',
            'forward_kinematics_service': '/compute_fk',
            'gripper_service': '/d1_arm_controller/set_joint',
            'arm_enable_service': '/d1_arm_controller/set_arm_enabled',
            'arm_trajectory_action': (
                '/d1_arm_controller/follow_joint_trajectory'
            ),
            'joint_angles_topic': '/d1_arm_controller/joint_angles',
            'planning_group': 'd1_arm',
            'planning_frame': 'base_link',
            'camera_frame': 'camera_link',
            'reach_reference_frame': 'd1_base_link',
            'tip_link': 'd1_gripper_center',
            'planner_id': '',
            'grasp_templates_path': (
                '~/.local/share/unitree_arm_control/grasp_templates.yaml'
            ),
            # Compact per-request grasp log; empty disables the file.
            'grasp_log_path': '~/.ros/d1_grasp_attempts.log',
            'ik_service': '/compute_ik',
        }
        for name, default in string_parameters.items():
            self.declare_parameter(name, default)

        self.declare_parameter('execution_enabled', False)
        self.declare_parameter('lay_down_raise_enabled', True)
        self.declare_parameter('lay_down_detection_tolerance_degrees', 3.0)
        self.declare_parameter('lay_down_raise_joint_2_degrees', 5.0)
        self.declare_parameter('lay_down_raise_duration_sec', 2.0)
        self.declare_parameter('minimum_confidence', 0.65)
        self.declare_parameter('minimum_observations', 3)
        self.declare_parameter('maximum_object_age_sec', 20.0)
        self.declare_parameter('minimum_reach_m', 0.10)
        self.declare_parameter('maximum_reach_m', 0.67)
        self.declare_parameter('minimum_target_z_m', -0.10)
        self.declare_parameter('maximum_target_z_m', 0.80)

        self.declare_parameter(
            'approach_yaw_offsets_rad',
            [0.0, math.pi / 6.0, -math.pi / 6.0,
             math.pi / 3.0, -math.pi / 3.0],
        )
        self.declare_parameter('approach_distance_m', 0.11)
        self.declare_parameter('grasp_center_offset_m', 0.0)
        self.declare_parameter('default_forward_grasp_depth_offset_m', 0.05)
        self.declare_parameter(
            'class_forward_grasp_depth_offsets_m', ['mug=0.04']
        )
        self.declare_parameter('class_grasp_height_offsets_m', ['mug=0.02'])
        self.declare_parameter('horizontal_forward_depth_classes', ['mug'])
        self.declare_parameter('position_tolerance_m', 0.02)
        self.declare_parameter('grasp_position_tolerance_m', 0.005)
        self.declare_parameter('band_pregrasp_position_tolerance_m', 0.01)
        self.declare_parameter('grasp_orientation_tolerance_rad', 0.35)
        self.declare_parameter('band_alignment_tolerance_rad', 0.10)
        self.declare_parameter('taught_orientation_tolerance_rad', 0.10)
        self.declare_parameter('taught_pregrasp_standoff_m', 0.10)
        # Taught replay: rotate a taught grasp about the object's vertical
        # axis in these steps, up to the class range (default otherwise), and
        # use the first rotation with collision-free IK and a straight path.
        self.declare_parameter('taught_yaw_step_rad', 0.2618)
        self.declare_parameter('taught_default_yaw_range_rad', 0.5236)
        self.declare_parameter(
            'taught_class_yaw_ranges_rad', ['cup=3.1416', 'soda can=3.1416']
        )
        self.declare_parameter('taught_ik_timeout_sec', 0.1)
        # If no rotation works from the taught pre-grasp distance, retry with
        # the approach starting closer, in 2 cm steps, but never closer than
        # this (the open jaws must start clear of the object).
        self.declare_parameter('taught_min_pregrasp_standoff_m', 0.06)
        self.declare_parameter('taught_joint_goal_tolerance_rad', 0.01)
        self.declare_parameter('taught_search_timeout_sec', 30.0)

        self.declare_parameter('planning_time_sec', 5.0)
        self.declare_parameter('planning_attempts', 5)
        self.declare_parameter('velocity_scaling', 0.15)
        self.declare_parameter('acceleration_scaling', 0.10)
        self.declare_parameter('fresh_detection_age_sec', 5.0)
        self.declare_parameter('use_grasp_band', True)
        self.declare_parameter('maximum_grasp_band_width_m', 0.06)
        self.declare_parameter('reacquire_timeout_sec', 10.0)
        self.declare_parameter('reacquire_match_distance_m', 0.25)
        self.declare_parameter('maximum_reacquire_correction_m', 0.08)
        self.declare_parameter('maximum_final_lateral_correction_m', 0.015)
        self.declare_parameter('maximum_alignment_lateral_correction_m', 0.025)
        self.declare_parameter('approach_corridor_radius_m', 0.015)
        self.declare_parameter('post_execution_pose_settle_sec', 0.20)
        self.declare_parameter('post_execution_pose_timeout_sec', 2.0)

        # The final contact motion is a single short Cartesian segment.
        self.declare_parameter('cartesian_step_m', 0.005)
        self.declare_parameter('minimum_cartesian_fraction', 1.0)
        self.declare_parameter('cartesian_jump_threshold', 2.0)
        self.declare_parameter('cartesian_velocity_scaling', 0.05)
        self.declare_parameter('cartesian_acceleration_scaling', 0.05)

        self.declare_parameter('gripper_joint_id', 6)
        self.declare_parameter('gripper_open_degrees', 49.0)
        self.declare_parameter('gripper_closed_degrees', 0.0)
        self.declare_parameter('gripper_tolerance_degrees', 2.0)
        self.declare_parameter('minimum_gripper_closure_degrees', 3.0)
        self.declare_parameter('gripper_stall_delta_degrees', 0.2)
        self.declare_parameter('gripper_stall_confirm_sec', 0.75)
        self.declare_parameter('gripper_operation_timeout_sec', 5.0)
        self.declare_parameter('require_grasp_obstruction', False)
        self.declare_parameter('hold_duration_sec', 3.0)

    def _parameter(self, name):
        return self.get_parameter(name).value

    def _load_parameters(self):
        string_names = (
            'semantic_map_topic', 'detections_topic', 'command_topic',
            'status_topic', 'pregrasp_pose_topic', 'grasp_pose_topic',
            'move_group_action', 'execute_trajectory_action',
            'cartesian_path_service', 'gripper_service',
            'arm_enable_service', 'arm_trajectory_action',
            'forward_kinematics_service',
            'joint_angles_topic', 'planning_group', 'planning_frame',
            'camera_frame',
            'reach_reference_frame', 'tip_link', 'planner_id',
            'grasp_templates_path', 'grasp_log_path', 'ik_service',
        )
        for name in string_names:
            setattr(self, '_' + name, str(self._parameter(name)).strip())

        bool_names = (
            'execution_enabled', 'require_grasp_obstruction',
            'use_grasp_band',
            'lay_down_raise_enabled',
        )
        for name in bool_names:
            setattr(self, '_' + name, bool(self._parameter(name)))

        float_names = (
            'minimum_confidence', 'maximum_object_age_sec',
            'minimum_reach_m', 'maximum_reach_m', 'minimum_target_z_m',
            'maximum_target_z_m', 'approach_distance_m',
            'grasp_center_offset_m',
            'default_forward_grasp_depth_offset_m',
            'position_tolerance_m',
            'grasp_position_tolerance_m',
            'band_pregrasp_position_tolerance_m',
            'lay_down_detection_tolerance_degrees',
            'lay_down_raise_joint_2_degrees',
            'lay_down_raise_duration_sec',
            'grasp_orientation_tolerance_rad',
            'band_alignment_tolerance_rad',
            'taught_orientation_tolerance_rad',
            'taught_pregrasp_standoff_m',
            'taught_yaw_step_rad', 'taught_default_yaw_range_rad',
            'taught_ik_timeout_sec', 'taught_joint_goal_tolerance_rad',
            'taught_min_pregrasp_standoff_m',
            'taught_search_timeout_sec',
            'planning_time_sec', 'velocity_scaling',
            'acceleration_scaling', 'fresh_detection_age_sec',
            'maximum_grasp_band_width_m',
            'reacquire_timeout_sec', 'reacquire_match_distance_m',
            'maximum_reacquire_correction_m',
            'maximum_final_lateral_correction_m',
            'maximum_alignment_lateral_correction_m',
            'approach_corridor_radius_m',
            'post_execution_pose_settle_sec',
            'post_execution_pose_timeout_sec',
            'cartesian_step_m', 'minimum_cartesian_fraction',
            'cartesian_jump_threshold', 'cartesian_velocity_scaling',
            'cartesian_acceleration_scaling', 'gripper_open_degrees',
            'gripper_closed_degrees', 'gripper_tolerance_degrees',
            'minimum_gripper_closure_degrees',
            'gripper_stall_delta_degrees', 'gripper_stall_confirm_sec',
            'gripper_operation_timeout_sec', 'hold_duration_sec',
        )
        for name in float_names:
            setattr(self, '_' + name, float(self._parameter(name)))

        self._minimum_observations = int(
            self._parameter('minimum_observations')
        )
        self._planning_attempts = int(self._parameter('planning_attempts'))
        self._gripper_joint_id = int(self._parameter('gripper_joint_id'))
        self._approach_yaw_offsets = tuple(
            float(value)
            for value in self._parameter('approach_yaw_offsets_rad')
        )
        self._class_forward_grasp_depth_offsets = (
            grasping.parse_forward_grasp_depth_offsets(
                self._parameter('class_forward_grasp_depth_offsets_m')
            )
        )
        self._class_grasp_height_offsets = (
            grasping.parse_grasp_height_offsets(
                self._parameter('class_grasp_height_offsets_m')
            )
        )
        self._taught_class_yaw_ranges = (
            grasp_templates.parse_class_yaw_ranges(
                self._parameter('taught_class_yaw_ranges_rad')
            )
        )
        self._horizontal_forward_depth_classes = {
            grasping.normalize_label(label)
            for label in self._parameter('horizontal_forward_depth_classes')
        }
        # Short aliases keep the motion code readable.
        self._maximum_object_age = self._maximum_object_age_sec
        self._minimum_reach = self._minimum_reach_m
        self._maximum_reach = self._maximum_reach_m
        self._minimum_target_z = self._minimum_target_z_m
        self._maximum_target_z = self._maximum_target_z_m
        self._approach_distance = self._approach_distance_m
        self._grasp_center_offset = self._grasp_center_offset_m
        self._default_forward_grasp_depth_offset = (
            self._default_forward_grasp_depth_offset_m
        )
        self._planning_time = self._planning_time_sec

    def _validate_parameters(self):
        required_strings = (
            'semantic_map_topic', 'detections_topic', 'command_topic',
            'status_topic', 'pregrasp_pose_topic', 'grasp_pose_topic',
            'move_group_action', 'execute_trajectory_action',
            'cartesian_path_service', 'gripper_service',
            'arm_enable_service', 'arm_trajectory_action',
            'forward_kinematics_service',
            'joint_angles_topic', 'planning_group', 'planning_frame',
            'camera_frame',
            'reach_reference_frame', 'tip_link',
            'grasp_templates_path', 'ik_service',
        )
        for name in required_strings:
            if not getattr(self, '_' + name):
                raise RuntimeError('{} must not be empty'.format(name))
        if not 0.0 <= self._minimum_confidence <= 1.0:
            raise RuntimeError('minimum_confidence must be in [0, 1]')
        if self._minimum_observations < 1:
            raise RuntimeError('minimum_observations must be at least one')
        if self._maximum_object_age <= 0.0:
            raise RuntimeError('maximum_object_age_sec must be positive')
        if not 0.0 <= self._minimum_reach < self._maximum_reach:
            raise RuntimeError('reach limits are invalid')
        if self._minimum_target_z >= self._maximum_target_z:
            raise RuntimeError('target z limits are invalid')
        if not self._approach_yaw_offsets:
            raise RuntimeError('approach_yaw_offsets_rad must not be empty')
        if self._approach_distance <= 0.0:
            raise RuntimeError('approach_distance_m must be positive')
        if self._grasp_center_offset < 0.0:
            raise RuntimeError('grasp_center_offset_m must not be negative')
        if (
                not math.isfinite(self._default_forward_grasp_depth_offset)
                or self._default_forward_grasp_depth_offset < 0.0):
            raise RuntimeError(
                'default_forward_grasp_depth_offset_m must be finite and '
                'nonnegative'
            )
        if self._position_tolerance_m <= 0.0:
            raise RuntimeError('position_tolerance_m must be positive')
        if self._grasp_position_tolerance_m <= 0.0:
            raise RuntimeError(
                'grasp_position_tolerance_m must be positive'
            )
        if self._band_pregrasp_position_tolerance_m <= 0.0:
            raise RuntimeError(
                'band_pregrasp_position_tolerance_m must be positive'
            )
        for name in (
                'lay_down_detection_tolerance_degrees',
                'lay_down_raise_joint_2_degrees',
                'lay_down_raise_duration_sec'):
            if getattr(self, '_' + name) <= 0.0:
                raise RuntimeError('{} must be positive'.format(name))
        if not 0.0 < self._grasp_orientation_tolerance_rad <= math.pi:
            raise RuntimeError(
                'grasp_orientation_tolerance_rad must be in (0, pi]'
            )
        if not 0.0 < self._band_alignment_tolerance_rad <= math.pi:
            raise RuntimeError(
                'band_alignment_tolerance_rad must be in (0, pi]'
            )
        if not 0.0 < self._taught_orientation_tolerance_rad <= math.pi / 2.0:
            raise RuntimeError(
                'taught_orientation_tolerance_rad must be in (0, pi/2]'
            )
        if not (grasp_templates.MINIMUM_STANDOFF_M
                <= self._taught_pregrasp_standoff_m
                <= grasp_templates.MAXIMUM_STANDOFF_M):
            raise RuntimeError('taught_pregrasp_standoff_m must be in [0.02, 0.25]')
        if not 0.0 < self._taught_yaw_step_rad <= math.pi:
            raise RuntimeError('taught_yaw_step_rad must be in (0, pi]')
        if not 0.0 <= self._taught_default_yaw_range_rad <= math.pi + 1.0e-3:
            raise RuntimeError('taught_default_yaw_range_rad must be in [0, pi]')
        if not (grasp_templates.MINIMUM_STANDOFF_M
                <= self._taught_min_pregrasp_standoff_m
                <= grasp_templates.MAXIMUM_STANDOFF_M):
            raise RuntimeError(
                'taught_min_pregrasp_standoff_m must be in [0.02, 0.25]'
            )
        if self._taught_ik_timeout_sec <= 0.0:
            raise RuntimeError('taught_ik_timeout_sec must be positive')
        if self._taught_joint_goal_tolerance_rad <= 0.0:
            raise RuntimeError('taught_joint_goal_tolerance_rad must be positive')
        if self._taught_search_timeout_sec <= 0.0:
            raise RuntimeError('taught_search_timeout_sec must be positive')
        if self._planning_time <= 0.0 or self._planning_attempts < 1:
            raise RuntimeError('planning limits are invalid')
        for name in ('velocity_scaling', 'acceleration_scaling'):
            value = getattr(self, '_' + name)
            if not 0.0 < value <= 1.0:
                raise RuntimeError('{} must be in (0, 1]'.format(name))
        if self._fresh_detection_age_sec <= 0.0:
            raise RuntimeError('fresh_detection_age_sec must be positive')
        if (not math.isfinite(self._maximum_grasp_band_width_m)
                or self._maximum_grasp_band_width_m <= 0.0):
            raise RuntimeError(
                'maximum_grasp_band_width_m must be finite and positive'
            )
        if self._reacquire_timeout_sec <= 0.0:
            raise RuntimeError('reacquire_timeout_sec must be positive')
        if self._reacquire_match_distance_m <= 0.0:
            raise RuntimeError('reacquire_match_distance_m must be positive')
        if self._maximum_reacquire_correction_m <= 0.0:
            raise RuntimeError(
                'maximum_reacquire_correction_m must be positive'
            )
        if self._maximum_final_lateral_correction_m <= 0.0:
            raise RuntimeError(
                'maximum_final_lateral_correction_m must be positive'
            )
        if self._maximum_alignment_lateral_correction_m <= 0.0:
            raise RuntimeError(
                'maximum_alignment_lateral_correction_m must be positive'
            )
        if self._approach_corridor_radius_m <= 0.0:
            raise RuntimeError('approach_corridor_radius_m must be positive')
        if self._post_execution_pose_settle_sec < 0.0:
            raise RuntimeError(
                'post_execution_pose_settle_sec must not be negative'
            )
        if (self._post_execution_pose_timeout_sec
                <= self._post_execution_pose_settle_sec):
            raise RuntimeError(
                'post_execution_pose_timeout_sec must exceed the settle time'
            )
        if self._cartesian_step_m <= 0.0:
            raise RuntimeError('cartesian_step_m must be positive')
        if not 0.0 < self._minimum_cartesian_fraction <= 1.0:
            raise RuntimeError(
                'minimum_cartesian_fraction must be in (0, 1]'
            )
        if self._cartesian_jump_threshold < 0.0:
            raise RuntimeError(
                'cartesian_jump_threshold must not be negative'
            )
        for name in (
                'cartesian_velocity_scaling',
                'cartesian_acceleration_scaling'):
            value = getattr(self, '_' + name)
            if not 0.0 < value <= 1.0:
                raise RuntimeError('{} must be in (0, 1]'.format(name))
        if not 0 <= self._gripper_joint_id <= 6:
            raise RuntimeError('gripper_joint_id must be in [0, 6]')
        if self._gripper_open_degrees == self._gripper_closed_degrees:
            raise RuntimeError('gripper open and closed values must differ')
        for name in (
                'gripper_tolerance_degrees',
                'minimum_gripper_closure_degrees',
                'gripper_stall_delta_degrees',
                'gripper_stall_confirm_sec',
                'gripper_operation_timeout_sec'):
            if getattr(self, '_' + name) <= 0.0:
                raise RuntimeError('{} must be positive'.format(name))
        if self._hold_duration_sec < 0.0:
            raise RuntimeError('hold_duration_sec must not be negative')

    def _map_callback(self, message):
        self._latest_map = message

    def _detections_callback(self, message):
        self._latest_detections = message
        self._latest_detection_receipt = self._now_seconds()
        if self._active is not None and (
                self._active['phase'] == 'reacquiring'):
            self._try_reacquire(message, self._latest_detection_receipt)

    def _joint_angles_callback(self, message):
        if len(message.angle_degrees) < protocol.JOINT_COUNT:
            return
        values = tuple(float(value) for value in message.angle_degrees)
        if not all(math.isfinite(value) for value in values):
            return
        self._latest_gripper_angle = values[self._gripper_joint_id]
        self._latest_gripper_receipt = self._now_seconds()
        self._latest_arm_angles = values[:protocol.ARM_JOINT_COUNT]
        self._latest_arm_receipt = self._latest_gripper_receipt

    def _near_lay_down(self):
        return (
            self._latest_arm_angles is not None
            and self._now_seconds() - self._latest_arm_receipt <= 1.0
            and all(
                abs(actual - reference)
                <= self._lay_down_detection_tolerance_degrees
                for actual, reference in zip(
                    self._latest_arm_angles,
                    protocol.LAY_DOWN_ANGLES_DEGREES[:protocol.ARM_JOINT_COUNT],
                )
            )
        )

    def _now_seconds(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _empty_pose(self):
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = self._planning_frame
        pose.pose.orientation.w = 1.0
        return pose

    def _target_pose(self, point, quaternion):
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = self._planning_frame
        pose.pose.position.x = point[0]
        pose.pose.position.y = point[1]
        pose.pose.position.z = point[2]
        pose.pose.orientation.x = quaternion[0]
        pose.pose.orientation.y = quaternion[1]
        pose.pose.orientation.z = quaternion[2]
        pose.pose.orientation.w = quaternion[3]
        return pose

    @staticmethod
    def _point_tuple(point):
        return (float(point.x), float(point.y), float(point.z))

    def _publish_status(
            self, command, stage, terminal, success, message,
            object_id='', target_pose=None):
        status = GraspStatus()
        status.header.stamp = self.get_clock().now().to_msg()
        status.header.frame_id = self._planning_frame
        status.request_id = command.request_id.strip()
        status.object_class = command.object_class.strip()
        status.object_id = object_id or command.object_id.strip()
        status.stage = stage
        status.terminal = terminal
        status.success = success
        status.message = message
        status.target_pose = target_pose or self._empty_pose()
        self._status_publisher.publish(status)

    def _publish_active_stage(self, stage, message, pose=None):
        if self._active is None:
            return
        target_pose = pose or self._active.get('target_pose')
        self._active['target_pose'] = target_pose
        self._active['last_stage'] = stage
        self._publish_status(
            self._active['command'], stage, False, False, message,
            self._active['object_id'], target_pose,
        )

    def _reject(self, command, message):
        self._grasp_log(
            'REQUEST REJECTED before any grasp was tried: ' + message,
            'warn', request_id=command.request_id.strip(),
        )
        self._publish_status(
            command, GraspStatus.STAGE_REJECTED, True, False, message
        )

    def _start_grasp_teaching(self, request, response):
        if self._active is not None:
            response.message = 'a grasp request is active'
            return response
        if self._template_library is None:
            response.message = (
                'grasp template library is unavailable: '
                + self._template_library_error
            )
            return response
        object_class = grasping.normalize_label(request.object_class)
        if not object_class or self._latest_map is None:
            response.message = 'object class or semantic map is unavailable'
            return response
        objects = grasping.eligible_objects(
            self._latest_map.objects, object_class, request.object_id,
            self._now_seconds(), self._minimum_confidence,
            self._minimum_observations, self._maximum_object_age,
            SemanticObject.STATUS_ACTIVE,
        )
        if not objects:
            response.message = 'no fresh ACTIVE object matches the class and ID'
            return response
        try:
            frame = self._latest_map.header.frame_id.strip()
            if not frame:
                raise ValueError('semantic map frame is empty')
            map_transform = self._lookup_transform(self._planning_frame, frame)
            arm_transform = self._lookup_transform(
                self._planning_frame, self._reach_reference_frame
            )
            arm_origin = (
                (0.0, 0.0, 0.0) if arm_transform is None
                else self._point_tuple(arm_transform.transform.translation)
            )
            choices = []
            for item in objects:
                object_point = self._transform_point(
                    self._point_tuple(item.position), map_transform
                )
                if self._target_problem(
                        object_point, 'teaching object center') is not None:
                    continue
                reference = grasp_templates.object_reference(
                    object_point, arm_origin
                )
                choices.append((
                    grasping.distance_between(object_point, arm_origin),
                    item, reference,
                ))
            if not choices:
                raise ValueError(
                    'no fresh reachable object center matches the request'
                )
            _, selected, reference = min(
                choices, key=lambda item: item[0]
            )
        except Exception as error:
            response.message = 'could not start teaching: {}'.format(error)
            return response
        self._teaching = {
            'object_class': object_class,
            'object_id': selected.object_id,
            'reference': reference,
            'poses': {},
            'demonstrated_pregrasp': None,
            'open_angle': None,
            'contact_joints': None,
            'replay_check': None,
        }
        self._taught_search = None
        response.success = True
        response.message = (
            'object-center reference frozen; capture contact with the jaws '
            'open (pregrasp is optional), then save after verifying the grasp'
        )
        response.selected_object_id = selected.object_id
        response.reference_pose = self._target_pose(*reference)
        return response

    def _capture_grasp_pose(self, request, response):
        if self._active is not None or self._teaching is None:
            response.message = 'teaching is not active or a grasp is running'
            return response
        stage = request.stage.strip().lower()
        if stage not in ('pregrasp', 'contact'):
            response.message = 'stage must be pregrasp or contact'
            return response
        if (self._latest_gripper_angle is None
                or self._now_seconds() - self._latest_gripper_receipt > 2.0):
            response.message = 'fresh gripper feedback is unavailable'
            return response
        try:
            transform = self._tf_buffer.lookup_transform(
                self._planning_frame, self._tip_link, Time()
            )
            point = self._point_tuple(transform.transform.translation)
            orientation = transform.transform.rotation
            quaternion = (
                orientation.x, orientation.y, orientation.z, orientation.w
            )
            recorded = grasp_templates.relative_pose(
                self._teaching['reference'], point, quaternion
            )
            detail = ''
            if stage == 'contact':
                # Only the contact pose must be accurate. The pre-grasp is
                # rebuilt on the contact gripper axis, so the approach is
                # straight by construction; a captured pregrasp contributes
                # only its standoff distance.
                demonstrated = self._teaching.get('demonstrated_pregrasp')
                pregrasp, standoff = grasp_templates.straight_pregrasp(
                    recorded, self._taught_pregrasp_standoff_m, demonstrated
                )
                open_angle = (
                    self._teaching['open_angle']
                    if demonstrated is not None
                    else self._latest_gripper_angle
                )
                grasp_templates.validate_template({
                    'name': 'capture',
                    'reference': 'target_object_center',
                    'pregrasp': pregrasp,
                    'contact': recorded,
                    'gripper_open_degrees': open_angle,
                    'enabled': True,
                })
                detail = (
                    '; straight approach derived with pre-grasp {:.1f} cm '
                    'behind contact along the gripper axis'.format(
                        100.0 * standoff
                    )
                )
        except Exception as error:
            response.message = 'could not capture {}: {}'.format(stage, error)
            return response
        self._taught_search = None
        if stage == 'pregrasp':
            self._teaching['demonstrated_pregrasp'] = recorded
            self._teaching['poses'].pop('pregrasp', None)
            self._teaching['poses'].pop('contact', None)
            self._teaching['open_angle'] = self._latest_gripper_angle
            self._teaching['contact_joints'] = None
            self._teaching['replay_check'] = None
        else:
            self._teaching['poses']['pregrasp'] = pregrasp
            self._teaching['poses']['contact'] = recorded
            self._teaching['open_angle'] = open_angle
            self._teaching['contact_joints'] = self._arm_joint_radians()
            self._start_teach_replay_check()
            detail += ('; now checking that this grasp can be replayed '
                       '(collision-free IK and a straight approach), run '
                       'save after a moment')
        response.success = True
        response.message = '{} recorded from achieved gripper TF{}'.format(
            stage, detail + self._joint_limit_warning()
        )
        response.achieved_pose = self._target_pose(point, quaternion)
        return response

    def _start_teach_replay_check(self):
        """Run the replay check on the grasp just captured."""
        teaching = self._teaching
        poses = teaching['poses']
        template = {
            'name': 'capture',
            'reference': 'target_object_center',
            'pregrasp': poses['pregrasp'],
            'contact': poses['contact'],
            'gripper_open_degrees': teaching['open_angle'],
            'enabled': True,
        }
        teaching['replay_check'] = {'state': 'pending'}
        reference = teaching['reference']

        def done(result, summary, teaching=teaching):
            if self._teaching is not teaching:
                return
            if result is None:
                teaching['replay_check'] = {
                    'state': 'failed', 'detail': summary,
                }
                self._grasp_log(
                    'TEACH check FAILED for the captured {} grasp: {}'.format(
                        teaching['object_class'], summary
                    ), 'warn', request_id='teach',
                )
            else:
                teaching['replay_check'] = {'state': 'ok', 'detail': summary}
                self._grasp_log(
                    'TEACH check passed for the captured {} grasp; it can '
                    'be saved'.format(teaching['object_class']),
                    request_id='teach',
                )

        self._start_taught_search(
            template, lambda yaw: reference, (0.0,),
            teaching.get('contact_joints'), done,
        )

    def _joint_limit_warning(self, margin_degrees=5.0):
        """Flag taught poses that rest on a joint limit.

        A pose demonstrated with released joints can sit at, or slightly past,
        a published limit. Planning then cannot reach it on a straight line,
        which shows up later as a low Cartesian approach fraction.
        """
        angles = self._latest_arm_angles
        if angles is None:
            return ''
        close = []
        for index, (angle, limits) in enumerate(
                zip(angles, protocol.JOINT_LIMITS_DEGREES)):
            if limits is None:
                continue
            lower, upper = limits
            if angle <= lower + margin_degrees or angle >= upper - margin_degrees:
                close.append('J{} at {:.1f} deg (limit {:.0f} to {:.0f})'.format(
                    index, angle, lower, upper
                ))
        if not close:
            return ''
        return (
            '. WARNING: {} within {:.0f} deg of its limit; replay may not '
            'reach this pose, so consider reteaching it with that joint '
            'farther from the limit'.format(', '.join(close), margin_degrees)
        )

    def _save_grasp_teaching(self, request, response):
        if self._active is not None or self._teaching is None:
            response.message = 'teaching is not active or a grasp is running'
            return response
        if not request.grasp_succeeded:
            response.message = 'grasp_succeeded must be true to save a template'
            return response
        poses = self._teaching['poses']
        if 'pregrasp' not in poses or 'contact' not in poses:
            response.message = 'capture contact before saving'
            return response
        check = self._teaching.get('replay_check') or {}
        if check.get('state') == 'pending':
            response.message = (
                'still checking that the captured grasp can be replayed; '
                'run save again in a moment'
            )
            return response
        if check.get('state') != 'ok':
            response.message = (
                'not saved: the captured grasp could not be replayed from '
                'here ({}). Adjust the arm and capture contact again'.format(
                    check.get('detail', 'check did not run')
                )
            )
            return response
        template = {
            'name': request.name.strip(),
            'reference': 'target_object_center',
            'pregrasp': poses['pregrasp'],
            'contact': poses['contact'],
            'gripper_open_degrees': self._teaching['open_angle'],
            'enabled': True,
        }
        if self._teaching.get('contact_joints') is not None:
            template['contact_joints'] = [
                float(value) for value in self._teaching['contact_joints']
            ]
        try:
            self._template_library.add(
                self._teaching['object_class'], template
            )
        except (ValueError, OSError) as error:
            response.message = 'could not save grasp: {}'.format(error)
            return response
        response.success = True
        response.message = 'saved {} under {} in {}'.format(
            template['name'], self._teaching['object_class'],
            self._template_library.path,
        )
        self._teaching = None
        return response

    def _cancel_grasp_teaching(self, request, response):
        if self._teaching is None:
            response.message = 'no teaching session is active'
            return response
        self._teaching = None
        self._taught_search = None
        response.success = True
        response.message = 'teaching session discarded without saving'
        return response

    def _command_callback(self, command):
        request_id = command.request_id.strip()
        object_class = grasping.normalize_label(command.object_class)
        if self._teaching is not None:
            self._reject(command, 'finish or restart the active teaching session first')
            return
        if not request_id:
            self._reject(command, 'request_id must not be empty')
            return
        if not object_class:
            self._reject(command, 'object_class must not be empty')
            return
        if request_id in self._recent_request_ids:
            self._reject(command, 'request_id has already been processed')
            return
        if self._active is not None:
            self._reject(command, 'another grasp request is active')
            return
        if self._latest_map is None:
            self._reject(command, 'no semantic map has been received')
            return
        if not self._move_group.wait_for_server(timeout_sec=0.0):
            self._reject(command, 'MoveIt move_group action is unavailable')
            return
        missing_planning = []
        if not self._cartesian_path.wait_for_service(timeout_sec=0.0):
            missing_planning.append(self._cartesian_path_service)
        if not self._forward_kinematics.wait_for_service(timeout_sec=0.0):
            missing_planning.append(self._forward_kinematics_service)
        if missing_planning:
            self._reject(
                command,
                'required planning interfaces are unavailable: {}'.format(
                    ', '.join(missing_planning)
                ),
            )
            return
        raise_from_laydown = False
        if self._execution_enabled:
            missing = []
            if not self._execute_trajectory.wait_for_server(timeout_sec=0.0):
                missing.append(self._execute_trajectory_action)
            if not self._set_joint.wait_for_service(timeout_sec=0.0):
                missing.append(self._gripper_service)
            if missing:
                self._reject(
                    command,
                    'required execution interfaces are unavailable: {}'.format(
                        ', '.join(missing)
                    ),
                )
                return
            if self._latest_arm_angles is None or (
                    self._now_seconds() - self._latest_arm_receipt > 1.0):
                self._reject(command, 'fresh D1 joint feedback is unavailable')
                return
            raise_from_laydown = (
                self._lay_down_raise_enabled and self._near_lay_down()
            )
            if raise_from_laydown:
                if not self._set_arm_enabled.wait_for_service(
                        timeout_sec=0.0):
                    self._reject(command, 'arm-enable service is unavailable')
                    return
                if not self._arm_trajectory.wait_for_server(
                        timeout_sec=0.0):
                    self._reject(command, 'D1 arm trajectory action is unavailable')
                    return

        objects = grasping.eligible_objects(
            self._latest_map.objects, object_class, command.object_id,
            self._now_seconds(), self._minimum_confidence,
            self._minimum_observations, self._maximum_object_age,
            SemanticObject.STATUS_ACTIVE,
        )
        if not objects:
            self._reject(command, self._selection_failure_message(command))
            return

        try:
            (
                selected, object_point, target_point, approach_origin,
                candidates, band,
            ) = self._select_object_and_candidates(objects, object_class)
        except Exception as error:
            self._reject(command, str(error))
            return

        self._recent_request_ids.append(request_id)
        first_pose = self._candidate_pose(candidates[0], 'pregrasp_point')
        self._active = {
            'token': object(),
            'command': command,
            'object_id': selected.object_id,
            'object_class': object_class,
            'forward_grasp_depth_offset': (
                grasping.forward_grasp_depth_for_class(
                    object_class,
                    self._default_forward_grasp_depth_offset,
                    self._class_forward_grasp_depth_offsets,
                )
            ),
            'grasp_height_offset': grasping.grasp_height_for_class(
                object_class, self._class_grasp_height_offsets
            ),
            'horizontal_depth': (
                object_class in self._horizontal_forward_depth_classes
            ),
            'object_point': object_point,
            'target_point': target_point,
            'grasp_band_width': None if band is None else band[1],
            'approach_origin': approach_origin,
            'candidates': candidates,
            'candidate_index': 0,
            'candidate_failures': [],
            'candidate': None,
            'preflight_trajectory': None,
            'preflight_motion_purpose': None,
            'predicted_pregrasp_state': None,
            'pregrasp_pose': first_pose,
            'grasp_pose': None,
            'cartesian_start_state': None,
            'retreat_pose': None,
            'target_pose': first_pose,
            'last_stage': None,
            'phase': 'accepted',
            'deadline': 0.0,
            'required_detection_receipt': 0.0,
            'gripper_start_angle': None,
            'gripper_last_angle': None,
            'gripper_last_change': 0.0,
            'open_after_laydown': False,
            'gripper_open_retry_used': False,
            'gripper_open_best_distance': None,
            'gripper_open_last_progress': 0.0,
            'gripper_open_command_receipt': None,
            'raise_from_laydown': raise_from_laydown,
        }
        taught_names = [
            candidate['template_name'] for candidate in candidates
            if candidate.get('taught')
        ]
        heuristic_count = len(candidates) - len(taught_names)
        self._grasp_log(
            'NEW REQUEST for {} ({}): {} TAUGHT{} + {} HEURISTIC '
            'candidate(s); {}'.format(
                object_class, selected.object_id, len(taught_names),
                ' [' + ', '.join(taught_names) + ']' if taught_names else '',
                heuristic_count,
                'TAUGHT grasps are tried first' if taught_names else
                'no usable taught grasp for this class, HEURISTIC only',
            )
        )
        self._publish_active_stage(
            GraspStatus.STAGE_ACCEPTED,
            'selected {} with {} safeguard-compatible approach candidate(s); '
            'forward grasp depth is {:.3f} m; {}'.format(
                selected.object_id,
                len(candidates),
                grasping.forward_grasp_depth_for_class(
                    object_class,
                    self._default_forward_grasp_depth_offset,
                    self._class_forward_grasp_depth_offsets,
                ),
                ('narrow band width {:.3f} m'.format(band[1])
                 if band is not None else (
                     'using taught object-center target'
                     if candidates[0].get('taught') else
                     'using generated center target'
                 )),
            ),
            first_pose,
        )
        if self._execution_enabled:
            if raise_from_laydown:
                self._enable_laydown_arm()
            else:
                self._command_gripper(
                    self._gripper_open_degrees,
                    'wait_open_before_pregrasp',
                    GraspStatus.STAGE_OPENING,
                    'opening gripper before validating the arm motion',
                )
        else:
            self._start_next_pregrasp_candidate()

    def _enable_laydown_arm(self):
        self._active['phase'] = 'enabling_laydown_arm'
        self._publish_active_stage(
            GraspStatus.STAGE_EXECUTING,
            'near laid-down pose; enabling the arm for a small feedback-checked '
            'raise before MoveIt planning',
        )
        request = SetArmEnabled.Request()
        request.enabled = True
        token = self._active['token']
        self._set_arm_enabled.call_async(request).add_done_callback(
            lambda done, token=token: self._laydown_enable_result(done, token)
        )

    def _laydown_enable_result(self, future, token):
        if (not self._active_token_valid(token)
                or self._active['phase'] != 'enabling_laydown_arm'):
            return
        try:
            response = future.result()
        except Exception as error:
            self._finish_failure('could not enable laid-down arm: {}'.format(error))
            return
        if not response.published:
            self._finish_failure(
                'arm-enable command was rejected: {}'.format(response.message)
            )
            return
        # The service confirms publication, not motor readiness. Give the
        # controller a brief interval before issuing the feedback-checked move.
        self._active['phase'] = 'waiting_laydown_enable'
        self._active['deadline'] = self._now_seconds() + 0.5

    def _start_laydown_raise(self):
        if self._latest_arm_angles is None or (
                self._now_seconds() - self._latest_arm_receipt > 1.0):
            self._finish_failure('joint feedback became stale before the raise')
            return
        if not self._near_lay_down():
            self._finish_failure(
                'arm left the near lay-down pose before the raise; no '
                'pre-planning motion was sent'
            )
            return
        angles = list(self._latest_arm_angles)
        # From the measured lay-down pose, decreasing J2 raises the gripper;
        # increasing J1 lowers it. Bring a marginally out-of-bounds J1 toward
        # its published limit at the same time.
        angles[1] = max(angles[1], protocol.JOINT_LIMITS_DEGREES[1][0])
        angles[2] = max(
            protocol.JOINT_LIMITS_DEGREES[2][0],
            angles[2] - self._lay_down_raise_joint_2_degrees,
        )
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = [
            'd1_joint_{}'.format(index)
            for index in range(protocol.ARM_JOINT_COUNT)
        ]
        point = JointTrajectoryPoint()
        point.positions = [math.radians(value) for value in angles]
        duration = self._lay_down_raise_duration_sec
        point.time_from_start = Duration(
            sec=int(duration), nanosec=int((duration % 1.0) * 1e9)
        )
        goal.trajectory.points = [point]
        goal.goal_time_tolerance = Duration(sec=8)
        self._active['phase'] = 'raising_laydown_arm'
        self._publish_active_stage(
            GraspStatus.STAGE_EXECUTING,
            'raising J2 by {:.1f} degrees before MoveIt planning'.format(
                self._lay_down_raise_joint_2_degrees
            ),
        )
        token = self._active['token']
        self._arm_trajectory.send_goal_async(goal).add_done_callback(
            lambda done, token=token: self._laydown_raise_goal(done, token)
        )

    def _laydown_raise_goal(self, future, token):
        if (not self._active_token_valid(token)
                or self._active['phase'] != 'raising_laydown_arm'):
            return
        try:
            handle = future.result()
        except Exception as error:
            self._finish_failure('could not send lay-down raise: {}'.format(error))
            return
        if not handle.accepted:
            self._finish_failure('D1 controller rejected the lay-down raise')
            return
        handle.get_result_async().add_done_callback(
            lambda done, token=token: self._laydown_raise_result(done, token)
        )

    def _laydown_raise_result(self, future, token):
        if (not self._active_token_valid(token)
                or self._active['phase'] != 'raising_laydown_arm'):
            return
        try:
            result = future.result().result
        except Exception as error:
            self._finish_failure('lay-down raise failed: {}'.format(error))
            return
        if result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            self._finish_failure(
                'lay-down raise failed: {}'.format(result.error_string)
            )
            return
        self._active['raise_from_laydown'] = False
        self._active['open_after_laydown'] = True
        self._command_gripper(
            self._gripper_open_degrees,
            'wait_open_before_pregrasp',
            GraspStatus.STAGE_OPENING,
            'lay-down raise reached; opening gripper before MoveIt planning',
        )

    def _selection_failure_message(self, command):
        objects = list(self._latest_map.objects)
        requested_class = grasping.normalize_label(command.object_class)
        requested_id = command.object_id.strip()
        relevant = [
            item for item in objects
            if grasping.normalize_label(item.label) == requested_class
        ]
        if requested_id:
            by_id = [item for item in objects if item.object_id == requested_id]
            if not by_id:
                available = ', '.join(item.object_id for item in relevant[:5])
                return 'object_id {} is absent; class IDs: {}'.format(
                    requested_id, available or 'none'
                )
            relevant = by_id
        if not relevant:
            labels = sorted({
                grasping.normalize_label(item.label) for item in objects
                if grasping.normalize_label(item.label)
            })
            return 'no object has class {}; available classes: {}'.format(
                requested_class, ', '.join(labels[:10]) or 'none'
            )
        details = []
        for item in relevant[:5]:
            failures = grasping.object_safeguard_failures(
                item, self._now_seconds(), self._minimum_confidence,
                self._minimum_observations, self._maximum_object_age,
                SemanticObject.STATUS_ACTIVE,
            )
            details.append('{}: {}'.format(
                item.object_id, '; '.join(failures) or 'class/ID mismatch'
            ))
        return 'no eligible {} object; {}'.format(
            requested_class, ' | '.join(details)
        )

    def _detection_point(
            self, message, detection, band=False, band_candidate=None):
        """Express a detection center or band center in the planning frame."""
        if band_candidate is not None:
            if message.map_transform_available and (
                    band_candidate.has_map_position):
                source = band_candidate.map_position
                source_frame = message.map_frame.strip()
            else:
                source = band_candidate.camera_position
                source_frame = message.camera_frame.strip()
        elif band:
            if not detection.has_grasp_band:
                return None
            if message.map_transform_available and (
                    detection.has_grasp_band_map_position):
                source = detection.grasp_band_map_position
                source_frame = message.map_frame.strip()
            else:
                source = detection.grasp_band_camera_position
                source_frame = message.camera_frame.strip()
        elif message.map_transform_available and detection.has_map_position:
            source = detection.map_position
            source_frame = message.map_frame.strip()
        elif detection.has_camera_position:
            source = detection.camera_position
            source_frame = message.camera_frame.strip()
        else:
            return None
        if not source_frame:
            source_frame = message.header.frame_id.strip()
        if not source_frame:
            return None
        transformed = self._transform_point(
            self._point_tuple(source),
            self._lookup_transform(self._planning_frame, source_frame),
        )
        if not all(math.isfinite(value) for value in transformed):
            return None
        return transformed

    def _grasp_band_points(self, message, detection):
        """Return ranked bands that fit the jaw-width safeguard."""
        if not self._use_grasp_band:
            return ()
        candidates = list(detection.grasp_band_candidates)
        if candidates:
            entries = [
                (candidate, float(candidate.width_m))
                for candidate in candidates
            ]
        elif detection.has_grasp_band:
            entries = [(None, float(detection.grasp_band_width_m))]
        else:
            return ()
        usable = []
        excessive = []
        within_limit = False
        for candidate, width in entries:
            if not math.isfinite(width) or width <= 0.0:
                continue
            if width > self._maximum_grasp_band_width_m:
                excessive.append(width)
                continue
            within_limit = True
            point = self._detection_point(
                message, detection, band=candidate is None,
                band_candidate=candidate,
            )
            if point is not None:
                usable.append((point, width))
        if not usable and excessive and not within_limit:
            raise ValueError(
                'all visible bands exceed the {:.3f} m gripper-width '
                'safeguard (narrowest {:.3f} m)'.format(
                    self._maximum_grasp_band_width_m, min(excessive)
                )
            )
        return tuple(usable)

    def _initial_grasp_bands(self, object_point, object_class):
        message = self._latest_detections
        if (not self._use_grasp_band or message is None
                or self._now_seconds() - self._latest_detection_receipt
                > self._fresh_detection_age_sec):
            return ()
        matches = []
        for detection in message.detections:
            label = detection.requested_target.strip() or detection.label.strip()
            if grasping.normalize_label(label) != object_class:
                continue
            if float(detection.score) < self._minimum_confidence:
                continue
            try:
                center = self._detection_point(message, detection)
            except Exception:
                continue
            if center is None:
                continue
            separation = grasping.distance_between(center, object_point)
            if separation <= self._maximum_reacquire_correction_m:
                matches.append((separation, detection))
        if not matches:
            return ()
        _, detection = min(matches, key=lambda item: item[0])
        return self._grasp_band_points(message, detection)

    def _select_object_and_candidates(self, objects, object_class):
        source_frame = self._latest_map.header.frame_id.strip()
        if not source_frame:
            raise RuntimeError('semantic map frame_id is empty')
        source_to_planning = self._lookup_transform(
            self._planning_frame, source_frame
        )
        arm_transform = self._tf_buffer.lookup_transform(
            self._planning_frame, self._reach_reference_frame, Time()
        )
        arm_origin = self._point_tuple(arm_transform.transform.translation)
        # A grasp's approach is a property of the object and arm mounting,
        # not of the current arm posture. Using the live tip here changes the
        # pre-grasp and wrist orientation between laid-down and zero starts.
        approach_origin = arm_origin
        camera_transform = self._lookup_transform(
            self._planning_frame, self._camera_frame
        )
        camera_origin = (
            (0.0, 0.0, 0.0)
            if camera_transform is None
            else self._point_tuple(camera_transform.transform.translation)
        )
        forward_grasp_depth = grasping.forward_grasp_depth_for_class(
            object_class,
            self._default_forward_grasp_depth_offset,
            self._class_forward_grasp_depth_offsets,
        )
        grasp_height_offset = grasping.grasp_height_for_class(
            object_class, self._class_grasp_height_offsets
        )
        horizontal_depth = (
            object_class in self._horizontal_forward_depth_classes
        )
        viable = []
        failures = []
        for item in objects:
            object_point = self._transform_point(
                self._point_tuple(item.position), source_to_planning
            )
            if not all(math.isfinite(value) for value in object_point):
                continue
            taught_candidates = []
            try:
                reference = grasp_templates.object_reference(
                    object_point, approach_origin
                )
            except ValueError as error:
                failures.append(str(error))
            else:
                templates = (
                    () if self._template_library is None
                    else self._template_library.templates_for(object_class)
                )
                for template in templates:
                    # Keep a taught grasp if any rotation about the object
                    # passes the reach safeguards; IK decides later.
                    first_problem = None
                    for yaw in self._taught_yaws(object_class):
                        try:
                            candidate = grasp_templates.candidate_from_template(
                                template,
                                grasp_templates.object_reference(
                                    object_point, approach_origin, yaw
                                ),
                                yaw,
                            )
                        except ValueError as error:
                            first_problem = first_problem or str(error)
                            continue
                        problem = self._candidate_problem(candidate)
                        if problem is None:
                            candidate['band_index'] = 0
                            candidate['band_count'] = 0
                            taught_candidates.append(candidate)
                            break
                        first_problem = first_problem or problem
                    else:
                        failures.append('taught grasp {}: {} at every '
                                        'rotation tried'.format(
                                            template.get('name'),
                                            first_problem))
            taught_candidates.sort(
                key=lambda candidate: candidate['template_name']
            )

            band_error = False
            try:
                bands = self._initial_grasp_bands(object_point, object_class)
            except (ValueError, TransformException) as error:
                failures.append(str(error))
                bands = ()
                band_error = True
            usable_candidates = []
            band_choices = () if band_error else (bands or (None,))
            for band_index, band in enumerate(band_choices):
                target_point = object_point if band is None else band[0]
                target_height_offset = (
                    grasp_height_offset if band is None else 0.0
                )
                depth_direction = tuple(
                    value - origin
                    for value, origin in zip(target_point, camera_origin)
                )
                try:
                    generated = grasping.generate_approach_candidates(
                        target_point,
                        approach_origin,
                        self._approach_yaw_offsets,
                        self._approach_distance,
                        self._grasp_center_offset,
                        0.0,
                        forward_grasp_depth,
                        depth_direction,
                        target_height_offset,
                        horizontal_depth or band is not None,
                        level_approach=band is not None,
                    )
                except ValueError as error:
                    failures.append(str(error))
                    continue
                for candidate in generated:
                    problem = self._candidate_problem(candidate)
                    if problem is not None:
                        failures.append(problem)
                        continue
                    candidate = dict(candidate)
                    candidate['band_target_point'] = target_point
                    candidate['band_width'] = None if band is None else band[1]
                    candidate['band_index'] = band_index + 1 if band else 0
                    candidate['band_count'] = len(bands)
                    usable_candidates.append(candidate)
            # Try another band at the same yaw before more oblique yaw angles.
            ordered = taught_candidates + sorted(
                usable_candidates,
                key=lambda candidate: (
                    abs(candidate['yaw_offset']), candidate['band_index']
                ),
            )
            if ordered:
                first = ordered[0]
                first_band = (
                    None if first['band_width'] is None else
                    (first['band_target_point'], first['band_width'])
                )
                viable.append((
                    item, object_point, first['band_target_point'],
                    approach_origin, tuple(ordered), first_band,
                    grasping.distance_between(object_point, arm_origin),
                ))
        if not viable:
            detail = failures[0] if failures else 'no finite object position'
            raise RuntimeError(
                'no approach candidate satisfies reach and height safeguards: '
                + detail
            )
        selected = min(viable, key=lambda value: value[6])
        return selected[:6]

    def _lookup_transform(self, target_frame, source_frame):
        if target_frame == source_frame:
            return None
        return self._tf_buffer.lookup_transform(
            target_frame, source_frame, Time()
        )

    def _transform_point(self, point, transform):
        if transform is None:
            return tuple(point)
        translation = self._point_tuple(transform.transform.translation)
        rotation = transform.transform.rotation
        quaternion = (rotation.x, rotation.y, rotation.z, rotation.w)
        return grasping.transform_point(point, translation, quaternion)

    def _current_tip_orientation(self):
        """Return the live tip orientation in the MoveIt planning frame."""
        transform = self._lookup_transform(
            self._planning_frame, self._tip_link
        )
        if transform is None:
            return (0.0, 0.0, 0.0, 1.0)
        rotation = transform.transform.rotation
        values = (
            float(rotation.x), float(rotation.y),
            float(rotation.z), float(rotation.w),
        )
        norm = math.sqrt(sum(value * value for value in values))
        if not all(math.isfinite(value) for value in values) or norm <= 1e-9:
            raise ValueError('tip transform contains an invalid orientation')
        return tuple(value / norm for value in values)

    def _point_in_reach_frame(self, point):
        transform = self._lookup_transform(
            self._reach_reference_frame, self._planning_frame
        )
        return self._transform_point(point, transform)

    def _target_problem(self, point, description):
        reach_point = self._point_in_reach_frame(point)
        problem = grasping.reach_safeguard_problem(
            reach_point, self._minimum_reach, self._maximum_reach,
            self._reach_reference_frame,
        )
        if problem is not None:
            return problem.replace('pre-grasp target', description)
        if not self._minimum_target_z <= point[2] <= self._maximum_target_z:
            return (
                '{} z is {:.3f} m in {}, outside [{:.3f}, {:.3f}] m'.format(
                    description, point[2], self._planning_frame,
                    self._minimum_target_z, self._maximum_target_z,
                )
            )
        return None

    def _candidate_problem(self, candidate):
        return (
            self._target_problem(candidate['pregrasp_point'], 'pre-grasp target')
            or self._target_problem(candidate['grasp_point'], 'grasp target')
        )

    def _candidate_pose(self, candidate, point_name):
        return self._target_pose(
            candidate[point_name], candidate['orientation']
        )

    def _band_alignment_problem(self, candidate, orientation):
        if candidate['band_width'] is None:
            return None
        error = grasping.gripper_opening_level_error(orientation)
        if error <= self._band_alignment_tolerance_rad:
            return None
        return (
            'measured-band gripper opening is {:.3f} rad from level about '
            'the achieved tool-forward axis (limit {:.3f} rad)'.format(
                error, self._band_alignment_tolerance_rad
            )
        )

    @staticmethod
    def _fixed_approach(candidate):
        return bool(
            candidate.get('taught') or candidate['band_width'] is not None
        )

    def _band_pointing_problem(self, candidate, orientation):
        if candidate['band_width'] is None:
            return None
        error = grasping.approach_axis_error(
            orientation, candidate['approach_direction']
        )
        if error > self._grasp_orientation_tolerance_rad:
            return (
                'gripper forward axis is {:.3f} rad from the measured-band '
                'approach (limit {:.3f} rad)'.format(
                    error, self._grasp_orientation_tolerance_rad
                )
            )
        return None

    def _band_alignment_target(self, candidate, orientation):
        if candidate.get('taught'):
            if self._taught_orientation_problem(candidate, orientation):
                return candidate['taught_orientation']
            return None
        if candidate['band_width'] is None:
            return None
        if self._band_pointing_problem(candidate, orientation) is not None:
            # Position-only transit may reach the band with an arbitrary tool
            # heading. Rotate at pre-grasp to the measured approach direction,
            # choosing the nearer of the two level-jaw wrist rolls.
            options = (
                grasping.quaternion_from_approach(
                    candidate['approach_direction']
                ),
                grasping.quaternion_from_approach(
                    candidate['approach_direction'], math.pi
                ),
            )
            return min(
                options,
                key=lambda option: grasping.quaternion_angular_distance(
                    option, orientation
                ),
            )
        if self._band_alignment_problem(candidate, orientation) is not None:
            return grasping.level_gripper_opening(orientation)
        return None

    def _taught_orientation_problem(self, candidate, orientation):
        if not candidate.get('taught'):
            return None
        error = grasping.quaternion_angular_distance(
            candidate['taught_orientation'], orientation
        )
        if error > self._taught_orientation_tolerance_rad:
            return (
                'gripper is {:.3f} rad from taught wrist orientation '
                '(limit {:.3f} rad)'.format(
                    error, self._taught_orientation_tolerance_rad
                )
            )
        return None

    def _start_next_pregrasp_candidate(self):
        if self._active is None:
            return
        index = self._active['candidate_index']
        candidates = self._active['candidates']
        if index >= len(candidates):
            failures = self._active.get('candidate_failures', [])
            detail = ''
            if failures:
                detail = '; recent failures: ' + ' | '.join(failures[-3:])
            self._finish_failure(
                'no approach candidate passed pre-grasp and Cartesian '
                'approach preflight' + detail
            )
            return
        candidate = candidates[index]
        pose = self._candidate_pose(candidate, 'pregrasp_point')
        self._active['candidate'] = candidate
        self._grasp_log('TRYING candidate {}/{}: {} (planning only, arm '
                        'not moving yet)'.format(
                            index + 1, len(candidates),
                            self._grasp_kind(candidate)))
        if candidate.get('taught'):
            self._begin_taught_candidate(candidate)
            return
        self._active['target_point'] = candidate['band_target_point']
        self._active['grasp_band_width'] = candidate['band_width']
        self._active['pregrasp_pose'] = pose
        self._active['phase'] = 'planning_pregrasp'
        self._pregrasp_publisher.publish(pose)
        band_description = (
            'band {} of {} ({:.3f} m wide)'.format(
                candidate['band_index'], candidate['band_count'],
                candidate['band_width'],
            ) if candidate['band_width'] is not None else 'center target'
        )
        self._publish_active_stage(
            GraspStatus.STAGE_PLANNING,
            'MoveIt is planning pre-grasp candidate {} of {} with {} '
            'at [{:.3f}, {:.3f}, {:.3f}] m '
            '(yaw offset {:.3f} rad; {})'.format(
                index + 1, len(candidates),
                ('taught grasp ' + candidate['template_name']
                 if candidate.get('taught') else
                 ('a position goal' if candidate['band_width'] is not None
                  else 'orientation tolerance')),
                candidate['pregrasp_point'][0],
                candidate['pregrasp_point'][1],
                candidate['pregrasp_point'][2],
                candidate['yaw_offset'],
                band_description,
            ),
            pose,
        )
        orientation_options = None
        if candidate.get('taught'):
            orientation_tolerance = self._taught_orientation_tolerance_rad
        elif candidate['band_width'] is not None:
            # Plan the transit to arrive already pointing along the band
            # approach with level jaws (either wrist roll). A position-only
            # goal let the random planner end at an arbitrary wrist heading,
            # and the large in-place rotation needed afterwards usually had no
            # complete Cartesian path, so band candidates failed or succeeded
            # by chance from run to run.
            orientation_tolerance = self._band_alignment_tolerance_rad
            orientation_options = (
                grasping.quaternion_from_approach(
                    candidate['approach_direction']
                ),
                grasping.quaternion_from_approach(
                    candidate['approach_direction'], math.pi
                ),
            )
        else:
            orientation_tolerance = self._grasp_orientation_tolerance_rad
        future = self._move_group.send_goal_async(
            self._move_group_goal(
                pose,
                position_tolerance=(
                    self._band_pregrasp_position_tolerance_m
                    if self._fixed_approach(candidate) else None
                ),
                orientation_tolerance=orientation_tolerance,
                orientation_options=orientation_options,
            ),
            feedback_callback=lambda message, token=self._active['token']:
                self._move_group_feedback(message, token),
        )
        token = self._active['token']
        future.add_done_callback(
            lambda done, token=token:
                self._move_group_goal_response(done, token)
        )

    def _move_group_goal(
            self, target_pose, position_tolerance=None,
            orientation_tolerance=None, path_constraints=None,
            orientation_options=None, goal_constraints=None):
        if position_tolerance is None:
            position_tolerance = self._position_tolerance_m
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.SPHERE
        primitive.dimensions = [position_tolerance]
        region_pose = Pose()
        region_pose.position = copy.deepcopy(target_pose.pose.position)
        region_pose.orientation.w = 1.0

        position = PositionConstraint()
        position.header = copy.deepcopy(target_pose.header)
        position.link_name = self._tip_link
        position.constraint_region.primitives = [primitive]
        position.constraint_region.primitive_poses = [region_pose]
        position.weight = 1.0

        if orientation_tolerance is None:
            quaternions = (None,)
        elif orientation_options:
            quaternions = tuple(orientation_options)
        else:
            quaternions = ((
                target_pose.pose.orientation.x,
                target_pose.pose.orientation.y,
                target_pose.pose.orientation.z,
                target_pose.pose.orientation.w,
            ),)
        goal_constraints_list = []
        # MoveIt treats several goal constraint sets as alternatives.
        for quaternion in quaternions:
            constraints = Constraints()
            constraints.name = (
                'semantic_pose_target'
                if quaternion is not None
                else 'semantic_position_target'
            )
            constraints.position_constraints = [copy.deepcopy(position)]
            if quaternion is not None:
                orientation = OrientationConstraint()
                orientation.header = copy.deepcopy(target_pose.header)
                orientation.link_name = self._tip_link
                orientation.orientation.x = float(quaternion[0])
                orientation.orientation.y = float(quaternion[1])
                orientation.orientation.z = float(quaternion[2])
                orientation.orientation.w = float(quaternion[3])
                orientation.absolute_x_axis_tolerance = orientation_tolerance
                orientation.absolute_y_axis_tolerance = orientation_tolerance
                orientation.absolute_z_axis_tolerance = orientation_tolerance
                orientation.weight = 1.0
                constraints.orientation_constraints = [orientation]
            goal_constraints_list.append(constraints)

        goal = MoveGroup.Goal()
        goal.request.start_state.is_diff = True
        goal.request.goal_constraints = (
            goal_constraints_list
            if goal_constraints is None else list(goal_constraints)
        )
        goal.request.pipeline_id = ''
        goal.request.planner_id = self._planner_id
        goal.request.group_name = self._planning_group
        goal.request.num_planning_attempts = self._planning_attempts
        goal.request.allowed_planning_time = self._planning_time
        goal.request.max_velocity_scaling_factor = self._velocity_scaling
        goal.request.max_acceleration_scaling_factor = (
            self._acceleration_scaling
        )
        if path_constraints is not None:
            goal.request.path_constraints = copy.deepcopy(path_constraints)
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True
        # Planning and execution are kept separate so every trajectory can be
        # inspected for success before it is sent to the physical controller.
        goal.planning_options.plan_only = True
        goal.planning_options.look_around = False
        goal.planning_options.replan = False
        return goal

    def _approach_path_constraints(self, start_point, target_pose):
        target_point = self._point_tuple(target_pose.pose.position)
        geometry = grasping.approach_corridor_geometry(
            start_point, target_point, self._approach_corridor_radius_m
        )

        corridor = SolidPrimitive()
        corridor.type = SolidPrimitive.BOX
        corridor.dimensions = list(geometry['dimensions'])
        corridor_pose = Pose()
        corridor_pose.position.x = geometry['center'][0]
        corridor_pose.position.y = geometry['center'][1]
        corridor_pose.position.z = geometry['center'][2]
        corridor_pose.orientation.x = geometry['orientation'][0]
        corridor_pose.orientation.y = geometry['orientation'][1]
        corridor_pose.orientation.z = geometry['orientation'][2]
        corridor_pose.orientation.w = geometry['orientation'][3]

        position = PositionConstraint()
        position.header = copy.deepcopy(target_pose.header)
        position.link_name = self._tip_link
        position.constraint_region.primitives = [corridor]
        position.constraint_region.primitive_poses = [corridor_pose]
        position.weight = 1.0

        orientation = OrientationConstraint()
        orientation.header = copy.deepcopy(target_pose.header)
        orientation.link_name = self._tip_link
        orientation.orientation = copy.deepcopy(target_pose.pose.orientation)
        orientation.absolute_x_axis_tolerance = (
            self._grasp_orientation_tolerance_rad
        )
        orientation.absolute_y_axis_tolerance = (
            self._grasp_orientation_tolerance_rad
        )
        orientation.absolute_z_axis_tolerance = (
            self._grasp_orientation_tolerance_rad
        )
        orientation.weight = 1.0

        constraints = Constraints()
        constraints.name = 'straight_final_approach_corridor'
        constraints.position_constraints = [position]
        constraints.orientation_constraints = [orientation]
        return constraints

    def _active_token_valid(self, token):
        return self._active is not None and self._active['token'] is token

    def _move_group_feedback(self, feedback_message, token):
        if not self._active_token_valid(token):
            return
        state = feedback_message.feedback.state.strip()
        if 'execut' in state.lower() or 'monitor' in state.lower():
            self._publish_active_stage(
                GraspStatus.STAGE_EXECUTING,
                'MoveIt state: {}'.format(state or 'unknown'),
            )

    def _move_group_goal_response(self, future, token):
        if not self._active_token_valid(token) or (
                self._active['phase'] != 'planning_pregrasp'):
            return
        try:
            goal_handle = future.result()
        except Exception as error:
            self._finish_failure(
                'failed to send pre-grasp request: {}'.format(error)
            )
            return
        if not goal_handle.accepted:
            self._try_next_candidate('MoveIt rejected the candidate')
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda done, token=token: self._move_group_result(done, token)
        )

    def _move_group_result(self, future, token):
        if not self._active_token_valid(token) or (
                self._active['phase'] != 'planning_pregrasp'):
            return
        try:
            result = future.result().result
            error_code = result.error_code.val
        except Exception as error:
            self._finish_failure(
                'failed while waiting for MoveIt: {}'.format(error)
            )
            return
        if error_code != MoveItErrorCodes.SUCCESS:
            description = MOVEIT_ERROR_NAMES.get(
                error_code, 'MoveIt error {}'.format(error_code)
            )
            if error_code in (
                    MoveItErrorCodes.CONTROL_FAILED,
                    MoveItErrorCodes.COMMUNICATION_FAILURE,
                    MoveItErrorCodes.ROBOT_STATE_STALE):
                self._finish_failure(description)
            else:
                self._try_next_candidate(description)
            return

        trajectory = result.planned_trajectory
        if not trajectory.joint_trajectory.points:
            self._try_next_candidate('MoveIt returned an empty trajectory')
            return
        self._check_planned_pregrasp(
            result.trajectory_start, trajectory, 'pregrasp'
        )

    def _preflight_failure(self, reason):
        if self._active['preflight_motion_purpose'] == 'pregrasp':
            self._try_next_candidate(reason)
        else:
            self._finish_failure(
                'corrected pre-grasp has no validated approach: ' + reason
            )

    def _check_planned_pregrasp(self, start_state, trajectory, purpose):
        self._active['preflight_motion_purpose'] = purpose
        try:
            predicted_state = grasping.trajectory_endpoint_state(
                start_state, trajectory
            )
        except ValueError as error:
            self._preflight_failure(str(error))
            return
        self._active['preflight_trajectory'] = trajectory
        self._active['predicted_pregrasp_state'] = predicted_state
        self._active['phase'] = 'computing_pregrasp_fk'
        self._publish_active_stage(
            GraspStatus.STAGE_PLANNING,
            '{} plan found; checking its predicted endpoint and '
            'complete final approach before arm execution'.format(
                purpose.replace('_', ' ')
            ),
            self._active['pregrasp_pose'],
        )
        request = GetPositionFK.Request()
        request.header.stamp = self.get_clock().now().to_msg()
        request.header.frame_id = self._planning_frame
        request.fk_link_names = [self._tip_link]
        request.robot_state = copy.deepcopy(predicted_state)
        future = self._forward_kinematics.call_async(request)
        token = self._active['token']
        future.add_done_callback(
            lambda done, token=token: self._pregrasp_fk_result(done, token)
        )

    def _pregrasp_fk_result(self, future, token):
        if not self._active_token_valid(token) or (
                self._active['phase'] != 'computing_pregrasp_fk'):
            return
        try:
            response = future.result()
        except Exception as error:
            self._preflight_failure(
                'predicted pre-grasp FK failed: {}'.format(error)
            )
            return
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            self._preflight_failure(
                'predicted pre-grasp FK failed: {}'.format(
                    MOVEIT_ERROR_NAMES.get(
                        response.error_code.val,
                        'MoveIt error {}'.format(response.error_code.val),
                    )
                )
            )
            return
        if (len(response.fk_link_names) != 1
                or response.fk_link_names[0] != self._tip_link
                or len(response.pose_stamped) != 1):
            self._preflight_failure(
                'predicted pre-grasp FK returned no unique tip pose'
            )
            return
        predicted_pose = response.pose_stamped[0]
        if (predicted_pose.header.frame_id.lstrip('/')
                != self._planning_frame.lstrip('/')):
            self._preflight_failure(
                'predicted pre-grasp FK used an unexpected frame'
            )
            return
        if not all(math.isfinite(value) for value in self._point_tuple(
                predicted_pose.pose.position)):
            self._preflight_failure(
                'predicted pre-grasp FK returned an invalid position'
            )
            return
        quaternion = predicted_pose.pose.orientation
        values = (
            float(quaternion.x), float(quaternion.y),
            float(quaternion.z), float(quaternion.w),
        )
        norm = math.sqrt(sum(value * value for value in values))
        if not all(math.isfinite(value) for value in values) or norm <= 1e-9:
            self._preflight_failure(
                'predicted pre-grasp FK returned an invalid orientation'
            )
            return
        candidate = dict(self._active['candidate'])
        candidate['orientation'] = tuple(value / norm for value in values)
        self._active['candidate'] = candidate
        self._active['pregrasp_pose'] = predicted_pose
        requested = self._band_alignment_target(
            candidate, candidate['orientation']
        )
        if requested is not None:
            # The transit plan may reach a safe pre-grasp with the wrist not
            # yet pointing at the band or with the jaws not level. Validate a
            # separate rotation there before checking the straight contact
            # segment; do not demand exact orientation of the transit plan.
            aligned_pose = copy.deepcopy(predicted_pose)
            aligned_pose.pose.orientation.x = requested[0]
            aligned_pose.pose.orientation.y = requested[1]
            aligned_pose.pose.orientation.z = requested[2]
            aligned_pose.pose.orientation.w = requested[3]
            self._active['cartesian_start_state'] = (
                self._active['predicted_pregrasp_state']
            )
            self._active['cartesian_best_fraction'] = 0.0
            self._request_cartesian_path(
                aligned_pose, 'alignment_preflight',
                GraspStatus.STAGE_PLANNING,
                'checking wrist alignment at pre-grasp before the straight '
                'band approach',
            )
            return
        grasp_pose = self._candidate_pose(candidate, 'grasp_point')
        self._active['grasp_pose'] = grasp_pose
        self._begin_final_approach(
            grasp_pose, 'preflight', GraspStatus.STAGE_PLANNING,
            'checking the full Cartesian approach from the predicted {} '
            'endpoint'.format(
                self._active['preflight_motion_purpose'].replace('_', ' ')
            ),
            start_state=self._active['predicted_pregrasp_state'],
        )

    def _begin_reacquisition(self):
        self._active['phase'] = 'reacquiring'
        self._active['required_detection_receipt'] = self._now_seconds()
        self._active['deadline'] = (
            self._now_seconds() + self._reacquire_timeout_sec
        )
        self._publish_active_stage(
            GraspStatus.STAGE_PREGRASP_REACHED,
            'pre-grasp reached; waiting for a new target observation before '
            'the final approach',
            self._active['pregrasp_pose'],
        )
        self._publish_active_stage(
            GraspStatus.STAGE_REACQUIRING,
            'reacquiring the selected object from a fresh detection',
            self._active['pregrasp_pose'],
        )

    def _try_next_candidate(self, reason):
        if self._active is None:
            return
        candidate = self._active['candidate']
        band_label = (
            'band {}'.format(candidate['band_index'])
            if candidate['band_width'] is not None else 'center target'
        )
        if candidate.get('taught'):
            band_label = 'taught {} {}'.format(
                candidate['template_name'], band_label
            )
        self._active.setdefault('candidate_failures', []).append(
            '{} ({}): {}'.format(
                self._active['candidate_index'] + 1, band_label, reason
            )
        )
        self._grasp_log(
            'REJECTED candidate {}/{}: {}: {}'.format(
                self._active['candidate_index'] + 1,
                len(self._active['candidates']),
                self._grasp_kind(candidate), reason,
            ),
            'warn',
        )
        self._publish_active_stage(
            GraspStatus.STAGE_PLANNING,
            'candidate {} ({}) rejected: {}'.format(
                self._active['candidate_index'] + 1, band_label, reason
            ),
        )
        self._active['candidate_index'] += 1
        self._start_next_pregrasp_candidate()

    def _try_reacquire(self, message, receipt_time):
        if self._active is None or self._active['phase'] != 'reacquiring':
            return
        if receipt_time <= self._active['required_detection_receipt']:
            return
        if self._now_seconds() - receipt_time > self._fresh_detection_age_sec:
            return
        matches = []
        for detection in message.detections:
            label = detection.requested_target.strip()
            if not label:
                label = detection.label.strip()
            if grasping.normalize_label(label) != self._active['object_class']:
                continue
            if float(detection.score) < self._minimum_confidence:
                continue
            try:
                point = self._detection_point(message, detection)
            except Exception as error:
                self.get_logger().warning(
                    'Could not transform fresh detection: {}'.format(error)
                )
                continue
            if point is None:
                continue
            separation = grasping.distance_between(
                point, self._active['object_point']
            )
            if separation <= self._reacquire_match_distance_m:
                matches.append((separation, point, detection))
        if not matches:
            return

        separation, object_point, detection = min(
            matches, key=lambda item: item[0]
        )
        if separation > self._maximum_reacquire_correction_m:
            self._finish_failure(
                'fresh detection moved {:.3f} m from the mapped point, above '
                'the {:.3f} m correction safeguard'.format(
                    separation, self._maximum_reacquire_correction_m
                )
            )
            return
        bands = ()
        if self._active['grasp_band_width'] is not None:
            try:
                bands = self._grasp_band_points(message, detection)
            except (ValueError, TransformException) as error:
                self._finish_failure(str(error))
                return
        band = None
        if self._active['grasp_band_width'] is not None:
            if not bands:
                # Preserve the validated band height rather than switching to
                # the old center target after physical pre-grasp execution.
                return
            band = min(
                bands,
                key=lambda item: grasping.distance_between(
                    item[0], self._active['target_point']
                ),
            )
            if (grasping.distance_between(
                    band[0], self._active['target_point'])
                    > self._maximum_reacquire_correction_m):
                return
        target_point = object_point if band is None else band[0]
        target_height_offset = (
            self._active['grasp_height_offset'] if band is None else 0.0
        )
        selected = self._active['candidate']
        camera_frame = message.camera_frame.strip() or self._camera_frame
        try:
            camera_transform = self._lookup_transform(
                self._planning_frame, camera_frame
            )
        except Exception as error:
            self._finish_failure(
                'could not transform camera frame {} for depth correction: '
                '{}'.format(camera_frame, error)
            )
            return
        camera_origin = (
            (0.0, 0.0, 0.0)
            if camera_transform is None
            else self._point_tuple(camera_transform.transform.translation)
        )
        depth_direction = tuple(
            value - origin
            for value, origin in zip(target_point, camera_origin)
        )
        if selected.get('taught'):
            try:
                yaw = float(selected.get('yaw_offset', 0.0))
                reference = grasp_templates.object_reference(
                    object_point, self._active['approach_origin'], yaw
                )
                candidate = grasp_templates.candidate_from_template(
                    selected['template'], reference, yaw
                )
            except ValueError as error:
                self._finish_failure('fresh taught grasp is invalid: ' + str(error))
                return
            candidate['band_index'] = selected['band_index']
            candidate['band_count'] = selected['band_count']
        else:
            candidate = grasping.retarget_approach_candidate(
                selected, target_point, self._approach_distance,
                self._grasp_center_offset,
                self._active['forward_grasp_depth_offset'],
                depth_direction,
                target_height_offset,
                self._active['horizontal_depth'] or band is not None,
            )
        problem = self._candidate_problem(candidate)
        if problem is not None:
            self._finish_failure('reacquired target is unsafe: ' + problem)
            return
        new_pregrasp = tuple(
            float(value) for value in candidate['pregrasp_point']
        )
        try:
            tip_transform = self._lookup_transform(
                self._planning_frame, self._tip_link
            )
            actual_pregrasp = self._point_tuple(
                tip_transform.transform.translation
            )
        except Exception as error:
            self._finish_failure(
                'could not read the reached pre-grasp position: {}'.format(error)
            )
            return
        try:
            achieved_orientation = self._current_tip_orientation()
            lateral_correction = grasping.approach_line_error(
                actual_pregrasp, new_pregrasp,
                candidate['approach_direction'],
            )
        except ValueError:
            self._finish_failure(
                'reacquired approach direction has near-zero magnitude'
            )
            return

        # The pre-grasp MoveGroup goal permits an orientation tolerance. Its
        # achieved orientation can therefore differ from the generated ideal
        # quaternion. GetCartesianPath accepts exact poses rather than an
        # orientation tolerance, so retaining the generated quaternion here
        # would turn the final approach into a translation plus a rotation and
        # can make an otherwise valid straight approach return a low fraction.
        candidate = dict(candidate)
        candidate['orientation'] = achieved_orientation

        grasp_pose = self._candidate_pose(candidate, 'grasp_point')
        retreat_pose = self._candidate_pose(candidate, 'pregrasp_point')
        self._active['object_point'] = object_point
        self._active['target_point'] = target_point
        self._active['grasp_band_width'] = None if band is None else band[1]
        self._active['candidate'] = candidate
        self._active['pregrasp_pose'] = retreat_pose
        self._active['grasp_pose'] = grasp_pose
        self._active['retreat_pose'] = retreat_pose
        self._grasp_publisher.publish(grasp_pose)

        lateral_limit = self._maximum_final_lateral_correction_m
        if lateral_correction > lateral_limit and candidate.get('taught'):
            self._begin_taught_reposition(
                'fresh target shifted laterally by {:.3f} m'.format(
                    lateral_correction
                )
            )
            return
        if lateral_correction > lateral_limit:
            self._pregrasp_publisher.publish(retreat_pose)
            self._request_position_plan(
                retreat_pose,
                'reposition_pregrasp',
                GraspStatus.STAGE_APPROACHING,
                'fresh target shifted laterally by {:.3f} m; repositioning to '
                'the corrected pre-grasp before final approach'.format(
                    lateral_correction
                ),
            )
            return

        self._align_or_approach_from_pregrasp(
            actual_pregrasp, achieved_orientation,
            'fresh {} acquired; computing the short Cartesian final approach'.format(
                'narrow band ({:.3f} m)'.format(band[1])
                if band is not None else 'center target'
            ),
        )

    def _align_or_approach_from_pregrasp(
            self, actual_pregrasp, achieved_orientation, message):
        candidate = self._active['candidate']
        if self._fixed_approach(candidate):
            # The reached pose, not the nominal pre-grasp goal, starts both
            # the in-place wrist adjustment and fixed-orientation approach.
            self._active['pregrasp_pose'] = self._target_pose(
                actual_pregrasp, achieved_orientation
            )
        requested = self._band_alignment_target(
            candidate, achieved_orientation
        )
        if requested is not None:
            alignment_pose = self._target_pose(
                actual_pregrasp, requested,
            )
            self._active['cartesian_start_state'] = None
            self._active['cartesian_best_fraction'] = 0.0
            self._request_cartesian_path(
                alignment_pose, 'align_pregrasp',
                GraspStatus.STAGE_APPROACHING,
                ('aligning the open gripper with the taught pose at '
                 'pre-grasp before the straight approach'
                 if candidate.get('taught') else
                 'aligning the open gripper with the measured band at '
                 'pre-grasp before the straight approach'),
            )
            return
        candidate = dict(candidate)
        candidate['orientation'] = achieved_orientation
        self._active['candidate'] = candidate
        grasp_pose = self._candidate_pose(candidate, 'grasp_point')
        self._active['grasp_pose'] = grasp_pose
        self._begin_final_approach(
            grasp_pose, 'approach', GraspStatus.STAGE_APPROACHING, message
        )

    def _begin_final_approach(
            self, pose, purpose, stage, message, start_state=None):
        orientation = pose.pose.orientation
        self._active['cartesian_start_state'] = start_state
        achieved_orientation = (
            orientation.x, orientation.y, orientation.z, orientation.w
        )
        self._active['cartesian_orientations'] = (
            (achieved_orientation,)
            if self._fixed_approach(self._active['candidate'])
            else grasping.approach_orientation_candidates(
                achieved_orientation, self._grasp_orientation_tolerance_rad
            )
        )
        self._active['cartesian_orientation_index'] = 0
        self._active['cartesian_best_fraction'] = 0.0
        self._request_cartesian_path(pose, purpose, stage, message)

    def _intermediate_cartesian_waypoints(self, pose, purpose=None):
        if (purpose in {'preflight', 'approach'}
                and self._fixed_approach(self._active['candidate'])):
            # MoveIt interpolates from request.start_state to this one pose.
            # Do not insert the stored pre-alignment pose or replace the
            # reached orientation during the straight contact motion.
            return [copy.deepcopy(pose.pose)]
        start_pose = self._active.get('pregrasp_pose')
        if start_pose is None or pose is None:
            return [copy.deepcopy(pose.pose)] if pose is not None else []
        start = copy.deepcopy(start_pose.pose)
        target = copy.deepcopy(pose.pose)
        if purpose in {'align_pregrasp', 'alignment_preflight'}:
            # The request start_state already supplies the current pose. An
            # explicit start waypoint can produce duplicate zero-time points
            # when MoveIt time-parameterizes an in-place wrist rotation.
            return [target]
        start_position = (
            float(start.position.x), float(start.position.y),
            float(start.position.z),
        )
        target_position = (
            float(target.position.x), float(target.position.y),
            float(target.position.z),
        )
        delta = tuple(
            target_value - start_value
            for start_value, target_value in zip(
                start_position, target_position
            )
        )
        length = math.sqrt(sum(value * value for value in delta))
        if length <= 1.0e-9:
            return [start, target]
        mid_point = tuple(
            start_value + 0.5 * offset for start_value, offset in zip(
                start_position, delta
            )
        )
        mid_pose = copy.deepcopy(start)
        mid_pose.position.x = mid_point[0]
        mid_pose.position.y = mid_point[1]
        mid_pose.position.z = mid_point[2]
        # Fixed-reference grasps keep wrist orientation constant while moving.
        mid_pose.orientation = copy.deepcopy(start.orientation)
        target.orientation = copy.deepcopy(start.orientation)
        return [start, mid_pose, target]

    def _request_cartesian_path(self, pose, purpose, stage, message):
        if self._active is None:
            return
        self._active['phase'] = 'computing_' + purpose
        self._active['cartesian_purpose'] = purpose
        self._active['cartesian_probe_pose'] = pose
        self._publish_active_stage(stage, message, pose)

        request = GetCartesianPath.Request()
        request.header.stamp = self.get_clock().now().to_msg()
        request.header.frame_id = self._planning_frame
        start_state = self._active['cartesian_start_state']
        if start_state is None:
            request.start_state.is_diff = True
        else:
            request.start_state = copy.deepcopy(start_state)
        request.group_name = self._planning_group
        request.link_name = self._tip_link
        waypoints = self._intermediate_cartesian_waypoints(pose, purpose)
        if not waypoints:
            waypoints = [copy.deepcopy(pose.pose)]
        request.waypoints = waypoints
        request.max_step = self._cartesian_step_m
        if hasattr(request, 'jump_threshold'):
            request.jump_threshold = self._cartesian_jump_threshold
        request.avoid_collisions = True

        # Newer MoveIt service definitions expose speed scaling directly.
        # Keep the hasattr guards so this remains compatible with older ROS 2
        # / MoveIt message definitions.
        if hasattr(request, 'max_velocity_scaling_factor'):
            request.max_velocity_scaling_factor = (
                self._cartesian_velocity_scaling
            )
        if hasattr(request, 'max_acceleration_scaling_factor'):
            request.max_acceleration_scaling_factor = (
                self._cartesian_acceleration_scaling
            )

        future = self._cartesian_path.call_async(request)
        token = self._active['token']
        future.add_done_callback(
            lambda done, token=token:
                self._cartesian_path_result(done, token)
        )

    def _prepare_cartesian_solution(self, solution, purpose):
        """Keep a complete Cartesian path executable by the D1 controller."""
        joint_path = solution.joint_trajectory
        try:
            trajectory.validate_trajectory(
                joint_path.joint_names, joint_path.points, D1_ARM_JOINT_NAMES
            )
            return
        except trajectory.TrajectoryError as error:
            # Only timing can be repaired without changing the planned path.
            if ('strictly increasing' not in str(error)
                    and 'time_from_start' not in str(error)):
                raise

        if set(joint_path.joint_names) != set(D1_ARM_JOINT_NAMES):
            raise trajectory.TrajectoryError(
                'Cartesian trajectory must contain exactly the six arm joints'
            )
        times = trajectory.conservative_waypoint_times(
            (point.positions for point in joint_path.points),
            D1_ARM_MIN_VELOCITY_LIMIT * self._cartesian_velocity_scaling,
            D1_ARM_MIN_ACCELERATION_LIMIT
            * self._cartesian_acceleration_scaling,
        )
        for point, seconds in zip(joint_path.points, times):
            nanoseconds = round(seconds * 1e9)
            point.time_from_start = Duration(
                sec=nanoseconds // 1000000000,
                nanosec=nanoseconds % 1000000000,
            )
            # The incoming derivatives describe the old timing, if present.
            point.velocities = []
            point.accelerations = []
        trajectory.validate_trajectory(
            joint_path.joint_names, joint_path.points, D1_ARM_JOINT_NAMES
        )
        self.get_logger().warning(
            'MoveIt returned non-increasing Cartesian {} timestamps; '
            'applied conservative joint-limit timing'.format(purpose)
        )

    def _cartesian_path_result(self, future, token):
        if (
                not self._active_token_valid(token)
                or not self._active['phase'].startswith('computing_')):
            return
        purpose = self._active['cartesian_purpose']
        try:
            response = future.result()
        except Exception as error:
            self._cartesian_path_failure(
                purpose, 'Cartesian {} service failed: {}'.format(
                    purpose, error
                )
            )
            return

        error_code = response.error_code.val
        if error_code != MoveItErrorCodes.SUCCESS:
            self._cartesian_path_failure(
                purpose, 'Cartesian {} failed: {}'.format(
                    purpose,
                    MOVEIT_ERROR_NAMES.get(
                        error_code, 'MoveIt error {}'.format(error_code)
                    ),
                )
            )
            return

        fraction = float(response.fraction)
        if not math.isfinite(fraction):
            self._cartesian_path_failure(
                purpose, 'Cartesian {} returned a non-finite fraction'.format(
                    purpose
                )
            )
            return
        self._active['cartesian_best_fraction'] = max(
            self._active['cartesian_best_fraction'], fraction
        )
        if fraction < self._minimum_cartesian_fraction:
            self._cartesian_path_failure(
                purpose,
                'Cartesian {} covered only {:.1f}% (minimum {:.1f}%)'.format(
                    purpose,
                    100.0 * fraction,
                    100.0 * self._minimum_cartesian_fraction,
                ),
            )
            return

        if not response.solution.joint_trajectory.points:
            self._cartesian_path_failure(
                purpose, 'MoveIt returned an empty Cartesian {} trajectory'.format(
                    purpose
                )
            )
            return

        try:
            self._prepare_cartesian_solution(response.solution, purpose)
        except trajectory.TrajectoryError as error:
            self._cartesian_path_failure(
                purpose, 'Cartesian {} trajectory cannot be executed: {}'.format(
                    purpose, error
                )
            )
            return

        accepted_pose = self._active['cartesian_probe_pose']
        if purpose == 'alignment_preflight':
            try:
                aligned_state = grasping.trajectory_endpoint_state(
                    self._active['predicted_pregrasp_state'],
                    response.solution,
                )
            except ValueError as error:
                self._preflight_failure(
                    'wrist alignment preflight returned an invalid state: '
                    + str(error)
                )
                return
            candidate = dict(self._active['candidate'])
            candidate['orientation'] = (
                accepted_pose.pose.orientation.x,
                accepted_pose.pose.orientation.y,
                accepted_pose.pose.orientation.z,
                accepted_pose.pose.orientation.w,
            )
            self._active['candidate'] = candidate
            grasp_pose = self._candidate_pose(candidate, 'grasp_point')
            self._active['grasp_pose'] = grasp_pose
            self._begin_final_approach(
                grasp_pose, 'preflight', GraspStatus.STAGE_PLANNING,
                'checking the straight approach after pre-grasp wrist '
                'alignment', start_state=aligned_state,
            )
            return
        if purpose == 'align_pregrasp':
            self._execute_robot_trajectory(
                response.solution,
                purpose,
                GraspStatus.STAGE_APPROACHING,
                'aligning the open gripper at pre-grasp',
                accepted_pose,
            )
            return
        if purpose == 'preflight':
            orientation = accepted_pose.pose.orientation
            candidate = dict(self._active['candidate'])
            candidate['orientation'] = (
                orientation.x, orientation.y, orientation.z, orientation.w
            )
            self._active['candidate'] = candidate
            self._active['grasp_pose'] = accepted_pose
            self._pregrasp_publisher.publish(
                self._active['pregrasp_pose']
            )
            self._grasp_publisher.publish(accepted_pose)
            if not self._execution_enabled:
                self._finish_success(
                    GraspStatus.STAGE_PLAN_READY,
                    'pre-grasp and complete Cartesian approach validated; '
                    'execution_enabled is false',
                )
                return
            motion_purpose = self._active['preflight_motion_purpose']
            self._execute_robot_trajectory(
                self._active['preflight_trajectory'],
                motion_purpose,
                (GraspStatus.STAGE_APPROACHING
                 if motion_purpose == 'reposition_pregrasp'
                 else GraspStatus.STAGE_EXECUTING),
                'executing {} after full approach validation'.format(
                    motion_purpose.replace('_', ' ')
                ),
                self._active['pregrasp_pose'],
            )
            return

        orientation = accepted_pose.pose.orientation
        candidate = dict(self._active['candidate'])
        candidate['orientation'] = (
            orientation.x, orientation.y, orientation.z, orientation.w
        )
        self._active['candidate'] = candidate
        self._active['grasp_pose'] = accepted_pose
        self._active['retreat_pose'] = self._candidate_pose(
            candidate, 'pregrasp_point'
        )
        self._grasp_publisher.publish(accepted_pose)
        self._execute_robot_trajectory(
            response.solution,
            purpose,
            GraspStatus.STAGE_APPROACHING,
            'executing short Cartesian {}'.format(purpose),
            self._active['target_pose'],
        )

    def _try_next_cartesian_orientation(self, reason):
        index = self._active['cartesian_orientation_index'] + 1
        orientations = self._active['cartesian_orientations']
        if index >= len(orientations):
            return False
        self._active['cartesian_orientation_index'] = index
        pose = copy.deepcopy(self._active['grasp_pose'])
        quaternion = orientations[index]
        pose.pose.orientation.x = quaternion[0]
        pose.pose.orientation.y = quaternion[1]
        pose.pose.orientation.z = quaternion[2]
        pose.pose.orientation.w = quaternion[3]
        self._request_cartesian_path(
            pose,
            self._active['cartesian_purpose'],
            (GraspStatus.STAGE_PLANNING
             if self._active['cartesian_purpose'] == 'preflight'
             else GraspStatus.STAGE_APPROACHING),
            '{}; checking nearby wrist orientation {} of {} along the same '
            'straight approach'.format(reason, index + 1, len(orientations)),
        )
        return True

    def _cartesian_path_failure(self, purpose, description):
        if purpose == 'alignment_preflight':
            self._preflight_failure(
                'no complete wrist alignment at planned pre-grasp; '
                + description
            )
            return
        if purpose == 'align_pregrasp':
            self._finish_failure(
                'no complete wrist alignment at reached pre-grasp; '
                + description
            )
            return
        if purpose == 'preflight':
            if self._try_next_cartesian_orientation(description):
                return
            self._preflight_failure(
                'no complete Cartesian approach from planned pre-grasp; '
                'best fraction {:.1f}%; {}'.format(
                    100.0 * self._active['cartesian_best_fraction'],
                    description,
                )
            )
            return
        if purpose == 'approach':
            if self._try_next_cartesian_orientation(description):
                return
            if (self._active['candidate'].get('taught')
                    and not self._active.get('taught_retry_used')):
                self._active['taught_retry_used'] = True
                self._begin_taught_reposition(
                    'straight approach from the reached pre-grasp was '
                    'incomplete ({})'.format(description)
                )
                return
            if self._fixed_approach(self._active['candidate']):
                self._finish_failure(
                    '{}; no complete straight Cartesian approach to the '
                    'fixed grasp target. A corridor plan could move the '
                    'gripper off that line, so no contact motion '
                    'was executed'.format(description)
                )
                return
            try:
                transform = self._lookup_transform(
                    self._planning_frame, self._tip_link
                )
                if transform is None:
                    start_point = (0.0, 0.0, 0.0)
                else:
                    start_point = self._point_tuple(
                        transform.transform.translation
                    )
                achieved_orientation = self._current_tip_orientation()
            except Exception as error:
                self._finish_failure(
                    '{}; could not construct constrained fallback: {}'.format(
                        description, error
                    )
                )
                return

            candidate = dict(self._active['candidate'])
            candidate['orientation'] = achieved_orientation
            self._active['candidate'] = candidate
            self._active['grasp_pose'] = self._candidate_pose(
                candidate, 'grasp_point'
            )
            self._active['retreat_pose'] = self._candidate_pose(
                candidate, 'pregrasp_point'
            )
            self._active['pregrasp_pose'] = self._active['retreat_pose']
            self._grasp_publisher.publish(self._active['grasp_pose'])
            try:
                path_constraints = self._approach_path_constraints(
                    start_point, self._active['grasp_pose']
                )
            except ValueError as error:
                self._finish_failure(
                    '{}; could not construct constrained fallback: {}'.format(
                        description, error
                    )
                )
                return
            self._request_position_plan(
                self._active['grasp_pose'],
                'approach',
                GraspStatus.STAGE_APPROACHING,
                '{}; best of {} collision-checked Cartesian orientations '
                'covered {:.1f}%. Planning from the reached pre-grasp inside '
                'a {:.3f} m corridor'.format(
                    description,
                    len(self._active['cartesian_orientations']),
                    100.0 * self._active['cartesian_best_fraction'],
                    self._approach_corridor_radius_m,
                ),
                path_constraints,
            )
        else:
            self._finish_failure(description)

    def _request_position_plan(
            self, pose, purpose, stage, message, path_constraints=None,
            goal_constraints=None):
        if self._active is None:
            return
        self._active['phase'] = 'planning_' + purpose
        self._active['position_plan_purpose'] = purpose
        self._publish_active_stage(stage, message, pose)
        tolerance = (
            self._grasp_position_tolerance_m
            if purpose == 'approach' or (
                purpose == 'reposition_pregrasp'
                and self._fixed_approach(self._active['candidate'])
            )
            else self._position_tolerance_m
        )
        preserve_grasp_orientation = purpose in (
            'approach', 'reposition_pregrasp'
        )
        future = self._move_group.send_goal_async(
            self._move_group_goal(
                pose,
                tolerance,
                self._grasp_orientation_tolerance_rad
                if preserve_grasp_orientation else None,
                path_constraints,
                goal_constraints=goal_constraints,
            ),
            feedback_callback=lambda feedback, token=self._active['token']:
                self._move_group_feedback(feedback, token),
        )
        token = self._active['token']
        future.add_done_callback(
            lambda done, token=token:
                self._position_goal_response(done, token)
        )

    def _position_goal_response(self, future, token):
        if not self._active_token_valid(token) or (
                not self._active['phase'].startswith('planning_')):
            return
        purpose = self._active['position_plan_purpose']
        try:
            goal_handle = future.result()
        except Exception as error:
            self._position_plan_failure(
                purpose, 'failed to send {} plan: {}'.format(purpose, error)
            )
            return
        if not goal_handle.accepted:
            self._position_plan_failure(
                purpose, 'MoveIt rejected the {} position goal'.format(purpose)
            )
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda done, token=token:
                self._position_plan_result(done, token)
        )

    def _position_plan_result(self, future, token):
        if not self._active_token_valid(token) or (
                not self._active['phase'].startswith('planning_')):
            return
        purpose = self._active['position_plan_purpose']
        try:
            result = future.result().result
            error_code = result.error_code.val
        except Exception as error:
            self._position_plan_failure(
                purpose, 'failed while waiting for the {} plan: {}'.format(
                    purpose, error
                )
            )
            return
        if error_code != MoveItErrorCodes.SUCCESS:
            self._position_plan_failure(
                purpose, '{} planning failed: {}'.format(
                    purpose, MOVEIT_ERROR_NAMES.get(
                        error_code, 'MoveIt error {}'.format(error_code)
                    )
                )
            )
            return
        trajectory = result.planned_trajectory
        if not trajectory.joint_trajectory.points:
            self._position_plan_failure(
                purpose, 'MoveIt returned an empty {} trajectory'.format(
                    purpose
                )
            )
            return
        if purpose == 'reposition_pregrasp':
            self._check_planned_pregrasp(
                result.trajectory_start, trajectory, purpose
            )
            return
        is_approach_motion = purpose == 'approach'
        self._execute_robot_trajectory(
            trajectory,
            purpose,
            GraspStatus.STAGE_APPROACHING if is_approach_motion else (
                GraspStatus.STAGE_RETREATING
            ),
            'executing {}'.format(purpose.replace('_', ' ')),
            self._active['target_pose'],
        )

    def _position_plan_failure(self, purpose, description):
        if purpose == 'approach':
            target = self._active['grasp_pose'].pose.position
            self._finish_failure(
                'final approach failed at target [{:.3f}, {:.3f}, {:.3f}] m '
                'in {}: best of {} Cartesian orientations covered {:.1f}%; '
                'constrained MoveIt plan: {}'.format(
                    target.x, target.y, target.z,
                    self._planning_frame,
                    len(self._active['cartesian_orientations']),
                    100.0 * self._active['cartesian_best_fraction'],
                    description,
                )
            )
        else:
            self._finish_failure(description)

    def _execute_robot_trajectory(
            self, trajectory, purpose, stage, message, target_pose):
        if self._active is None:
            return
        self._active['phase'] = 'executing_' + purpose
        self._active['trajectory_purpose'] = purpose
        if purpose == 'approach':
            # Retreat later returns to exactly where the approach starts,
            # a configuration the arm has just occupied without collision.
            self._active['retreat_joints'] = self._arm_joint_radians()
        candidate = self._active.get('candidate')
        self._grasp_log('ARM MOVING ({}) with {}'.format(
            purpose.replace('_', ' '),
            self._grasp_kind(candidate) if candidate else 'no grasp selected',
        ))
        self._publish_active_stage(stage, message, target_pose)
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = trajectory
        future = self._execute_trajectory.send_goal_async(goal)
        token = self._active['token']
        future.add_done_callback(
            lambda done, token=token:
                self._execute_goal_response(done, token)
        )

    def _execute_goal_response(self, future, token):
        if (
                not self._active_token_valid(token)
                or not self._active['phase'].startswith('executing_')):
            return
        purpose = self._active['trajectory_purpose']
        try:
            goal_handle = future.result()
        except Exception as error:
            self._finish_failure(
                'could not execute {} trajectory: {}'.format(purpose, error)
            )
            return
        if not goal_handle.accepted:
            self._finish_failure(
                'MoveIt rejected {} trajectory execution'.format(purpose)
            )
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda done, token=token: self._execute_result(done, token)
        )

    def _execute_result(self, future, token):
        if (
                not self._active_token_valid(token)
                or not self._active['phase'].startswith('executing_')):
            return
        purpose = self._active['trajectory_purpose']
        try:
            result = future.result().result
            error_code = result.error_code.val
        except Exception as error:
            self._finish_failure(
                '{} trajectory execution failed: {}'.format(purpose, error)
            )
            return
        if error_code != MoveItErrorCodes.SUCCESS:
            self._finish_failure(
                '{} trajectory execution failed: {}'.format(
                    purpose, MOVEIT_ERROR_NAMES.get(
                        error_code, 'MoveIt error {}'.format(error_code)
                    )
                )
            )
            return
        if purpose == 'pregrasp':
            self._begin_reacquisition()
        elif purpose in ('reposition_pregrasp', 'align_pregrasp'):
            self._begin_pose_verification(purpose)
        elif purpose == 'approach':
            self._command_gripper(
                self._gripper_closed_degrees,
                'wait_close',
                GraspStatus.STAGE_CLOSING,
                'final approach complete; closing the gripper',
            )
        else:
            self._finish_success(
                GraspStatus.STAGE_RELEASED,
                'object released and the arm retreated to pre-grasp; no lift '
                'was attempted',
            )

    def _begin_pose_verification(self, purpose):
        now = self._now_seconds()
        self._active['phase'] = 'verifying_' + purpose
        self._active['pose_verification_purpose'] = purpose
        self._active['pose_verification_not_before'] = (
            now + self._post_execution_pose_settle_sec
        )
        self._active['deadline'] = (
            now + self._post_execution_pose_timeout_sec
        )
        self._active['pose_verification_error'] = None
        self._publish_active_stage(
            GraspStatus.STAGE_APPROACHING,
            'motion complete; waiting for the achieved arm pose to update',
        )

    def _verify_completed_pregrasp(self, purpose):
        try:
            transform = self._lookup_transform(
                self._planning_frame, self._tip_link
            )
            actual_pregrasp = self._point_tuple(
                transform.transform.translation
            )
            rotation = transform.transform.rotation
            achieved_orientation = (
                rotation.x, rotation.y, rotation.z, rotation.w
            )
            candidate = dict(self._active['candidate'])
            if self._fixed_approach(candidate):
                line_error = grasping.approach_line_error(
                    actual_pregrasp, candidate['pregrasp_point'],
                    candidate['approach_direction'],
                )
                limit = self._maximum_final_lateral_correction_m
                if purpose == 'align_pregrasp':
                    limit = max(
                        limit,
                        self._maximum_alignment_lateral_correction_m,
                    )
                if line_error > limit:
                    return (
                        'reached tool center is {:.3f} m off the fixed '
                        'approach line (limit {:.3f} m); no contact motion '
                        'was executed'.format(line_error, limit)
                    )
            if purpose == 'align_pregrasp':
                if candidate.get('taught'):
                    orientation_problem = self._taught_orientation_problem(
                        candidate, achieved_orientation
                    )
                    if orientation_problem is not None:
                        return (
                            'wrist alignment did not reach the taught pose: '
                            + orientation_problem
                            + '; no contact motion was executed'
                        )
                else:
                    pointing_problem = self._band_pointing_problem(
                        candidate, achieved_orientation
                    )
                    if pointing_problem is not None:
                        return (
                            'wrist alignment did not reach the band approach: '
                            + pointing_problem
                            + '; no contact motion was executed'
                        )
                    alignment_problem = self._band_alignment_problem(
                        candidate, achieved_orientation
                    )
                    if alignment_problem is not None:
                        return (
                            'wrist alignment did not level the jaw opening: '
                            + alignment_problem
                        )
        except (ValueError, TransformException) as error:
            return 'could not verify reached pre-grasp pose: {}'.format(error)

        candidate['orientation'] = achieved_orientation
        self._active['candidate'] = candidate
        self._active['grasp_pose'] = self._candidate_pose(
            candidate, 'grasp_point'
        )
        self._active['retreat_pose'] = self._candidate_pose(
            candidate, 'pregrasp_point'
        )
        self._active['pregrasp_pose'] = self._active['retreat_pose']
        self._grasp_publisher.publish(self._active['grasp_pose'])
        self._align_or_approach_from_pregrasp(
            actual_pregrasp, achieved_orientation,
            ('corrected pre-grasp reached; computing the straight Cartesian '
             'final approach' if purpose == 'reposition_pregrasp' else
             'open gripper aligned; computing the straight Cartesian final '
             'approach'),
        )
        return None

    def _command_gripper(self, angle, next_phase, stage, message):
        if self._active is None:
            return
        request = SetJoint.Request()
        request.joint_id = self._gripper_joint_id
        request.angle_degrees = float(angle)
        self._active['pending_gripper_phase'] = next_phase
        self._active['pending_gripper_target'] = float(angle)
        if (next_phase == 'wait_open_before_pregrasp'
                and self._active['open_after_laydown']):
            self._active['gripper_open_command_receipt'] = (
                self._latest_gripper_receipt
            )
        self._publish_active_stage(stage, message)
        future = self._set_joint.call_async(request)
        token = self._active['token']
        future.add_done_callback(
            lambda done, token=token:
                self._gripper_command_result(done, token)
        )

    def _gripper_command_result(self, future, token):
        if not self._active_token_valid(token):
            return
        try:
            response = future.result()
        except Exception as error:
            self._finish_failure('gripper service failed: {}'.format(error))
            return
        if not response.published:
            self._finish_failure(
                'gripper command was rejected: {}'.format(response.message)
            )
            return
        now = self._now_seconds()
        phase = self._active['pending_gripper_phase']
        self._active['phase'] = phase
        self._active['deadline'] = now + self._gripper_operation_timeout_sec
        self._active['gripper_start_angle'] = self._latest_gripper_angle
        self._active['gripper_last_angle'] = self._latest_gripper_angle
        self._active['gripper_last_change'] = now
        if phase == 'wait_open_before_pregrasp' and (
                self._active['open_after_laydown']):
            self._active['gripper_open_best_distance'] = (
                None if self._latest_gripper_angle is None
                else abs(self._latest_gripper_angle
                         - self._active['pending_gripper_target'])
            )
            self._active['gripper_open_last_progress'] = now

    def _periodic_update(self):
        if self._active is None:
            return
        phase = self._active['phase']
        now = self._now_seconds()
        if phase == 'waiting_laydown_enable':
            if now >= self._active['deadline']:
                self._start_laydown_raise()
            return
        if phase.startswith('searching_taught'):
            if now > self._active['deadline']:
                self._taught_search = None
                self._finish_failure(
                    'taught grasp search did not finish within {:.1f}s; '
                    'is move_group running?'.format(
                        self._taught_search_timeout_sec
                    )
                )
            return
        if phase == 'reacquiring':
            if now > self._active['deadline']:
                self._finish_failure(
                    'no fresh matching detection arrived within {:.1f}s'.format(
                        self._reacquire_timeout_sec
                    )
                )
            return
        if phase.startswith('verifying_'):
            if now < self._active['pose_verification_not_before']:
                return
            failure = self._verify_completed_pregrasp(
                self._active['pose_verification_purpose']
            )
            if failure is None:
                return
            self._active['pose_verification_error'] = failure
            if now >= self._active['deadline']:
                self._finish_failure(failure)
            return
        if phase in (
                'wait_open_before_pregrasp', 'wait_close',
                'wait_open_before_retreat'):
            self._update_gripper_operation(phase, now)
            return
        if phase == 'holding' and now >= self._active['deadline']:
            self._command_gripper(
                self._gripper_open_degrees,
                'wait_open_before_retreat',
                GraspStatus.STAGE_RELEASING,
                'hold complete; opening the gripper to release the object',
            )

    def _update_gripper_operation(self, phase, now):
        if now > self._active['deadline']:
            if (phase == 'wait_open_before_pregrasp'
                    and self._active['open_after_laydown']):
                self._finish_failure(
                    'gripper did not open after the lay-down raise{}; '
                    'no arm planning was started'.format(
                        ' and one feedback-triggered resend'
                        if self._active['gripper_open_retry_used'] else ''
                    )
                )
            else:
                self._finish_failure(
                    'gripper did not complete within {:.1f}s'.format(
                        self._gripper_operation_timeout_sec
                    )
                )
            return
        if self._latest_gripper_angle is None:
            return
        if now - self._latest_gripper_receipt > 1.0:
            return
        if (phase == 'wait_open_before_pregrasp'
                and self._active['open_after_laydown']
                and self._active['gripper_open_command_receipt'] is not None
                and self._latest_gripper_receipt
                <= self._active['gripper_open_command_receipt']):
            return
        current = self._latest_gripper_angle
        target = self._active['pending_gripper_target']
        if abs(current - target) <= self._gripper_tolerance_degrees:
            if phase == 'wait_open_before_pregrasp':
                self._active['open_after_laydown'] = False
                self._start_next_pregrasp_candidate()
            elif phase == 'wait_close':
                if self._require_grasp_obstruction:
                    self._finish_failure(
                        'gripper reached the fully closed position; no '
                        'obstruction was detected'
                    )
                else:
                    self._start_hold(
                        'gripper reached its commanded closed position'
                    )
            else:
                self._start_retreat()
            return
        if (phase == 'wait_open_before_pregrasp'
                and self._active['open_after_laydown']):
            distance = abs(current - target)
            best = self._active['gripper_open_best_distance']
            if (best is None or distance < best - max(
                    0.5, self._gripper_stall_delta_degrees)):
                self._active['gripper_open_best_distance'] = distance
                self._active['gripper_open_last_progress'] = now
            if (not self._active['gripper_open_retry_used']
                    and now - self._active['gripper_open_last_progress'] >= 1.5):
                self._active['gripper_open_retry_used'] = True
                self._active['phase'] = 'retrying_open_after_laydown'
                self._command_gripper(
                    target, 'wait_open_before_pregrasp',
                    GraspStatus.STAGE_OPENING,
                    'gripper feedback did not advance after the lay-down '
                    'raise; resending the opening command once',
                )
            return
        if phase != 'wait_close':
            return

        start = self._active['gripper_start_angle']
        if start is None:
            self._active['gripper_start_angle'] = current
            start = current
        previous = self._active['gripper_last_angle']
        if previous is None or abs(current - previous) > (
                self._gripper_stall_delta_degrees):
            self._active['gripper_last_angle'] = current
            self._active['gripper_last_change'] = now
            return
        start_distance = abs(start - self._gripper_closed_degrees)
        current_distance = abs(current - self._gripper_closed_degrees)
        closure = start_distance - current_distance
        stalled_for = now - self._active['gripper_last_change']
        if closure >= self._minimum_gripper_closure_degrees and (
                stalled_for >= self._gripper_stall_confirm_sec):
            self._start_hold(
                'gripper stopped {:.1f} degrees before fully closed after '
                '{:.1f} degrees of closure'.format(current_distance, closure)
            )

    def _start_hold(self, detail):
        if self._active is None:
            return
        self._active['phase'] = 'holding'
        self._active['deadline'] = (
            self._now_seconds() + self._hold_duration_sec
        )
        self._publish_active_stage(
            GraspStatus.STAGE_HOLDING,
            '{}; holding for {:.1f}s without lifting'.format(
                detail, self._hold_duration_sec
            ),
            self._active['grasp_pose'],
        )

    def _start_retreat(self):
        joints = self._active.get('retreat_joints')
        if joints is not None:
            self._request_position_plan(
                self._active['retreat_pose'],
                'retreat',
                GraspStatus.STAGE_RETREATING,
                'object released; planning a retreat to the joint angles '
                'where the final approach started',
                goal_constraints=self._joint_goal_constraints(joints),
            )
            return
        self._request_position_plan(
            self._active['retreat_pose'],
            'retreat',
            GraspStatus.STAGE_RETREATING,
            'object released; planning a position-only retreat to pre-grasp',
        )

    # ------------------------------------------------------------------
    # Taught-grasp replay
    #
    # A taught grasp is a gripper pose relative to the object. To replay it
    # anywhere the arm can reach, rotate it about the object's vertical axis
    # (nearest the taught side first) and accept the first rotation where
    # MoveIt finds collision-free IK for the contact and the pre-grasp, plus
    # a complete straight path between them. The arm is then planned to
    # those exact pre-grasp joint angles, so it arrives in the configuration
    # that was checked instead of wherever a pose-goal plan happens to end.
    # ------------------------------------------------------------------

    def _taught_yaw_range(self, object_class):
        return self._taught_class_yaw_ranges.get(
            grasping.normalize_label(object_class),
            self._taught_default_yaw_range_rad,
        )

    def _taught_yaws(self, object_class, around=0.0):
        """Rotations to try, ordered by distance from ``around``."""
        offsets = grasp_templates.yaw_offsets(
            self._taught_yaw_range(object_class), self._taught_yaw_step_rad
        )

        def separation(value):
            delta = (value - around + math.pi) % (2.0 * math.pi) - math.pi
            return abs(delta)

        return tuple(sorted(offsets, key=separation))

    def _taught_attempts(self, template, yaws):
        """(template variant, yaw) pairs: taught standoff first, then closer."""
        contact = template['contact']
        pregrasp = template['pregrasp']
        taught = grasping.distance_between(
            pregrasp['position'], contact['position']
        )
        variants = [template]
        standoff = taught - 0.02
        while standoff >= self._taught_min_pregrasp_standoff_m - 1.0e-9:
            try:
                shorter, _ = grasp_templates.straight_pregrasp(
                    contact, standoff
                )
            except ValueError:
                break
            variant = dict(template)
            variant['pregrasp'] = shorter
            variants.append(variant)
            standoff -= 0.02
        return [(variant, yaw) for variant in variants for yaw in yaws]

    def _arm_joint_radians(self):
        if self._latest_arm_angles is None or (
                self._now_seconds() - self._latest_arm_receipt > 1.0):
            return None
        return [math.radians(value) for value in self._latest_arm_angles]

    @staticmethod
    def _arm_positions(robot_state):
        names = list(robot_state.joint_state.name)
        positions = list(robot_state.joint_state.position)
        by_name = dict(zip(names, positions))
        if any(name not in by_name for name in D1_ARM_JOINT_NAMES):
            raise ValueError('IK solution lacks D1 arm joints')
        return [float(by_name[name]) for name in D1_ARM_JOINT_NAMES]

    def _joint_goal_constraints(self, joints):
        constraints = Constraints()
        constraints.name = 'taught_pregrasp_joints'
        for name, value in zip(D1_ARM_JOINT_NAMES, joints):
            joint = JointConstraint()
            joint.joint_name = name
            joint.position = float(value)
            joint.tolerance_above = self._taught_joint_goal_tolerance_rad
            joint.tolerance_below = self._taught_joint_goal_tolerance_rad
            joint.weight = 1.0
            constraints.joint_constraints.append(joint)
        return [constraints]

    def _start_taught_search(self, template, reference_for_yaw, yaws, seed,
                             done):
        """Find the first rotation with IK and a straight approach.

        ``done(result, summary)`` is called once, unless the search is
        superseded. ``result`` is None when no rotation works.
        """
        search = {
            'reference_for_yaw': reference_for_yaw,
            'attempts': self._taught_attempts(template, list(yaws)),
            'rotations': len(list(yaws)),
            'index': 0,
            'seed': None if seed is None else [float(v) for v in seed],
            'done': done,
            'checked': 0,
            'counts': {},
            'best_fraction': 0.0,
        }
        self._taught_search = search
        if not self._inverse_kinematics.wait_for_service(timeout_sec=0.0):
            self._taught_search = None
            done(None, 'IK service {} is unavailable'.format(self._ik_service))
            return
        if not self._cartesian_path.wait_for_service(timeout_sec=0.0):
            self._taught_search = None
            done(None, 'Cartesian path service {} is unavailable'.format(
                self._cartesian_path_service
            ))
            return
        self._taught_search_next(search)

    def _taught_search_count(self, search, reason):
        search['counts'][reason] = search['counts'].get(reason, 0) + 1

    def _taught_search_summary(self, search):
        parts = ['{} {}'.format(count, reason)
                 for reason, count in sorted(search['counts'].items())]
        standoffs = len(search['attempts']) // max(1, search['rotations'])
        text = 'tried {} rotation(s) about the object'.format(
            search['rotations']
        )
        if standoffs > 1:
            text += ' at {} pre-grasp distances'.format(standoffs)
        if parts:
            text += ': ' + ', '.join(parts)
        if search['best_fraction'] > 0.0:
            text += ' (best straight approach {:.0f}%)'.format(
                100.0 * search['best_fraction']
            )
        return text

    def _taught_search_finish(self, search, result):
        if self._taught_search is not search:
            return
        self._taught_search = None
        search['done'](result, self._taught_search_summary(search))

    def _taught_search_next(self, search):
        if self._taught_search is not search:
            return
        while search['index'] < len(search['attempts']):
            template, yaw = search['attempts'][search['index']]
            search['index'] += 1
            try:
                candidate = grasp_templates.candidate_from_template(
                    template, search['reference_for_yaw'](yaw), yaw
                )
            except ValueError:
                self._taught_search_count(search, 'invalid geometry')
                continue
            try:
                problem = self._candidate_problem(candidate)
            except Exception:
                # A transform lookup failing here must not crash the node.
                problem = 'reach check failed'
            if problem is not None:
                self._taught_search_count(
                    search,
                    'reach check failed' if problem == 'reach check failed'
                    else 'outside reach safeguard',
                )
                continue
            search['candidate'] = candidate
            search['checked'] += 1
            self._request_taught_ik(
                search, candidate['grasp_point'], candidate['orientation'],
                search['seed'], self._taught_contact_ik_result,
            )
            return
        self._taught_search_finish(search, None)

    def _request_taught_ik(self, search, point, orientation, seed, handler):
        request = GetPositionIK.Request()
        ik = request.ik_request
        ik.group_name = self._planning_group
        ik.ik_link_name = self._tip_link
        ik.avoid_collisions = True
        ik.pose_stamped = self._target_pose(point, orientation)
        timeout = self._taught_ik_timeout_sec
        ik.timeout = Duration(
            sec=int(timeout), nanosec=int((timeout % 1.0) * 1e9)
        )
        ik.robot_state.is_diff = True
        if seed is not None:
            ik.robot_state.joint_state.name = list(D1_ARM_JOINT_NAMES)
            ik.robot_state.joint_state.position = [float(v) for v in seed]
        future = self._inverse_kinematics.call_async(request)
        future.add_done_callback(
            lambda done, search=search: handler(done, search)
        )

    def _taught_ik_solution(self, future):
        try:
            response = future.result()
        except Exception:
            return None
        if response is None or (
                response.error_code.val != MoveItErrorCodes.SUCCESS):
            return None
        try:
            return response.solution, self._arm_positions(response.solution)
        except ValueError:
            return None

    def _taught_contact_ik_result(self, future, search):
        if self._taught_search is not search:
            return
        solution = self._taught_ik_solution(future)
        if solution is None:
            self._taught_search_count(search, 'no collision-free IK at contact')
            self._taught_search_next(search)
            return
        search['contact_joints'] = solution[1]
        candidate = search['candidate']
        # Seed the pre-grasp with the contact solution to stay in the same
        # arm configuration along the short approach.
        self._request_taught_ik(
            search, candidate['pregrasp_point'], candidate['orientation'],
            solution[1], self._taught_pregrasp_ik_result,
        )

    def _taught_pregrasp_ik_result(self, future, search):
        if self._taught_search is not search:
            return
        solution = self._taught_ik_solution(future)
        if solution is None:
            self._taught_search_count(
                search, 'no collision-free IK at pre-grasp'
            )
            self._taught_search_next(search)
            return
        search['pregrasp_state'], search['pregrasp_joints'] = solution
        candidate = search['candidate']
        request = GetCartesianPath.Request()
        request.header.stamp = self.get_clock().now().to_msg()
        request.header.frame_id = self._planning_frame
        request.start_state = copy.deepcopy(search['pregrasp_state'])
        request.start_state.is_diff = True
        request.group_name = self._planning_group
        request.link_name = self._tip_link
        request.waypoints = [copy.deepcopy(
            self._target_pose(
                candidate['grasp_point'], candidate['orientation']
            ).pose
        )]
        request.max_step = self._cartesian_step_m
        if hasattr(request, 'jump_threshold'):
            request.jump_threshold = self._cartesian_jump_threshold
        request.avoid_collisions = True
        future = self._cartesian_path.call_async(request)
        future.add_done_callback(
            lambda done, search=search:
                self._taught_cartesian_result(done, search)
        )

    def _taught_cartesian_result(self, future, search):
        if self._taught_search is not search:
            return
        try:
            response = future.result()
            ok = response.error_code.val == MoveItErrorCodes.SUCCESS
            fraction = float(response.fraction) if ok else 0.0
        except Exception:
            fraction = 0.0
        if not math.isfinite(fraction):
            fraction = 0.0
        search['best_fraction'] = max(search['best_fraction'], fraction)
        if fraction < self._minimum_cartesian_fraction:
            self._taught_search_count(search, 'incomplete straight approach')
            self._taught_search_next(search)
            return
        self._taught_search_finish(search, {
            'candidate': search['candidate'],
            'pregrasp_joints': search['pregrasp_joints'],
            'contact_joints': search['contact_joints'],
            'checked': search['checked'],
        })

    def _begin_taught_candidate(self, candidate):
        template = candidate['template']
        object_point = self._active['object_point']
        origin = self._active['approach_origin']
        self._active['phase'] = 'searching_taught'
        self._active['deadline'] = (
            self._now_seconds() + self._taught_search_timeout_sec
        )
        self._publish_active_stage(
            GraspStatus.STAGE_PLANNING,
            'checking taught grasp {} for collision-free IK and a straight '
            'approach around the object'.format(candidate['template_name']),
        )
        token = self._active['token']
        self._start_taught_search(
            template,
            lambda yaw: grasp_templates.object_reference(
                object_point, origin, yaw
            ),
            self._taught_yaws(self._active['object_class']),
            template.get('contact_joints'),
            lambda result, summary, token=token:
                self._taught_candidate_found(result, summary, token),
        )

    def _taught_candidate_found(self, result, summary, token):
        if (not self._active_token_valid(token)
                or self._active['phase'] != 'searching_taught'):
            return
        name = self._active['candidate']['template_name']
        if result is None:
            self._try_next_candidate(
                'no reachable rotation of taught grasp {}: {}'.format(
                    name, summary
                )
            )
            return
        selected = self._active['candidate']
        candidate = dict(result['candidate'])
        candidate['band_index'] = selected.get('band_index', 0)
        candidate['band_count'] = selected.get('band_count', 0)
        candidate['taught_pregrasp_joints'] = result['pregrasp_joints']
        self._active['candidate'] = candidate
        self._active['target_point'] = candidate['band_target_point']
        self._active['grasp_band_width'] = None
        pose = self._candidate_pose(candidate, 'pregrasp_point')
        self._active['pregrasp_pose'] = pose
        self._active['phase'] = 'planning_pregrasp'
        self._pregrasp_publisher.publish(pose)
        self._grasp_log('FOUND {}: reachable after checking {} option(s); '
                        'planning to its checked pre-grasp joint angles'.format(
                            self._grasp_kind(candidate), result['checked']))
        self._publish_active_stage(
            GraspStatus.STAGE_PLANNING,
            'taught grasp {} is reachable (rotated {:+.0f} deg about the '
            'object); MoveIt is planning to its pre-grasp joint angles'.format(
                name, math.degrees(candidate['yaw_offset'])
            ),
            pose,
        )
        future = self._move_group.send_goal_async(
            self._move_group_goal(
                pose,
                goal_constraints=self._joint_goal_constraints(
                    result['pregrasp_joints']
                ),
            ),
            feedback_callback=lambda message, token=token:
                self._move_group_feedback(message, token),
        )
        future.add_done_callback(
            lambda done, token=token:
                self._move_group_goal_response(done, token)
        )

    def _begin_taught_reposition(self, reason):
        """Re-check a taught grasp from the reached pose and move to it."""
        candidate = self._active['candidate']
        object_point = self._active['object_point']
        origin = self._active['approach_origin']
        self._active['phase'] = 'searching_taught_reposition'
        self._active['deadline'] = (
            self._now_seconds() + self._taught_search_timeout_sec
        )
        self._grasp_log('{}; re-checking {} from the reached pose'.format(
            reason, self._grasp_kind(candidate)))
        self._publish_active_stage(
            GraspStatus.STAGE_APPROACHING,
            '{}; re-checking the taught grasp from the reached pose'.format(
                reason
            ),
        )
        token = self._active['token']
        self._start_taught_search(
            candidate['template'],
            lambda yaw: grasp_templates.object_reference(
                object_point, origin, yaw
            ),
            self._taught_yaws(
                self._active['object_class'],
                float(candidate.get('yaw_offset', 0.0)),
            ),
            self._arm_joint_radians(),
            lambda result, summary, token=token:
                self._taught_reposition_found(result, summary, token),
        )

    def _taught_reposition_found(self, result, summary, token):
        if (not self._active_token_valid(token)
                or self._active['phase'] != 'searching_taught_reposition'):
            return
        if result is None:
            self._finish_failure(
                'taught grasp is no longer reachable from the reached '
                'pre-grasp: {}; no contact motion was executed'.format(summary)
            )
            return
        previous = self._active['candidate']
        candidate = dict(result['candidate'])
        candidate['band_index'] = previous.get('band_index', 0)
        candidate['band_count'] = previous.get('band_count', 0)
        candidate['taught_pregrasp_joints'] = result['pregrasp_joints']
        self._active['candidate'] = candidate
        self._active['target_point'] = candidate['band_target_point']
        retreat_pose = self._candidate_pose(candidate, 'pregrasp_point')
        self._active['pregrasp_pose'] = retreat_pose
        self._active['retreat_pose'] = retreat_pose
        self._active['grasp_pose'] = self._candidate_pose(
            candidate, 'grasp_point'
        )
        self._pregrasp_publisher.publish(retreat_pose)
        self._grasp_publisher.publish(self._active['grasp_pose'])
        self._request_position_plan(
            retreat_pose,
            'reposition_pregrasp',
            GraspStatus.STAGE_APPROACHING,
            'moving to the re-checked pre-grasp joint angles of {}'.format(
                self._grasp_kind(candidate)
            ),
            goal_constraints=self._joint_goal_constraints(
                result['pregrasp_joints']
            ),
        )

    @staticmethod
    def _grasp_kind(candidate):
        if not candidate:
            return 'no candidate started'
        if candidate.get('taught'):
            yaw = float(candidate.get('yaw_offset', 0.0))
            details = []
            if abs(yaw) >= 1.0e-6:
                details.append('rotated {:+.0f} deg about the object'.format(
                    math.degrees(yaw)
                ))
            standoff = grasping.distance_between(
                candidate['pregrasp_point'], candidate['grasp_point']
            )
            details.append('approach from {:.0f} cm'.format(100.0 * standoff))
            return "TAUGHT grasp '{}' ({})".format(
                candidate.get('template_name'), '; '.join(details)
            )
        if candidate.get('band_width') is not None:
            where = 'band {} of {}'.format(
                candidate.get('band_index'), candidate.get('band_count')
            )
        else:
            where = 'object center'
        return 'HEURISTIC grasp ({}, yaw {:+.2f} rad)'.format(
            where, float(candidate.get('yaw_offset', 0.0))
        )

    def _grasp_log(self, text, level='info', request_id=None):
        """One consistent, greppable line per grasp event: [GRASP] ..."""
        if request_id is None:
            request_id = (
                self._active['command'].request_id.strip()
                if self._active is not None else '-'
            )
        line = '[GRASP] {} | {}'.format(request_id or '<empty>', text)
        logger = self.get_logger()
        # Foxy rclpy ties each logging call site to one severity and
        # raises if it changes, so every severity needs its own line.
        if level == 'error':
            logger.error(line)
        elif level == 'warn':
            logger.warning(line)
        else:
            logger.info(line)
        if not self._grasp_log_path:
            return
        try:
            path = os.path.expanduser(self._grasp_log_path)
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            with open(path, 'a', encoding='utf-8') as stream:
                stream.write('{} {:<5} {}\n'.format(
                    time.strftime('%Y-%m-%d %H:%M:%S'), level.upper(), line
                ))
        except OSError as error:
            logger.warning('could not write grasp log {}: {}'.format(
                self._grasp_log_path, error
            ))
            self._grasp_log_path = ''

    def _finish_success(self, stage, message):
        if self._active is None:
            return
        command = self._active['command']
        self._grasp_log('RESULT: SUCCESS with {}: {}'.format(
            self._grasp_kind(self._active.get('candidate')), message
        ))
        self._publish_status(
            command, stage, True, True, message,
            self._active['object_id'], self._active['target_pose'],
        )
        self.get_logger().info(
            'Grasp request {} completed: {}'.format(
                command.request_id, message
            )
        )
        self._active = None

    def _finish_failure(self, message):
        if self._active is None:
            return
        self._taught_search = None
        active = self._active
        self._grasp_log('RESULT: FAILED with {}: {}'.format(
            self._grasp_kind(active.get('candidate')), message
        ), 'error')
        self._publish_status(
            active['command'], GraspStatus.STAGE_FAILED, True, False,
            message, active['object_id'], active.get('target_pose'),
        )
        # If contact may have occurred, issue a best-effort release. Do not
        # delay or hide the original failure if this secondary call fails.
        if self._execution_enabled and active.get('phase') in (
                'wait_close', 'holding', 'wait_open_before_retreat',
                'computing_retreat', 'executing_retreat'):
            request = SetJoint.Request()
            request.joint_id = self._gripper_joint_id
            request.angle_degrees = self._gripper_open_degrees
            self._set_joint.call_async(request)
        self._active = None


def main(args=None):
    """Run the D1 semantic grasp coordinator."""
    rclpy.init(args=args)
    node = D1GraspCoordinator()
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
