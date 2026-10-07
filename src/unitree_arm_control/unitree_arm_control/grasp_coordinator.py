"""Coordinate guarded semantic-object grasp-and-release motions through MoveIt."""

from collections import deque
import copy
import math

from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseStamped
from intel_realsense_interfaces.msg import GroundedDetectionArray
from intel_realsense_interfaces.msg import SemanticMap
from intel_realsense_interfaces.msg import SemanticObject
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.msg import OrientationConstraint
from moveit_msgs.msg import PositionConstraint
from moveit_msgs.srv import GetCartesianPath
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
from unitree_arm.msg import GraspCommand
from unitree_arm.msg import GraspStatus
from unitree_arm.msg import JointAngles
from unitree_arm.srv import SetJoint

from unitree_arm_control import grasping


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

class D1GraspCoordinator(Node):
    """Run a guarded pre-grasp, grasp, hold, release, and retreat sequence."""

    def __init__(self):
        super().__init__('d1_grasp_coordinator')
        self._declare_parameters()
        self._load_parameters()
        self._validate_parameters()

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
        self._set_joint = self.create_client(
            SetJoint, self._gripper_service
        )
        self._latest_map = None
        self._latest_detections = None
        self._latest_detection_receipt = 0.0
        self._latest_gripper_angle = None
        self._latest_gripper_receipt = 0.0
        self._active = None
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
            'gripper_service': '/d1_arm_controller/set_joint',
            'joint_angles_topic': '/d1_arm_controller/joint_angles',
            'planning_group': 'd1_arm',
            'planning_frame': 'base_link',
            'camera_frame': 'camera_link',
            'reach_reference_frame': 'd1_base_link',
            'tip_link': 'd1_gripper_center',
            'planner_id': '',
        }
        for name, default in string_parameters.items():
            self.declare_parameter(name, default)

        self.declare_parameter('execution_enabled', False)
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
        self.declare_parameter('grasp_orientation_tolerance_rad', 0.35)

        self.declare_parameter('planning_time_sec', 5.0)
        self.declare_parameter('planning_attempts', 5)
        self.declare_parameter('velocity_scaling', 0.15)
        self.declare_parameter('acceleration_scaling', 0.10)
        self.declare_parameter('fresh_detection_age_sec', 5.0)
        self.declare_parameter('reacquire_timeout_sec', 10.0)
        self.declare_parameter('reacquire_match_distance_m', 0.25)
        self.declare_parameter('maximum_reacquire_correction_m', 0.08)
        self.declare_parameter('maximum_final_lateral_correction_m', 0.015)
        self.declare_parameter('approach_corridor_radius_m', 0.015)

        # The final contact motion is a single short Cartesian segment.
        self.declare_parameter('cartesian_step_m', 0.005)
        self.declare_parameter('minimum_cartesian_fraction', 0.95)
        self.declare_parameter('cartesian_jump_threshold', 2.0)
        self.declare_parameter('cartesian_velocity_scaling', 0.05)
        self.declare_parameter('cartesian_acceleration_scaling', 0.05)

        self.declare_parameter('gripper_joint_id', 6)
        self.declare_parameter('gripper_open_degrees', 45.0)
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
            'joint_angles_topic', 'planning_group', 'planning_frame',
            'camera_frame',
            'reach_reference_frame', 'tip_link', 'planner_id',
        )
        for name in string_names:
            setattr(self, '_' + name, str(self._parameter(name)).strip())

        bool_names = ('execution_enabled', 'require_grasp_obstruction')
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
            'grasp_orientation_tolerance_rad',
            'planning_time_sec', 'velocity_scaling',
            'acceleration_scaling', 'fresh_detection_age_sec',
            'reacquire_timeout_sec', 'reacquire_match_distance_m',
            'maximum_reacquire_correction_m',
            'maximum_final_lateral_correction_m',
            'approach_corridor_radius_m',
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
            'joint_angles_topic', 'planning_group', 'planning_frame',
            'camera_frame',
            'reach_reference_frame', 'tip_link',
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
        if not 0.0 < self._grasp_orientation_tolerance_rad <= math.pi:
            raise RuntimeError(
                'grasp_orientation_tolerance_rad must be in (0, pi]'
            )
        if self._planning_time <= 0.0 or self._planning_attempts < 1:
            raise RuntimeError('planning limits are invalid')
        for name in ('velocity_scaling', 'acceleration_scaling'):
            value = getattr(self, '_' + name)
            if not 0.0 < value <= 1.0:
                raise RuntimeError('{} must be in (0, 1]'.format(name))
        if self._fresh_detection_age_sec <= 0.0:
            raise RuntimeError('fresh_detection_age_sec must be positive')
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
        if self._approach_corridor_radius_m <= 0.0:
            raise RuntimeError('approach_corridor_radius_m must be positive')
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
        if len(message.angle_degrees) <= self._gripper_joint_id:
            return
        values = tuple(float(value) for value in message.angle_degrees)
        if not all(math.isfinite(value) for value in values):
            return
        self._latest_gripper_angle = values[self._gripper_joint_id]
        self._latest_gripper_receipt = self._now_seconds()

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
        self.get_logger().warning(
            'Rejected grasp request {}: {}'.format(
                command.request_id.strip() or '<empty>', message
            )
        )
        self._publish_status(
            command, GraspStatus.STAGE_REJECTED, True, False, message
        )

    def _command_callback(self, command):
        request_id = command.request_id.strip()
        object_class = grasping.normalize_label(command.object_class)
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
        if self._execution_enabled:
            missing = []
            if not self._execute_trajectory.wait_for_server(timeout_sec=0.0):
                missing.append(self._execute_trajectory_action)
            if not self._cartesian_path.wait_for_service(timeout_sec=0.0):
                missing.append(self._cartesian_path_service)
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
            selected, object_point, approach_origin, candidates = (
                self._select_object_and_candidates(objects, object_class)
            )
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
            'approach_origin': approach_origin,
            'candidates': candidates,
            'candidate_index': 0,
            'candidate_failures': [],
            'candidate': None,
            'pregrasp_pose': first_pose,
            'grasp_pose': None,
            'retreat_pose': None,
            'target_pose': first_pose,
            'last_stage': None,
            'phase': 'accepted',
            'deadline': 0.0,
            'required_detection_receipt': 0.0,
            'gripper_start_angle': None,
            'gripper_last_angle': None,
            'gripper_last_change': 0.0,
        }
        self._pregrasp_publisher.publish(first_pose)
        self._publish_active_stage(
            GraspStatus.STAGE_ACCEPTED,
            'selected {} with {} safeguard-compatible approach candidate(s); '
            'forward grasp depth is {:.3f} m'.format(
                selected.object_id,
                len(candidates),
                grasping.forward_grasp_depth_for_class(
                    object_class,
                    self._default_forward_grasp_depth_offset,
                    self._class_forward_grasp_depth_offsets,
                ),
            ),
            first_pose,
        )
        if self._execution_enabled:
            self._command_gripper(
                self._gripper_open_degrees,
                'wait_open_before_pregrasp',
                GraspStatus.STAGE_OPENING,
                'opening gripper before planning the pre-grasp',
            )
        else:
            self._start_next_pregrasp_candidate()

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
        tip_transform = self._tf_buffer.lookup_transform(
            self._planning_frame, self._tip_link, Time()
        )
        approach_origin = self._point_tuple(
            tip_transform.transform.translation
        )
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
            depth_direction = tuple(
                value - origin
                for value, origin in zip(object_point, camera_origin)
            )
            generated = []
            try:
                generated = list(grasping.generate_approach_candidates(
                    object_point,
                    approach_origin,
                    self._approach_yaw_offsets,
                    self._approach_distance,
                    self._grasp_center_offset,
                    0.0,
                    forward_grasp_depth,
                    depth_direction,
                    grasp_height_offset,
                    horizontal_depth,
                ))
            except ValueError as error:
                failures.append(str(error))
                continue
            valid = []
            for candidate in generated:
                problem = self._candidate_problem(candidate)
                if problem is None:
                    valid.append(candidate)
                else:
                    failures.append(problem)
            if valid:
                viable.append((
                    item, object_point, approach_origin, tuple(valid),
                    grasping.distance_between(object_point, arm_origin),
                ))
        if not viable:
            detail = failures[0] if failures else 'no finite object position'
            raise RuntimeError(
                'no approach candidate satisfies reach and height safeguards: '
                + detail
            )
        selected = min(viable, key=lambda value: value[4])
        return selected[0], selected[1], selected[2], selected[3]

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
                'no approach candidate produced a valid orientation-tolerant '
                'pre-grasp plan' + detail
            )
            return
        candidate = candidates[index]
        pose = self._candidate_pose(candidate, 'pregrasp_point')
        self._active['candidate'] = candidate
        self._active['pregrasp_pose'] = pose
        self._active['phase'] = 'planning_pregrasp'
        self._pregrasp_publisher.publish(pose)
        self._publish_active_stage(
            GraspStatus.STAGE_PLANNING,
            'MoveIt is planning pre-grasp candidate {} of {} with orientation '
            'tolerance at [{:.3f}, {:.3f}, {:.3f}] m '
            '(yaw offset {:.3f} rad)'.format(
                index + 1, len(candidates),
                candidate['pregrasp_point'][0],
                candidate['pregrasp_point'][1],
                candidate['pregrasp_point'][2],
                candidate['yaw_offset'],
            ),
            pose,
        )
        future = self._move_group.send_goal_async(
            self._move_group_goal(
                pose,
                orientation_tolerance=self._grasp_orientation_tolerance_rad,
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
            orientation_tolerance=None, path_constraints=None):
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

        constraints = Constraints()
        constraints.name = (
            'semantic_pose_target'
            if orientation_tolerance is not None
            else 'semantic_position_target'
        )
        constraints.position_constraints = [position]
        if orientation_tolerance is not None:
            orientation = OrientationConstraint()
            orientation.header = copy.deepcopy(target_pose.header)
            orientation.link_name = self._tip_link
            orientation.orientation = copy.deepcopy(
                target_pose.pose.orientation
            )
            orientation.absolute_x_axis_tolerance = orientation_tolerance
            orientation.absolute_y_axis_tolerance = orientation_tolerance
            orientation.absolute_z_axis_tolerance = orientation_tolerance
            orientation.weight = 1.0
            constraints.orientation_constraints = [orientation]

        goal = MoveGroup.Goal()
        goal.request.start_state.is_diff = True
        goal.request.goal_constraints = [constraints]
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

        if not self._execution_enabled:
            self._finish_success(
                GraspStatus.STAGE_PLAN_READY,
                'pre-grasp plan succeeded; execution_enabled is false',
            )
            return

        trajectory = result.planned_trajectory
        if not trajectory.joint_trajectory.points:
            self._try_next_candidate('MoveIt returned an empty trajectory')
            return
        self._execute_robot_trajectory(
            trajectory,
            'pregrasp',
            GraspStatus.STAGE_EXECUTING,
            'executing orientation-tolerant pre-grasp candidate {}'.format(
                self._active['candidate_index'] + 1
            ),
            self._active['pregrasp_pose'],
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
        self._active.setdefault('candidate_failures', []).append(
            '{}: {}'.format(
                self._active['candidate_index'] + 1, reason
            )
        )
        self.get_logger().warning(
            'Approach candidate {} failed: {}'.format(
                self._active['candidate_index'] + 1, reason
            )
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
            if message.map_transform_available and detection.has_map_position:
                source_point = self._point_tuple(detection.map_position)
                source_frame = message.map_frame.strip()
            elif detection.has_camera_position:
                source_point = self._point_tuple(detection.camera_position)
                source_frame = message.camera_frame.strip()
            else:
                continue
            if not source_frame:
                source_frame = message.header.frame_id.strip()
            if not source_frame:
                continue
            try:
                transform = self._lookup_transform(
                    self._planning_frame, source_frame
                )
            except Exception as error:
                self.get_logger().warning(
                    'Could not transform fresh detection from {}: {}'.format(
                        source_frame, error
                    )
                )
                continue
            point = self._transform_point(source_point, transform)
            separation = grasping.distance_between(
                point, self._active['object_point']
            )
            if separation <= self._reacquire_match_distance_m:
                matches.append((separation, point, detection))
        if not matches:
            return

        separation, object_point, _ = min(matches, key=lambda item: item[0])
        if separation > self._maximum_reacquire_correction_m:
            self._finish_failure(
                'fresh detection moved {:.3f} m from the mapped point, above '
                'the {:.3f} m correction safeguard'.format(
                    separation, self._maximum_reacquire_correction_m
                )
            )
            return
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
            for value, origin in zip(object_point, camera_origin)
        )
        candidate = grasping.retarget_approach_candidate(
            selected, object_point, self._approach_distance,
            self._grasp_center_offset,
            self._active['forward_grasp_depth_offset'],
            depth_direction,
            self._active['grasp_height_offset'],
            self._active['horizontal_depth'],
        )
        problem = self._candidate_problem(candidate)
        if problem is not None:
            self._finish_failure('reacquired target is unsafe: ' + problem)
            return
        old_pregrasp = tuple(
            float(value) for value in selected['pregrasp_point']
        )
        new_pregrasp = tuple(
            float(value) for value in candidate['pregrasp_point']
        )
        correction = tuple(
            new_value - old_value
            for new_value, old_value in zip(new_pregrasp, old_pregrasp)
        )
        approach = tuple(
            float(value) for value in candidate['approach_direction']
        )
        approach_norm = math.sqrt(sum(value * value for value in approach))
        if approach_norm <= 1e-9:
            self._finish_failure(
                'reacquired approach direction has near-zero magnitude'
            )
            return
        approach_axis = tuple(value / approach_norm for value in approach)
        parallel_amount = sum(
            delta * axis for delta, axis in zip(correction, approach_axis)
        )
        lateral = tuple(
            delta - parallel_amount * axis
            for delta, axis in zip(correction, approach_axis)
        )
        lateral_correction = math.sqrt(
            sum(value * value for value in lateral)
        )

        # The pre-grasp MoveGroup goal permits an orientation tolerance. Its
        # achieved orientation can therefore differ from the generated ideal
        # quaternion. GetCartesianPath accepts exact poses rather than an
        # orientation tolerance, so retaining the generated quaternion here
        # would turn the final approach into a translation plus a rotation and
        # can make an otherwise valid straight approach return a low fraction.
        try:
            achieved_orientation = self._current_tip_orientation()
        except Exception as error:
            self.get_logger().warning(
                'Could not read the achieved pre-grasp orientation: {}'.format(
                    error
                )
            )
            return
        candidate = dict(candidate)
        candidate['orientation'] = achieved_orientation

        grasp_pose = self._candidate_pose(candidate, 'grasp_point')
        retreat_pose = self._candidate_pose(candidate, 'pregrasp_point')
        self._active['object_point'] = object_point
        self._active['candidate'] = candidate
        self._active['pregrasp_pose'] = retreat_pose
        self._active['grasp_pose'] = grasp_pose
        self._active['retreat_pose'] = retreat_pose
        self._grasp_publisher.publish(grasp_pose)

        if lateral_correction > self._maximum_final_lateral_correction_m:
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

        self._begin_final_approach(
            grasp_pose, 'approach', GraspStatus.STAGE_APPROACHING,
            'fresh target acquired; computing the short Cartesian final approach',
        )

    def _begin_final_approach(self, pose, purpose, stage, message):
        orientation = pose.pose.orientation
        self._active['cartesian_orientations'] = (
            grasping.approach_orientation_candidates(
                (orientation.x, orientation.y, orientation.z, orientation.w),
                self._grasp_orientation_tolerance_rad,
            )
        )
        self._active['cartesian_orientation_index'] = 0
        self._active['cartesian_best_fraction'] = 0.0
        self._request_cartesian_path(pose, purpose, stage, message)

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
        request.start_state.is_diff = True
        request.group_name = self._planning_group
        request.link_name = self._tip_link
        request.waypoints = [copy.deepcopy(pose.pose)]
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

        accepted_pose = self._active['cartesian_probe_pose']
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
            'approach',
            GraspStatus.STAGE_APPROACHING,
            '{}; checking nearby wrist orientation {} of {} along the same '
            'straight approach'.format(reason, index + 1, len(orientations)),
        )
        return True

    def _cartesian_path_failure(self, purpose, description):
        if purpose == 'approach':
            if self._try_next_cartesian_orientation(description):
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
            self, pose, purpose, stage, message, path_constraints=None):
        if self._active is None:
            return
        self._active['phase'] = 'planning_' + purpose
        self._active['position_plan_purpose'] = purpose
        self._publish_active_stage(stage, message, pose)
        tolerance = (
            self._grasp_position_tolerance_m
            if purpose == 'approach'
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
        is_approach_motion = purpose in ('approach', 'reposition_pregrasp')
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
        elif purpose == 'reposition_pregrasp':
            try:
                achieved_orientation = self._current_tip_orientation()
            except Exception as error:
                self._finish_failure(
                    'could not read corrected pre-grasp orientation: {}'.format(
                        error
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
            self._begin_final_approach(
                self._active['grasp_pose'],
                'approach',
                GraspStatus.STAGE_APPROACHING,
                'corrected pre-grasp reached; computing the short Cartesian '
                'final approach while preserving achieved orientation',
            )
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

    def _command_gripper(self, angle, next_phase, stage, message):
        if self._active is None:
            return
        request = SetJoint.Request()
        request.joint_id = self._gripper_joint_id
        request.angle_degrees = float(angle)
        self._active['pending_gripper_phase'] = next_phase
        self._active['pending_gripper_target'] = float(angle)
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

    def _periodic_update(self):
        if self._active is None:
            return
        phase = self._active['phase']
        now = self._now_seconds()
        if phase == 'reacquiring':
            if now > self._active['deadline']:
                self._finish_failure(
                    'no fresh matching detection arrived within {:.1f}s'.format(
                        self._reacquire_timeout_sec
                    )
                )
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
        current = self._latest_gripper_angle
        target = self._active['pending_gripper_target']
        if abs(current - target) <= self._gripper_tolerance_degrees:
            if phase == 'wait_open_before_pregrasp':
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
        self._request_position_plan(
            self._active['retreat_pose'],
            'retreat',
            GraspStatus.STAGE_RETREATING,
            'object released; planning a position-only retreat to pre-grasp',
        )

    def _finish_success(self, stage, message):
        if self._active is None:
            return
        command = self._active['command']
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
        active = self._active
        self.get_logger().error(
            'Grasp request {} failed: {}'.format(
                active['command'].request_id, message
            )
        )
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
