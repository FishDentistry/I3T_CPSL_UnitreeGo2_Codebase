"""Coordinate guarded semantic-object pre-grasp motions through MoveIt."""

from collections import deque
import math

from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseStamped
from intel_realsense_interfaces.msg import SemanticMap
from intel_realsense_interfaces.msg import SemanticObject
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.msg import OrientationConstraint
from moveit_msgs.msg import PositionConstraint
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
    MoveItErrorCodes.START_STATE_IN_COLLISION: (
        'start state is in collision'
    ),
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
    MoveItErrorCodes.INVALID_GOAL_CONSTRAINTS: (
        'invalid goal constraints'
    ),
    MoveItErrorCodes.INVALID_ROBOT_STATE: 'invalid robot state',
    MoveItErrorCodes.INVALID_LINK_NAME: 'invalid end-effector link',
    MoveItErrorCodes.FRAME_TRANSFORM_FAILURE: 'frame transform failed',
    MoveItErrorCodes.ROBOT_STATE_STALE: 'robot state is stale',
    MoveItErrorCodes.COMMUNICATION_FAILURE: 'communication failure',
    MoveItErrorCodes.NO_IK_SOLUTION: 'no inverse-kinematics solution',
}


class D1GraspCoordinator(Node):
    """Convert semantic grasp commands into guarded MoveIt requests."""

    def __init__(self):
        super().__init__('d1_grasp_coordinator')

        self.declare_parameter('semantic_map_topic', '/semantic_map')
        self.declare_parameter('command_topic', '/d1_grasp/command')
        self.declare_parameter('status_topic', '/d1_grasp/status')
        self.declare_parameter(
            'pregrasp_pose_topic', '/d1_grasp/pregrasp_pose'
        )
        self.declare_parameter('move_group_action', '/move_action')
        self.declare_parameter('planning_group', 'd1_arm')
        self.declare_parameter('planning_frame', 'base_link')
        self.declare_parameter('reach_reference_frame', 'd1_base_link')
        self.declare_parameter('tip_link', 'd1_gripper_center')
        self.declare_parameter('planner_id', '')

        # Motion is deliberately opt-in during the initial integration phase.
        self.declare_parameter('execution_enabled', False)
        self.declare_parameter('minimum_confidence', 0.65)
        self.declare_parameter('minimum_observations', 3)
        self.declare_parameter('maximum_object_age_sec', 20.0)
        self.declare_parameter('minimum_reach_m', 0.10)
        self.declare_parameter('maximum_reach_m', 0.67)
        self.declare_parameter('minimum_target_z_m', -0.10)
        self.declare_parameter('maximum_target_z_m', 0.80)

        self.declare_parameter('pregrasp_offset_x_m', 0.0)
        self.declare_parameter('pregrasp_offset_y_m', 0.0)
        self.declare_parameter('pregrasp_offset_z_m', 0.15)
        self.declare_parameter('preserve_current_orientation', True)
        self.declare_parameter('pregrasp_roll_rad', math.pi)
        self.declare_parameter('pregrasp_pitch_rad', 0.0)
        self.declare_parameter('pregrasp_yaw_rad', 0.0)
        self.declare_parameter('position_tolerance_m', 0.02)
        self.declare_parameter('orientation_tolerance_rad', 0.15)

        self.declare_parameter('planning_time_sec', 5.0)
        self.declare_parameter('planning_attempts', 5)
        self.declare_parameter('velocity_scaling', 0.15)
        self.declare_parameter('acceleration_scaling', 0.10)

        self._load_parameters()
        self._validate_parameters()

        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        pose_qos = QoSProfile(
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
        self._pose_publisher = self.create_publisher(
            PoseStamped, self._pregrasp_pose_topic, pose_qos
        )
        self.create_subscription(
            SemanticMap,
            self._semantic_map_topic,
            self._map_callback,
            map_qos,
        )
        self.create_subscription(
            GraspCommand,
            self._command_topic,
            self._command_callback,
            10,
        )

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._move_group = ActionClient(
            self, MoveGroup, self._move_group_action
        )
        self._latest_map = None
        self._active = None
        self._recent_request_ids = deque(maxlen=100)

        mode = 'pre-grasp execution' if self._execution_enabled else (
            'plan-only'
        )
        self.get_logger().info(
            'D1 grasp coordinator ready on {} in {} mode; contact and '
            'gripper closure remain disabled'.format(
                self._command_topic, mode
            )
        )
        if self._execution_enabled:
            self.get_logger().warning(
                'Guarded pre-grasp execution is enabled. Support surfaces '
                'and object geometry are not yet inserted into the MoveIt '
                'planning scene; use only a manually verified clear volume.'
            )

    def _parameter(self, name):
        return self.get_parameter(name).value

    def _load_parameters(self):
        self._semantic_map_topic = str(
            self._parameter('semantic_map_topic')
        ).strip()
        self._command_topic = str(self._parameter('command_topic')).strip()
        self._status_topic = str(self._parameter('status_topic')).strip()
        self._pregrasp_pose_topic = str(
            self._parameter('pregrasp_pose_topic')
        ).strip()
        self._move_group_action = str(
            self._parameter('move_group_action')
        ).strip()
        self._planning_group = str(
            self._parameter('planning_group')
        ).strip()
        self._planning_frame = str(
            self._parameter('planning_frame')
        ).strip()
        self._reach_reference_frame = str(
            self._parameter('reach_reference_frame')
        ).strip()
        self._tip_link = str(self._parameter('tip_link')).strip()
        self._planner_id = str(self._parameter('planner_id')).strip()
        self._execution_enabled = bool(
            self._parameter('execution_enabled')
        )
        self._minimum_confidence = float(
            self._parameter('minimum_confidence')
        )
        self._minimum_observations = int(
            self._parameter('minimum_observations')
        )
        self._maximum_object_age = float(
            self._parameter('maximum_object_age_sec')
        )
        self._minimum_reach = float(self._parameter('minimum_reach_m'))
        self._maximum_reach = float(self._parameter('maximum_reach_m'))
        self._minimum_target_z = float(
            self._parameter('minimum_target_z_m')
        )
        self._maximum_target_z = float(
            self._parameter('maximum_target_z_m')
        )
        self._pregrasp_offset = (
            float(self._parameter('pregrasp_offset_x_m')),
            float(self._parameter('pregrasp_offset_y_m')),
            float(self._parameter('pregrasp_offset_z_m')),
        )
        self._preserve_current_orientation = bool(
            self._parameter('preserve_current_orientation')
        )
        self._pregrasp_rpy = (
            float(self._parameter('pregrasp_roll_rad')),
            float(self._parameter('pregrasp_pitch_rad')),
            float(self._parameter('pregrasp_yaw_rad')),
        )
        self._position_tolerance = float(
            self._parameter('position_tolerance_m')
        )
        self._orientation_tolerance = float(
            self._parameter('orientation_tolerance_rad')
        )
        self._planning_time = float(self._parameter('planning_time_sec'))
        self._planning_attempts = int(
            self._parameter('planning_attempts')
        )
        self._velocity_scaling = float(
            self._parameter('velocity_scaling')
        )
        self._acceleration_scaling = float(
            self._parameter('acceleration_scaling')
        )

    def _validate_parameters(self):
        required_strings = {
            'semantic_map_topic': self._semantic_map_topic,
            'command_topic': self._command_topic,
            'status_topic': self._status_topic,
            'pregrasp_pose_topic': self._pregrasp_pose_topic,
            'move_group_action': self._move_group_action,
            'planning_group': self._planning_group,
            'planning_frame': self._planning_frame,
            'reach_reference_frame': self._reach_reference_frame,
            'tip_link': self._tip_link,
        }
        for name, value in required_strings.items():
            if not value:
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
        if self._position_tolerance <= 0.0:
            raise RuntimeError('position_tolerance_m must be positive')
        if self._orientation_tolerance <= 0.0:
            raise RuntimeError('orientation_tolerance_rad must be positive')
        if self._planning_time <= 0.0 or self._planning_attempts < 1:
            raise RuntimeError('planning limits are invalid')
        for name, value in (
                ('velocity_scaling', self._velocity_scaling),
                ('acceleration_scaling', self._acceleration_scaling)):
            if not 0.0 < value <= 1.0:
                raise RuntimeError('{} must be in (0, 1]'.format(name))

    def _map_callback(self, message):
        self._latest_map = message

    def _now_seconds(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _empty_pose(self):
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = self._planning_frame
        pose.pose.orientation.w = 1.0
        return pose

    def _publish_status(
            self,
            command,
            stage,
            terminal,
            success,
            message,
            object_id='',
            target_pose=None):
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

    def _reject(self, command, message):
        self.get_logger().warning(
            'Rejected grasp request {}: {}'.format(
                command.request_id.strip() or '<empty>', message
            )
        )
        self._publish_status(
            command,
            GraspStatus.STAGE_REJECTED,
            True,
            False,
            message,
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

        candidates = grasping.eligible_objects(
            self._latest_map.objects,
            object_class,
            command.object_id,
            self._now_seconds(),
            self._minimum_confidence,
            self._minimum_observations,
            self._maximum_object_age,
            SemanticObject.STATUS_ACTIVE,
        )
        if not candidates:
            self._reject(
                command,
                self._selection_failure_message(command),
            )
            return

        try:
            transformed = self._transform_candidates(candidates)
        except Exception as error:
            self._reject(
                command,
                'could not transform semantic-map position into {}: {}'.format(
                    self._planning_frame, error
                ),
            )
            return

        (
            selected,
            object_point,
            target_point,
            target_orientation,
            reach_point,
        ) = min(
            transformed,
            key=lambda item: grasping.distance_from_origin(item[4]),
        )
        problem = self._target_problem(target_point, reach_point)
        if problem is not None:
            self._reject(command, problem)
            return

        target_pose = self._target_pose(
            target_point, target_orientation
        )
        self._pose_publisher.publish(target_pose)
        self._recent_request_ids.append(request_id)
        self._active = {
            'command': command,
            'object_id': selected.object_id,
            'target_pose': target_pose,
            'last_stage': None,
        }
        self._publish_status(
            command,
            GraspStatus.STAGE_ACCEPTED,
            False,
            False,
            'selected {} and generated a guarded pre-grasp pose'.format(
                selected.object_id
            ),
            selected.object_id,
            target_pose,
        )
        self._publish_active_stage(
            GraspStatus.STAGE_PLANNING, 'MoveIt is planning to pre-grasp'
        )
        future = self._move_group.send_goal_async(
            self._move_group_goal(target_pose),
            feedback_callback=self._move_group_feedback,
        )
        future.add_done_callback(self._move_group_goal_response)

    def _selection_failure_message(self, command):
        objects = list(self._latest_map.objects)
        requested_class = grasping.normalize_label(command.object_class)
        requested_id = command.object_id.strip()

        if requested_id:
            matching_id = [
                candidate for candidate in objects
                if candidate.object_id == requested_id
            ]
            if not matching_id:
                class_ids = [
                    candidate.object_id for candidate in objects
                    if grasping.normalize_label(candidate.label)
                    == requested_class
                ]
                available = ', '.join(class_ids[:5]) or 'none'
                return (
                    'object_id {} is not present; current IDs for class {}: '
                    '{}'.format(requested_id, requested_class, available)
                )
            candidate = matching_id[0]
            actual_class = grasping.normalize_label(candidate.label)
            if actual_class != requested_class:
                return (
                    'object_id {} has class {}, not requested class {}'.format(
                        requested_id, actual_class, requested_class
                    )
                )
            relevant = matching_id
        else:
            relevant = [
                candidate for candidate in objects
                if grasping.normalize_label(candidate.label)
                == requested_class
            ]
            if not relevant:
                labels = sorted({
                    grasping.normalize_label(candidate.label)
                    for candidate in objects
                    if grasping.normalize_label(candidate.label)
                })
                available = ', '.join(labels[:10]) or 'none'
                return (
                    'no object has requested class {}; available classes: '
                    '{}'.format(requested_class, available)
                )

        descriptions = []
        for candidate in relevant[:5]:
            failures = grasping.object_safeguard_failures(
                candidate,
                self._now_seconds(),
                self._minimum_confidence,
                self._minimum_observations,
                self._maximum_object_age,
                SemanticObject.STATUS_ACTIVE,
            )
            descriptions.append(
                '{}: {}'.format(
                    candidate.object_id,
                    '; '.join(failures) or 'no safeguard failure',
                )
            )
        return 'no eligible {} object; {}'.format(
            requested_class, ' | '.join(descriptions)
        )

    def _transform_candidates(self, candidates):
        source_frame = self._latest_map.header.frame_id.strip()
        if not source_frame:
            raise RuntimeError('semantic map frame_id is empty')
        transform = self._tf_buffer.lookup_transform(
            self._planning_frame, source_frame, Time()
        )
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        transform_translation = (
            translation.x, translation.y, translation.z
        )
        transform_quaternion = (
            rotation.x, rotation.y, rotation.z, rotation.w
        )
        if self._preserve_current_orientation:
            tip_transform = self._tf_buffer.lookup_transform(
                self._planning_frame, self._tip_link, Time()
            )
            tip_rotation = tip_transform.transform.rotation
            target_orientation = (
                tip_rotation.x,
                tip_rotation.y,
                tip_rotation.z,
                tip_rotation.w,
            )
        else:
            target_orientation = grasping.multiply_quaternions(
                transform_quaternion,
                grasping.quaternion_from_rpy(*self._pregrasp_rpy),
            )
        reach_translation = None
        reach_quaternion = None
        if self._reach_reference_frame != self._planning_frame:
            reach_transform = self._tf_buffer.lookup_transform(
                self._reach_reference_frame, self._planning_frame, Time()
            )
            reach_translation_message = (
                reach_transform.transform.translation
            )
            reach_rotation_message = reach_transform.transform.rotation
            reach_translation = (
                reach_translation_message.x,
                reach_translation_message.y,
                reach_translation_message.z,
            )
            reach_quaternion = (
                reach_rotation_message.x,
                reach_rotation_message.y,
                reach_rotation_message.z,
                reach_rotation_message.w,
            )
        transformed = []
        for candidate in candidates:
            source_point = (
                candidate.position.x,
                candidate.position.y,
                candidate.position.z,
            )
            source_target = tuple(
                value + offset
                for value, offset in zip(
                    source_point, self._pregrasp_offset
                )
            )
            object_point = grasping.transform_point(
                source_point,
                transform_translation,
                transform_quaternion,
            )
            target_point = grasping.transform_point(
                source_target,
                transform_translation,
                transform_quaternion,
            )
            if reach_translation is None:
                reach_point = target_point
            else:
                reach_point = grasping.transform_point(
                    target_point,
                    reach_translation,
                    reach_quaternion,
                )
            if all(
                    math.isfinite(value)
                    for value in object_point + target_point + reach_point):
                transformed.append(
                    (
                        candidate,
                        object_point,
                        target_point,
                        target_orientation,
                        reach_point,
                    )
                )
        if not transformed:
            raise RuntimeError('all transformed object positions are invalid')
        return transformed

    def _target_problem(self, target_point, reach_point):
        reach_problem = grasping.reach_safeguard_problem(
            reach_point,
            self._minimum_reach,
            self._maximum_reach,
            self._reach_reference_frame,
        )
        if reach_problem is not None:
            return reach_problem
        if not self._minimum_target_z <= target_point[2] <= (
                self._maximum_target_z):
            return (
                'pre-grasp target z is {:.3f} m in {}, outside permitted '
                'range [{:.3f}, {:.3f}] m'.format(
                    target_point[2],
                    self._planning_frame,
                    self._minimum_target_z,
                    self._maximum_target_z,
                )
            )
        return None

    def _target_pose(self, target_point, quaternion):
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = self._planning_frame
        pose.pose.position.x = target_point[0]
        pose.pose.position.y = target_point[1]
        pose.pose.position.z = target_point[2]
        pose.pose.orientation.x = quaternion[0]
        pose.pose.orientation.y = quaternion[1]
        pose.pose.orientation.z = quaternion[2]
        pose.pose.orientation.w = quaternion[3]
        return pose

    def _move_group_goal(self, target_pose):
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.SPHERE
        primitive.dimensions = [self._position_tolerance]

        region_pose = Pose()
        region_pose.position = target_pose.pose.position
        region_pose.orientation.w = 1.0

        position = PositionConstraint()
        position.header = target_pose.header
        position.link_name = self._tip_link
        position.constraint_region.primitives = [primitive]
        position.constraint_region.primitive_poses = [region_pose]
        position.weight = 1.0

        orientation = OrientationConstraint()
        orientation.header = target_pose.header
        orientation.link_name = self._tip_link
        orientation.orientation = target_pose.pose.orientation
        orientation.absolute_x_axis_tolerance = self._orientation_tolerance
        orientation.absolute_y_axis_tolerance = self._orientation_tolerance
        orientation.absolute_z_axis_tolerance = self._orientation_tolerance
        orientation.weight = 1.0

        constraints = Constraints()
        constraints.name = 'semantic_pregrasp'
        constraints.position_constraints = [position]
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
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True
        goal.planning_options.plan_only = not self._execution_enabled
        goal.planning_options.look_around = False
        goal.planning_options.replan = False
        return goal

    def _publish_active_stage(self, stage, message):
        if self._active is None or self._active['last_stage'] == stage:
            return
        self._active['last_stage'] = stage
        self._publish_status(
            self._active['command'],
            stage,
            False,
            False,
            message,
            self._active['object_id'],
            self._active['target_pose'],
        )

    def _move_group_feedback(self, feedback_message):
        if self._active is None:
            return
        state = feedback_message.feedback.state.strip()
        lowered = state.lower()
        if 'execut' in lowered or 'monitor' in lowered:
            stage = GraspStatus.STAGE_EXECUTING
        else:
            stage = GraspStatus.STAGE_PLANNING
        self._publish_active_stage(
            stage, 'MoveIt state: {}'.format(state or 'unknown')
        )

    def _move_group_goal_response(self, future):
        if self._active is None:
            return
        try:
            goal_handle = future.result()
        except Exception as error:
            self._finish_failure(
                'failed to send request to MoveIt: {}'.format(error)
            )
            return
        if not goal_handle.accepted:
            self._finish_failure('MoveIt rejected the pre-grasp request')
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._move_group_result)

    def _move_group_result(self, future):
        if self._active is None:
            return
        try:
            wrapped_result = future.result()
            result = wrapped_result.result
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
            self._finish_failure(description)
            return

        command = self._active['command']
        object_id = self._active['object_id']
        target_pose = self._active['target_pose']
        if self._execution_enabled:
            stage = GraspStatus.STAGE_PREGRASP_REACHED
            message = (
                'pre-grasp reached; safety stop active, so no contact or '
                'gripper closure was attempted'
            )
        else:
            stage = GraspStatus.STAGE_PLAN_READY
            message = (
                'pre-grasp plan succeeded; execution_enabled is false, so '
                'the arm was not moved'
            )
        self._publish_status(
            command,
            stage,
            True,
            True,
            message,
            object_id,
            target_pose,
        )
        self.get_logger().info(
            'Grasp request {} completed at guarded stage {}'.format(
                command.request_id, stage
            )
        )
        self._active = None

    def _finish_failure(self, message):
        if self._active is None:
            return
        command = self._active['command']
        self.get_logger().error(
            'Grasp request {} failed: {}'.format(
                command.request_id, message
            )
        )
        self._publish_status(
            command,
            GraspStatus.STAGE_FAILED,
            True,
            False,
            message,
            self._active['object_id'],
            self._active['target_pose'],
        )
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
