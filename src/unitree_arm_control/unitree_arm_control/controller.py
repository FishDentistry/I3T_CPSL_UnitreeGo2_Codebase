"""ROS 2 node wrapping the D1 ArmString command and feedback topics."""

import math
import threading
import time

import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionServer
from rclpy.action import CancelResponse
from rclpy.action import GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

from unitree_arm.msg import ArmStatus
from unitree_arm.msg import ArmString
from unitree_arm.msg import CommandResult
from unitree_arm.msg import JointAngles
from unitree_arm.msg import MotorStatus
from unitree_arm.srv import LayDownArm
from unitree_arm.srv import SetArmEnabled
from unitree_arm.srv import SetJoint
from unitree_arm.srv import SetJointAngles
from unitree_arm.srv import SetJointEnabled
from unitree_arm.srv import SetPower
from unitree_arm.srv import ZeroArm

from unitree_arm_control import protocol
from unitree_arm_control import trajectory


D1_JOINT_STATE_NAMES = (
    'd1_joint_0',
    'd1_joint_1',
    'd1_joint_2',
    'd1_joint_3',
    'd1_joint_4',
    'd1_joint_5',
    'd1_gripper_joint',
)

# Limits mirror the current combined Go2/D1 URDF. Revolute values are radians
# per second and the gripper value is metres per second.
D1_JOINT_VELOCITY_LIMITS = (
    1.05,
    1.05,
    1.05,
    1.73,
    1.73,
    1.73,
    0.02,
)


class D1ArmController(Node):
    """Publish validated D1 commands and expose parsed feedback."""

    def __init__(self):
        super().__init__('d1_arm_controller')

        self.declare_parameter('command_topic', '/arm_Command')
        self.declare_parameter('feedback_topic', '/arm_Feedback')
        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('gripper_closed_degrees', 0.0)
        self.declare_parameter('gripper_open_degrees', 30.0)
        self.declare_parameter('gripper_max_travel_m', 0.03)
        self.declare_parameter('commanding_enabled', False)
        self.declare_parameter('require_fresh_feedback', True)
        self.declare_parameter('feedback_timeout_sec', 2.0)
        self.declare_parameter('enforce_joint_limits', True)
        self.declare_parameter('initial_sequence', 1)
        self.declare_parameter('lay_down_tolerance_degrees', 2.0)
        self.declare_parameter('lay_down_timeout_sec', 15.0)
        self.declare_parameter('lay_down_required_samples', 3)
        self.declare_parameter('trajectory_command_rate_hz', 10.0)
        self.declare_parameter('trajectory_goal_tolerance_radians', 0.035)
        self.declare_parameter('trajectory_gripper_tolerance_m', 0.005)
        self.declare_parameter('trajectory_goal_timeout_sec', 3.0)

        command_topic = self._string_parameter('command_topic')
        feedback_topic = self._string_parameter('feedback_topic')
        joint_states_topic = self._string_parameter('joint_states_topic')
        self._command_topic = command_topic
        self._feedback_topic = feedback_topic
        self._gripper_conversion = (
            self._double_parameter('gripper_closed_degrees'),
            self._double_parameter('gripper_open_degrees'),
            self._double_parameter('gripper_max_travel_m'),
        )
        protocol.joint_state_positions_from_degrees(
            (0.0,) * protocol.JOINT_COUNT,
            *self._gripper_conversion
        )
        initial_sequence = self._integer_parameter('initial_sequence')
        self._sequences = protocol.SequenceGenerator(initial_sequence)
        self._last_feedback_time = None
        self._last_joint_feedback_time = None
        self._latest_joint_positions = None
        self._latest_angles_degrees = None
        self._pending_commands = {}
        self._feedback_warning_active = False
        self._lay_down_operation = None
        self._trajectory_goal_active = False
        self._trajectory_goal_lock = threading.Lock()
        self._trajectory_callback_group = ReentrantCallbackGroup()

        command_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        feedback_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self._command_publisher = self.create_publisher(
            ArmString, command_topic, command_qos
        )
        self.create_subscription(
            ArmString,
            feedback_topic,
            self._feedback_callback,
            feedback_qos,
        )

        self._joint_angles_publisher = self.create_publisher(
            JointAngles, '~/joint_angles', 10
        )
        self._joint_state_publisher = self.create_publisher(
            JointState, joint_states_topic, 10
        )
        self._arm_status_publisher = self.create_publisher(
            ArmStatus, '~/status', 10
        )
        self._motor_status_publisher = self.create_publisher(
            MotorStatus, '~/motor_status', 10
        )
        self._command_result_publisher = self.create_publisher(
            CommandResult, '~/command_result', 10
        )

        self.create_service(
            SetJoint, '~/set_joint', self._set_joint_callback
        )
        self.create_service(
            SetJointAngles,
            '~/set_joint_angles',
            self._set_joint_angles_callback,
        )
        self.create_service(
            SetJointEnabled,
            '~/set_joint_enabled',
            self._set_joint_enabled_callback,
        )
        self.create_service(
            SetArmEnabled,
            '~/set_arm_enabled',
            self._set_arm_enabled_callback,
        )
        self.create_service(
            SetPower, '~/set_power', self._set_power_callback
        )
        self.create_service(
            ZeroArm, '~/zero', self._zero_callback
        )
        self.create_service(
            LayDownArm,
            '~/lay_down_and_release',
            self._lay_down_and_release_callback,
        )
        self._trajectory_action = ActionServer(
            self,
            FollowJointTrajectory,
            '~/follow_joint_trajectory',
            execute_callback=self._execute_trajectory,
            goal_callback=self._trajectory_goal_callback,
            cancel_callback=self._trajectory_cancel_callback,
            callback_group=self._trajectory_callback_group,
        )

        self.create_timer(5.0, self._feedback_watchdog)
        self.create_timer(0.1, self._lay_down_watchdog)

        self.get_logger().info(
            'D1 arm wrapper listening on {} and publishing to {}'.format(
                feedback_topic, command_topic
            )
        )
        self.get_logger().info(
            'Publishing D1 joint feedback on {}'.format(
                joint_states_topic
            )
        )
        self.get_logger().info(
            'D1 trajectories available on '
            '~/follow_joint_trajectory'
        )
        if not self._bool_parameter('commanding_enabled'):
            self.get_logger().info(
                'Commanding is disabled; feedback parsing is active. Set '
                'commanding_enabled:=true only when the arm is safe to move.'
            )

    def _string_parameter(self, name):
        return self.get_parameter(name).get_parameter_value().string_value

    def _bool_parameter(self, name):
        return self.get_parameter(name).get_parameter_value().bool_value

    def _integer_parameter(self, name):
        return self.get_parameter(name).get_parameter_value().integer_value

    def _double_parameter(self, name):
        return self.get_parameter(name).get_parameter_value().double_value

    def _now_message(self):
        return self.get_clock().now().to_msg()

    def _feedback_callback(self, message):
        try:
            feedback = protocol.parse_feedback(message.data)
        except protocol.ProtocolError as error:
            self.get_logger().warning(
                'Ignored malformed {} payload: {}'.format(
                    self._feedback_topic, error
                )
            )
            return

        self._last_feedback_time = time.monotonic()
        self._feedback_warning_active = False

        if isinstance(feedback, protocol.JointAnglesFeedback):
            stamp = self._now_message()
            self._last_joint_feedback_time = time.monotonic()
            self._latest_angles_degrees = feedback.angles_degrees
            self._latest_joint_positions = (
                protocol.joint_state_positions_from_degrees(
                    feedback.angles_degrees, *self._gripper_conversion
                )
            )
            parsed = JointAngles()
            parsed.stamp = stamp
            parsed.sequence = feedback.sequence
            parsed.angle_degrees = list(feedback.angles_degrees)
            parsed.raw_json = feedback.raw_json
            self._joint_angles_publisher.publish(parsed)
            self._publish_joint_state(feedback.angles_degrees, stamp)
            self._update_lay_down_position(feedback.angles_degrees)
            return

        if isinstance(feedback, protocol.ArmStatusFeedback):
            parsed = ArmStatus()
            parsed.stamp = self._now_message()
            parsed.sequence = feedback.sequence
            parsed.enabled = feedback.enabled
            parsed.powered = feedback.powered
            parsed.healthy = feedback.healthy
            parsed.raw_json = feedback.raw_json
            self._arm_status_publisher.publish(parsed)
            return

        if isinstance(feedback, protocol.MotorStatusFeedback):
            parsed = MotorStatus()
            parsed.stamp = self._now_message()
            parsed.sequence = feedback.sequence
            parsed.motor_ok = list(feedback.motor_ok)
            parsed.raw_json = feedback.raw_json
            self._motor_status_publisher.publish(parsed)
            return

        if isinstance(feedback, protocol.CommandResultFeedback):
            parsed = CommandResult()
            parsed.stamp = self._now_message()
            parsed.sequence = feedback.sequence
            parsed.stage = feedback.stage
            parsed.success = feedback.success
            parsed.command = self._pending_commands.get(
                feedback.sequence, 'unknown'
            )
            parsed.raw_json = feedback.raw_json
            self._command_result_publisher.publish(parsed)
            self._update_lay_down_result(feedback)
            if feedback.stage == protocol.EXECUTE_RESULT:
                self._pending_commands.pop(feedback.sequence, None)
            return

        self.get_logger().debug(
            'Received unsupported D1 feedback address={} funcode={}'.format(
                feedback.address, feedback.function_code
            )
        )

    def _publish_joint_state(self, angles_degrees, stamp):
        joint_state = JointState()
        joint_state.header.stamp = stamp
        joint_state.name = list(D1_JOINT_STATE_NAMES)
        joint_state.position = list(
            protocol.joint_state_positions_from_degrees(
                angles_degrees, *self._gripper_conversion
            )
        )
        self._joint_state_publisher.publish(joint_state)

    def _feedback_watchdog(self):
        if self._last_feedback_time is None:
            if not self._feedback_warning_active:
                self.get_logger().warning(
                    'No valid messages received from {} yet'.format(
                        self._feedback_topic
                    )
                )
                self._feedback_warning_active = True
            return
        age = time.monotonic() - self._last_feedback_time
        timeout = self._double_parameter('feedback_timeout_sec')
        if age > timeout:
            if not self._feedback_warning_active:
                self.get_logger().warning(
                    '{} is stale ({:.1f} seconds old)'.format(
                        self._feedback_topic, age
                    )
                )
                self._feedback_warning_active = True
        else:
            self._feedback_warning_active = False

    def _command_block_reason(self, ignore_trajectory=False):
        if self._lay_down_operation is not None:
            return 'lay-down-and-release is already in progress'
        if self._trajectory_goal_active and not ignore_trajectory:
            return 'a joint trajectory is already in progress'
        if not self._bool_parameter('commanding_enabled'):
            return 'commanding_enabled is false'
        if not self._bool_parameter('require_fresh_feedback'):
            return None
        if self._last_feedback_time is None:
            return 'no valid {} message has been received'.format(
                self._feedback_topic
            )
        feedback_age = time.monotonic() - self._last_feedback_time
        timeout = self._double_parameter('feedback_timeout_sec')
        if feedback_age > timeout:
            return '{} is stale ({:.1f} seconds)'.format(
                self._feedback_topic, feedback_age
            )
        return None

    def _trajectory_goal_callback(self, goal_request):
        with self._trajectory_goal_lock:
            block_reason = self._trajectory_block_reason()
            if block_reason is not None:
                self.get_logger().warning(
                    'Rejected trajectory: {}'.format(block_reason)
                )
                return GoalResponse.REJECT
            self._trajectory_goal_active = True

        try:
            trajectory.validate_trajectory(
                goal_request.trajectory.joint_names,
                goal_request.trajectory.points,
                D1_JOINT_STATE_NAMES,
            )
            self._validate_trajectory_tolerances(goal_request)
        except trajectory.TrajectoryError as error:
            with self._trajectory_goal_lock:
                self._trajectory_goal_active = False
            self.get_logger().warning(
                'Rejected trajectory: {}'.format(error)
            )
            return GoalResponse.REJECT

        rate_hz = self._double_parameter('trajectory_command_rate_hz')
        if not math.isfinite(rate_hz) or rate_hz <= 0.0:
            with self._trajectory_goal_lock:
                self._trajectory_goal_active = False
            self.get_logger().error(
                'Rejected trajectory: trajectory_command_rate_hz must be '
                'positive'
            )
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _trajectory_block_reason(self, ignore_trajectory=False):
        block_reason = self._command_block_reason(ignore_trajectory)
        if block_reason is not None:
            return block_reason
        if self._last_joint_feedback_time is None:
            return 'no joint-angle feedback has been received'
        feedback_age = time.monotonic() - self._last_joint_feedback_time
        timeout = self._double_parameter('feedback_timeout_sec')
        if feedback_age > timeout:
            return 'joint-angle feedback is stale ({:.1f} seconds)'.format(
                feedback_age
            )
        return None

    @staticmethod
    def _trajectory_cancel_callback(goal_handle):
        del goal_handle
        return CancelResponse.ACCEPT

    @staticmethod
    def _validate_trajectory_tolerances(goal_request):
        valid_names = set(goal_request.trajectory.joint_names)
        if goal_request.path_tolerance:
            raise trajectory.TrajectoryError(
                'path_tolerance is not supported by the D1 adapter'
            )
        for tolerance in goal_request.goal_tolerance:
            if tolerance.name not in valid_names:
                raise trajectory.TrajectoryError(
                    'tolerance refers to unknown trajectory joint '
                    '{}'.format(tolerance.name)
                )
            if not math.isfinite(tolerance.position):
                raise trajectory.TrajectoryError(
                    'position tolerance for {} must be finite'.format(
                        tolerance.name
                    )
                )
            if tolerance.position < 0.0 and tolerance.position != -1.0:
                raise trajectory.TrajectoryError(
                    'position tolerance for {} must be positive, zero, '
                    'or -1'.format(tolerance.name)
                )
            if tolerance.velocity != 0.0 or tolerance.acceleration != 0.0:
                raise trajectory.TrajectoryError(
                    'only position goal tolerances are supported'
                )
        if trajectory.duration_seconds(
                goal_request.goal_time_tolerance) < 0.0:
            raise trajectory.TrajectoryError(
                'goal_time_tolerance must not be negative'
            )

    def _execute_trajectory(self, goal_handle):
        result = FollowJointTrajectory.Result()
        goal = goal_handle.request
        names = tuple(goal.trajectory.joint_names)

        try:
            if self._latest_joint_positions is None:
                return self._abort_trajectory(
                    goal_handle,
                    result,
                    FollowJointTrajectory.Result.INVALID_GOAL,
                    'no joint-angle feedback is available',
                )

            waypoint_times = trajectory.validate_trajectory(
                names,
                goal.trajectory.points,
                D1_JOINT_STATE_NAMES,
            )
            start_positions = tuple(self._latest_joint_positions)
            held_gripper_degrees = None
            if 'd1_gripper_joint' not in names:
                held_gripper_degrees = self._latest_angles_degrees[6]
            waypoints = tuple(
                trajectory.expand_positions(
                    names,
                    point.positions,
                    D1_JOINT_STATE_NAMES,
                    start_positions,
                )
                for point in goal.trajectory.points
            )
            if self._bool_parameter('enforce_joint_limits'):
                position_limits = tuple(
                    (
                        math.radians(limits[0]),
                        math.radians(limits[1]),
                    )
                    if limits is not None else None
                    for limits in protocol.JOINT_LIMITS_DEGREES
                )
                trajectory.validate_position_limits(
                    waypoints,
                    start_positions,
                    D1_JOINT_STATE_NAMES,
                    position_limits,
                )
            trajectory.validate_segment_velocities(
                waypoint_times,
                waypoints,
                start_positions,
                D1_JOINT_STATE_NAMES,
                D1_JOINT_VELOCITY_LIMITS,
                tuple(
                    self._double_parameter(
                        'trajectory_gripper_tolerance_m'
                    )
                    if name == 'd1_gripper_joint'
                    else self._double_parameter(
                        'trajectory_goal_tolerance_radians'
                    )
                    for name in D1_JOINT_STATE_NAMES
                ),
            )

            # Validate every expanded waypoint against the configured D1
            # conversion and mechanical limits before publishing any motion.
            for positions in waypoints:
                angles = self._trajectory_angles_degrees(
                    positions, held_gripper_degrees
                )
                protocol.set_joint_angles_command(
                    1, angles, protocol.SMOOTH_10_HZ,
                    False
                )

            rate_hz = self._double_parameter(
                'trajectory_command_rate_hz'
            )
            period = 1.0 / rate_hz
            started = time.monotonic()

            while True:
                if goal_handle.is_cancel_requested:
                    self._hold_current_position()
                    goal_handle.canceled()
                    result.error_code = (
                        FollowJointTrajectory.Result.SUCCESSFUL
                    )
                    result.error_string = (
                        'trajectory canceled; commanded the most recent '
                        'feedback position as a hold target'
                    )
                    return result

                block_reason = self._trajectory_block_reason(
                    ignore_trajectory=True
                )
                if block_reason is not None:
                    return self._abort_trajectory(
                        goal_handle,
                        result,
                        FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED,
                        block_reason,
                    )

                elapsed = time.monotonic() - started
                desired = trajectory.interpolate_positions(
                    elapsed,
                    waypoint_times,
                    waypoints,
                    start_positions,
                )
                self._send_trajectory_sample(
                    desired, held_gripper_degrees
                )
                self._publish_trajectory_feedback(
                    goal_handle, names, desired
                )

                if elapsed >= waypoint_times[-1]:
                    break
                time.sleep(
                    min(period, max(0.0, waypoint_times[-1] - elapsed))
                )

            return self._wait_for_trajectory_goal(
                goal_handle, result, names, waypoints[-1], goal
            )
        except (protocol.ProtocolError, trajectory.TrajectoryError) as error:
            return self._abort_trajectory(
                goal_handle,
                result,
                FollowJointTrajectory.Result.INVALID_GOAL,
                str(error),
            )
        except Exception as error:  # keep an action failure from killing node
            self.get_logger().error(
                'Unexpected trajectory execution failure: {}'.format(error)
            )
            return self._abort_trajectory(
                goal_handle,
                result,
                FollowJointTrajectory.Result.INVALID_GOAL,
                'unexpected controller error: {}'.format(error),
            )
        finally:
            with self._trajectory_goal_lock:
                self._trajectory_goal_active = False

    def _trajectory_angles_degrees(
            self, joint_positions, held_gripper_degrees=None):
        angles = list(protocol.d1_degrees_from_joint_state_positions(
            joint_positions, *self._gripper_conversion
        ))
        if held_gripper_degrees is not None:
            angles[6] = held_gripper_degrees
        return tuple(angles)

    def _send_trajectory_sample(
            self, joint_positions, held_gripper_degrees=None):
        angles = self._trajectory_angles_degrees(
            joint_positions, held_gripper_degrees
        )
        sequence = self._sequences.next()
        payload = protocol.set_joint_angles_command(
            sequence,
            angles,
            protocol.SMOOTH_10_HZ,
            False,
        )
        self._send_command(
            payload,
            sequence,
            'joint trajectory sample',
            log=False,
            track=False,
        )

    def _hold_current_position(self):
        if self._latest_angles_degrees is None:
            return
        try:
            sequence = self._sequences.next()
            payload = protocol.set_joint_angles_command(
                sequence,
                self._latest_angles_degrees,
                protocol.TRAJECTORY,
                False,
            )
            self._send_command(
                payload, sequence, 'hold after trajectory stop'
            )
        except protocol.ProtocolError as error:
            self.get_logger().error(
                'Could not command a hold position: {}'.format(error)
            )

    def _publish_trajectory_feedback(
            self, goal_handle, commanded_names, desired_positions):
        if self._latest_joint_positions is None:
            return
        name_to_index = {
            name: index
            for index, name in enumerate(D1_JOINT_STATE_NAMES)
        }
        desired = tuple(
            desired_positions[name_to_index[name]]
            for name in commanded_names
        )
        actual = tuple(
            self._latest_joint_positions[name_to_index[name]]
            for name in commanded_names
        )

        feedback = FollowJointTrajectory.Feedback()
        feedback.header.stamp = self._now_message()
        feedback.joint_names = list(commanded_names)
        feedback.desired = JointTrajectoryPoint()
        feedback.desired.positions = list(desired)
        feedback.actual = JointTrajectoryPoint()
        feedback.actual.positions = list(actual)
        feedback.error = JointTrajectoryPoint()
        feedback.error.positions = [
            target - measured
            for target, measured in zip(desired, actual)
        ]
        goal_handle.publish_feedback(feedback)

    def _wait_for_trajectory_goal(
            self, goal_handle, result, names, final_positions, goal):
        timeout = trajectory.duration_seconds(goal.goal_time_tolerance)
        if timeout <= 0.0:
            timeout = self._double_parameter(
                'trajectory_goal_timeout_sec'
            )
        deadline = time.monotonic() + timeout
        tolerances = self._trajectory_goal_tolerances(goal, names)
        name_to_index = {
            name: index
            for index, name in enumerate(D1_JOINT_STATE_NAMES)
        }

        while time.monotonic() <= deadline:
            if goal_handle.is_cancel_requested:
                self._hold_current_position()
                goal_handle.canceled()
                result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
                result.error_string = 'trajectory canceled while settling'
                return result

            block_reason = self._trajectory_block_reason(
                ignore_trajectory=True
            )
            if block_reason is not None:
                return self._abort_trajectory(
                    goal_handle,
                    result,
                    FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED,
                    block_reason,
                )

            actual = self._latest_joint_positions
            if actual is not None and all(
                    abs(final_positions[name_to_index[name]]
                        - actual[name_to_index[name]]) <= tolerances[name]
                    for name in names):
                goal_handle.succeed()
                result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
                result.error_string = 'trajectory reached its final point'
                return result

            self._publish_trajectory_feedback(
                goal_handle, names, final_positions
            )
            time.sleep(0.1)

        self._hold_current_position()
        return self._abort_trajectory(
            goal_handle,
            result,
            FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED,
            'final joint positions were not reached before timeout',
        )

    def _trajectory_goal_tolerances(self, goal, names):
        arm_default = self._double_parameter(
            'trajectory_goal_tolerance_radians'
        )
        gripper_default = self._double_parameter(
            'trajectory_gripper_tolerance_m'
        )
        tolerances = {
            name: (
                gripper_default
                if name == 'd1_gripper_joint'
                else arm_default
            )
            for name in names
        }
        for requested in goal.goal_tolerance:
            if requested.position == -1.0:
                tolerances[requested.name] = float('inf')
            elif requested.position > 0.0:
                tolerances[requested.name] = requested.position
        return tolerances

    def _abort_trajectory(
            self, goal_handle, result, error_code, message):
        self.get_logger().error(
            'Trajectory aborted: {}'.format(message)
        )
        goal_handle.abort()
        result.error_code = error_code
        result.error_string = message
        return result

    @staticmethod
    def _reject(response, message):
        response.published = False
        response.sequence = 0
        response.message = message
        return response

    def _publish_command(self, payload, sequence, description, response):
        with self._trajectory_goal_lock:
            block_reason = self._command_block_reason()
            if block_reason is not None:
                return self._reject(response, block_reason)
            self._send_command(payload, sequence, description)
        response.published = True
        response.sequence = sequence
        response.message = (
            'published; delivery/execution is reported on the controller\'s '
            'command_result topic'
        )
        return response

    def _send_command(
            self, payload, sequence, description, log=True, track=True):
        message = ArmString()
        message.data = payload
        self._command_publisher.publish(message)
        if track:
            self._pending_commands[sequence] = description
        if log:
            self.get_logger().info(
                'Published D1 command seq={} ({})'.format(
                    sequence, description
                )
            )

    def _build_and_publish(self, builder, description, response):
        sequence = self._sequences.next()
        try:
            payload = builder(sequence)
        except protocol.ProtocolError as error:
            return self._reject(response, str(error))
        return self._publish_command(
            payload, sequence, description, response
        )

    def _set_joint_callback(self, request, response):
        enforce_limits = self._bool_parameter('enforce_joint_limits')
        return self._build_and_publish(
            lambda sequence: protocol.set_joint_command(
                sequence,
                request.joint_id,
                request.angle_degrees,
                enforce_limits,
            ),
            'set joint {}'.format(request.joint_id),
            response,
        )

    def _set_joint_angles_callback(self, request, response):
        enforce_limits = self._bool_parameter('enforce_joint_limits')
        return self._build_and_publish(
            lambda sequence: protocol.set_joint_angles_command(
                sequence,
                request.angle_degrees,
                request.mode,
                enforce_limits,
            ),
            'set all joint angles',
            response,
        )

    def _set_joint_enabled_callback(self, request, response):
        return self._build_and_publish(
            lambda sequence: protocol.set_joint_enabled_command(
                sequence, request.joint_id, request.enabled
            ),
            '{} joint {}'.format(
                'enable' if request.enabled else 'release',
                request.joint_id,
            ),
            response,
        )

    def _set_arm_enabled_callback(self, request, response):
        return self._build_and_publish(
            lambda sequence: protocol.set_arm_enabled_command(
                sequence, request.enabled
            ),
            'enable arm' if request.enabled else 'release arm',
            response,
        )

    def _set_power_callback(self, request, response):
        return self._build_and_publish(
            lambda sequence: protocol.set_power_command(
                sequence, request.powered
            ),
            'power on' if request.powered else 'power off',
            response,
        )

    def _zero_callback(self, request, response):
        del request
        return self._build_and_publish(
            protocol.zero_arm_command, 'return to zero', response
        )

    def _lay_down_and_release_callback(self, request, response):
        del request
        tolerance = self._double_parameter('lay_down_tolerance_degrees')
        timeout = self._double_parameter('lay_down_timeout_sec')
        required_samples = self._integer_parameter(
            'lay_down_required_samples'
        )
        if tolerance < 0.0:
            return self._reject(
                response, 'lay_down_tolerance_degrees must not be negative'
            )
        if timeout <= 0.0:
            return self._reject(
                response, 'lay_down_timeout_sec must be positive'
            )
        if required_samples < 1:
            return self._reject(
                response, 'lay_down_required_samples must be at least one'
            )

        with self._trajectory_goal_lock:
            block_reason = self._command_block_reason()
            if block_reason is not None:
                return self._reject(response, block_reason)

            sequence = self._sequences.next()
            payload = protocol.lay_down_command(sequence)
            self._send_command(payload, sequence, 'move to lay-down pose')
            self._lay_down_operation = {
                'phase': 'moving',
                'move_sequence': sequence,
                'release_sequence': None,
                'consecutive_samples': 0,
                'position_confirmed': False,
                'move_executed': False,
                'deadline': time.monotonic() + timeout,
            }

        response.published = True
        response.sequence = sequence
        response.message = (
            'lay-down motion started; joints will be released only after '
            'successful execution and target confirmation from feedback'
        )
        return response

    def _update_lay_down_position(self, angles_degrees):
        operation = self._lay_down_operation
        if operation is None or operation['phase'] != 'moving':
            return

        maximum_error = max(
            abs(actual - target)
            for actual, target in zip(
                angles_degrees, protocol.LAY_DOWN_ANGLES_DEGREES
            )
        )
        tolerance = self._double_parameter('lay_down_tolerance_degrees')
        if maximum_error > tolerance:
            operation['consecutive_samples'] = 0
            operation['position_confirmed'] = False
            return

        operation['consecutive_samples'] += 1
        required_samples = self._integer_parameter(
            'lay_down_required_samples'
        )
        if operation['consecutive_samples'] < required_samples:
            return

        operation['position_confirmed'] = True
        self._try_lay_down_release()

    def _try_lay_down_release(self):
        operation = self._lay_down_operation
        if (
                operation is None
                or operation['phase'] != 'moving'
                or not operation['position_confirmed']
                or not operation['move_executed']):
            return

        release_sequence = self._sequences.next()
        release_payload = protocol.set_arm_enabled_command(
            release_sequence, False
        )
        operation['phase'] = 'releasing'
        operation['release_sequence'] = release_sequence
        operation['deadline'] = (
            time.monotonic()
            + self._double_parameter('lay_down_timeout_sec')
        )
        self._send_command(
            release_payload,
            release_sequence,
            'release arm after confirmed lay-down pose',
        )
        self.get_logger().info(
            'Lay-down motion execution and position confirmed; sent joint '
            'release seq={}'.format(release_sequence)
        )

    def _update_lay_down_result(self, feedback):
        operation = self._lay_down_operation
        if operation is None:
            return

        if (
                operation['phase'] == 'moving'
                and feedback.sequence == operation['move_sequence']):
            if not feedback.success:
                self.get_logger().error(
                    'Lay-down motion command failed; joints were not released'
                )
                self._lay_down_operation = None
            elif feedback.stage == protocol.EXECUTE_RESULT:
                operation['move_executed'] = True
                self._try_lay_down_release()
            return

        if feedback.sequence != operation['release_sequence']:
            return
        if not feedback.success:
            self.get_logger().error(
                'Lay-down pose was reached, but the joint release command '
                'failed; verify the arm state before touching it'
            )
            self._lay_down_operation = None
            return
        if feedback.stage == protocol.EXECUTE_RESULT:
            self.get_logger().info(
                'Lay-down-and-release completed successfully'
            )
            self._lay_down_operation = None

    def _lay_down_watchdog(self):
        operation = self._lay_down_operation
        if operation is None or time.monotonic() <= operation['deadline']:
            return

        if operation['phase'] == 'moving':
            self.get_logger().error(
                'Lay-down motion was not both execution-acknowledged and '
                'position-confirmed before timeout; joints were not released'
            )
        else:
            self.get_logger().error(
                'Joint release command was not confirmed before timeout; '
                'verify the arm state before touching it'
            )
        self._lay_down_operation = None


def main(args=None):
    """Run the D1 arm controller node."""
    rclpy.init(args=args)
    node = D1ArmController()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
