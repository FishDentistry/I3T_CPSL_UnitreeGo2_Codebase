"""ROS 2 node wrapping the D1 ArmString command and feedback topics."""

import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy

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


class D1ArmController(Node):
    """Publish validated D1 commands and expose parsed feedback."""

    def __init__(self):
        super().__init__('d1_arm_controller')

        self.declare_parameter('command_topic', '/arm_Command')
        self.declare_parameter('feedback_topic', '/arm_Feedback')
        self.declare_parameter('commanding_enabled', False)
        self.declare_parameter('require_fresh_feedback', True)
        self.declare_parameter('feedback_timeout_sec', 2.0)
        self.declare_parameter('enforce_joint_limits', True)
        self.declare_parameter('initial_sequence', 1)
        self.declare_parameter('lay_down_tolerance_degrees', 2.0)
        self.declare_parameter('lay_down_timeout_sec', 15.0)
        self.declare_parameter('lay_down_required_samples', 3)

        command_topic = self._string_parameter('command_topic')
        feedback_topic = self._string_parameter('feedback_topic')
        self._command_topic = command_topic
        self._feedback_topic = feedback_topic
        initial_sequence = self._integer_parameter('initial_sequence')
        self._sequences = protocol.SequenceGenerator(initial_sequence)
        self._last_feedback_time = None
        self._pending_commands = {}
        self._feedback_warning_active = False
        self._lay_down_operation = None

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

        self.create_timer(5.0, self._feedback_watchdog)
        self.create_timer(0.1, self._lay_down_watchdog)

        self.get_logger().info(
            'D1 arm wrapper listening on {} and publishing to {}'.format(
                feedback_topic, command_topic
            )
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
            parsed = JointAngles()
            parsed.stamp = self._now_message()
            parsed.sequence = feedback.sequence
            parsed.angle_degrees = list(feedback.angles_degrees)
            parsed.raw_json = feedback.raw_json
            self._joint_angles_publisher.publish(parsed)
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

    def _command_block_reason(self):
        if self._lay_down_operation is not None:
            return 'lay-down-and-release is already in progress'
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

    @staticmethod
    def _reject(response, message):
        response.published = False
        response.sequence = 0
        response.message = message
        return response

    def _publish_command(self, payload, sequence, description, response):
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

    def _send_command(self, payload, sequence, description):
        message = ArmString()
        message.data = payload
        self._command_publisher.publish(message)
        self._pending_commands[sequence] = description
        self.get_logger().info(
            'Published D1 command seq={} ({})'.format(sequence, description)
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
        block_reason = self._command_block_reason()
        if block_reason is not None:
            return self._reject(response, block_reason)

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
