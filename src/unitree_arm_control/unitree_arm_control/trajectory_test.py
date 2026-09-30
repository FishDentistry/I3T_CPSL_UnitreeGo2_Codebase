"""Conservative relative-motion test for the D1 trajectory action."""

import math
import time

import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

from unitree_arm_control import protocol
from unitree_arm_control.controller import D1_JOINT_STATE_NAMES


def _duration(seconds):
    value = Duration()
    value.sec = int(seconds)
    value.nanosec = int(round((seconds - value.sec) * 1e9))
    if value.nanosec == 1000000000:
        value.sec += 1
        value.nanosec = 0
    return value


class D1TrajectoryTest(Node):
    """Build a small out-and-back trajectory from live joint feedback."""

    def __init__(self):
        super().__init__('d1_trajectory_test')
        self.declare_parameter(
            'action_name',
            '/d1_arm_controller/follow_joint_trajectory',
        )
        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('joint_name', 'd1_joint_5')
        self.declare_parameter('delta_degrees', 5.0)
        self.declare_parameter('move_duration_sec', 2.0)
        self.declare_parameter('hold_duration_sec', 1.0)
        self.declare_parameter('feedback_wait_timeout_sec', 5.0)
        self.declare_parameter('minimum_excursion_fraction', 0.5)

        self._positions = None
        self._goal_handle = None
        self._test_joint_name = None
        self._test_start_position = None
        self._commanded_excursion = None
        self._maximum_observed_excursion = 0.0
        topic = self.get_parameter(
            'joint_states_topic'
        ).get_parameter_value().string_value
        self.create_subscription(
            JointState, topic, self._joint_state_callback, 10
        )
        action_name = self.get_parameter(
            'action_name'
        ).get_parameter_value().string_value
        self._client = ActionClient(
            self, FollowJointTrajectory, action_name
        )

    def _joint_state_callback(self, message):
        by_name = dict(zip(message.name, message.position))
        if not all(name in by_name for name in D1_JOINT_STATE_NAMES):
            return
        self._positions = tuple(
            by_name[name] for name in D1_JOINT_STATE_NAMES
        )

    def _trajectory_feedback_callback(self, feedback_message):
        if self._test_joint_name is None:
            return
        feedback = feedback_message.feedback
        try:
            joint_index = feedback.joint_names.index(self._test_joint_name)
            actual_position = feedback.actual.positions[joint_index]
        except (ValueError, IndexError):
            return
        excursion = abs(actual_position - self._test_start_position)
        self._maximum_observed_excursion = max(
            self._maximum_observed_excursion,
            excursion,
        )

    def run(self):
        """Wait for feedback, send the test, and return on completion."""
        timeout = self.get_parameter(
            'feedback_wait_timeout_sec'
        ).get_parameter_value().double_value
        deadline = time.monotonic() + timeout
        while rclpy.ok() and self._positions is None:
            if time.monotonic() > deadline:
                raise RuntimeError(
                    'timed out waiting for a complete D1 joint state'
                )
            rclpy.spin_once(self, timeout_sec=0.1)

        if not self._client.wait_for_server(timeout_sec=timeout):
            raise RuntimeError(
                'D1 follow_joint_trajectory action is not available'
            )

        minimum_fraction = self.get_parameter(
            'minimum_excursion_fraction'
        ).get_parameter_value().double_value
        if not 0.0 < minimum_fraction <= 1.0:
            raise RuntimeError(
                'minimum_excursion_fraction must be greater than 0 and no '
                'more than 1'
            )
        goal = self._build_goal()
        self.get_logger().warning(
            'Sending the small out-and-back trajectory now'
        )
        send_future = self._client.send_goal_async(
            goal,
            feedback_callback=self._trajectory_feedback_callback,
        )
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError('trajectory goal was rejected')
        self._goal_handle = goal_handle

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        wrapped_result = result_future.result()
        if wrapped_result is None:
            raise RuntimeError('trajectory action returned no result')
        result = wrapped_result.result
        if result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            raise RuntimeError(
                'trajectory failed: {}'.format(result.error_string)
            )
        minimum_excursion = self._commanded_excursion * minimum_fraction
        if self._maximum_observed_excursion < minimum_excursion:
            raise RuntimeError(
                'trajectory returned to its start, but the commanded '
                'outward movement was not observed: maximum feedback '
                'excursion was {:.2f} degrees; expected at least {:.2f} '
                'degrees'.format(
                    math.degrees(self._maximum_observed_excursion),
                    math.degrees(minimum_excursion),
                )
            )
        self.get_logger().info(
            'Observed a maximum {} excursion of {:.2f} degrees'.format(
                self._test_joint_name,
                math.degrees(self._maximum_observed_excursion),
            )
        )
        self.get_logger().info(result.error_string)

    def cancel_active_goal(self):
        """Request cancellation before the test process exits."""
        if self._goal_handle is None:
            return
        self.get_logger().warning('Requesting trajectory cancellation')
        future = self._goal_handle.cancel_goal_async()
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

    def destroy_action_client(self):
        """Destroy the action client while its parent node is still valid."""
        self._client.destroy()

    def _build_goal(self):
        joint_name = self.get_parameter(
            'joint_name'
        ).get_parameter_value().string_value
        if joint_name not in D1_JOINT_STATE_NAMES[:6]:
            raise RuntimeError(
                'joint_name must be one of {}'.format(
                    ', '.join(D1_JOINT_STATE_NAMES[:6])
                )
            )
        delta = self.get_parameter(
            'delta_degrees'
        ).get_parameter_value().double_value
        if not math.isfinite(delta) or delta == 0.0 or abs(delta) > 10.0:
            raise RuntimeError(
                'delta_degrees must be non-zero and no more than 10 degrees'
            )
        move_duration = self.get_parameter(
            'move_duration_sec'
        ).get_parameter_value().double_value
        hold_duration = self.get_parameter(
            'hold_duration_sec'
        ).get_parameter_value().double_value
        if move_duration <= 0.0 or hold_duration <= 0.0:
            raise RuntimeError(
                'move_duration_sec must be positive and hold_duration_sec '
                'must be positive'
            )

        start = tuple(self._positions[:6])
        target = list(start)
        joint_index = D1_JOINT_STATE_NAMES.index(joint_name)
        current_degrees = math.degrees(start[joint_index])
        lower, upper = protocol.JOINT_LIMITS_DEGREES[joint_index]
        target_degrees = current_degrees + delta
        if not lower <= target_degrees <= upper:
            target_degrees = current_degrees - delta
        if not lower <= target_degrees <= upper:
            raise RuntimeError(
                '{} is too close to its limits for this test'.format(
                    joint_name
                )
            )
        target[joint_index] = math.radians(target_degrees)
        self._test_joint_name = joint_name
        self._test_start_position = start[joint_index]
        self._commanded_excursion = abs(
            target[joint_index] - start[joint_index]
        )
        self._maximum_observed_excursion = 0.0
        self.get_logger().info(
            'Testing {} from {:.2f} to {:.2f} degrees'.format(
                joint_name,
                current_degrees,
                target_degrees,
            )
        )

        first_time = 0.5
        target_time = first_time + move_duration
        return_time = target_time + hold_duration + move_duration
        goal = FollowJointTrajectory.Goal()
        # Omit the gripper so the controller preserves its raw D1 feedback
        # value without applying the provisional angle-to-travel calibration.
        goal.trajectory.joint_names = list(D1_JOINT_STATE_NAMES[:6])
        for positions, point_time in (
                (start, first_time),
                (target, target_time),
                (target, target_time + hold_duration),
                (start, return_time)):
            point = JointTrajectoryPoint()
            point.positions = list(positions)
            point.time_from_start = _duration(point_time)
            goal.trajectory.points.append(point)
        return goal


def main(args=None):
    """Run the conservative D1 trajectory test."""
    rclpy.init(args=args)
    node = D1TrajectoryTest()
    exit_code = 0
    try:
        node.run()
    except KeyboardInterrupt:
        node.cancel_active_goal()
        node.get_logger().warning('Trajectory test interrupted')
        exit_code = 130
    except RuntimeError as error:
        node.get_logger().error(str(error))
        exit_code = 1
    finally:
        node.destroy_action_client()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return exit_code


if __name__ == '__main__':
    main()
