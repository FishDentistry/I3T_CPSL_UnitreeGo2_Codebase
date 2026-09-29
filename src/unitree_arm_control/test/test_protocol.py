"""Tests for D1 JSON encoding and feedback parsing."""

import json
import unittest

from unitree_arm_control import protocol


class ProtocolTest(unittest.TestCase):
    """Exercise the protocol independently from ROS 2."""

    def test_set_joint_command_matches_d1_protocol(self):
        payload = json.loads(protocol.set_joint_command(4, 5, 60))
        self.assertEqual(payload, {
            'seq': 4,
            'address': 1,
            'funcode': 1,
            'data': {'id': 5, 'angle': 60.0, 'delay_ms': 0},
        })

    def test_set_all_joint_angles_matches_d1_protocol(self):
        payload = json.loads(protocol.set_joint_angles_command(
            8, [0, -60, 60, 0, 30, 0, 12], protocol.TRAJECTORY
        ))
        self.assertEqual(payload['seq'], 8)
        self.assertEqual(payload['address'], 1)
        self.assertEqual(payload['funcode'], 2)
        self.assertEqual(payload['data']['mode'], 1)
        self.assertEqual(
            [payload['data']['angle{}'.format(index)]
             for index in range(7)],
            [0, -60, 60, 0, 30, 0, 12],
        )

    def test_joint_limit_is_enforced_for_revolute_joints(self):
        with self.assertRaisesRegex(protocol.ProtocolError, 'outside'):
            protocol.set_joint_command(1, 1, 91)

    def test_gripper_has_no_unverified_limit(self):
        payload = json.loads(protocol.set_joint_command(1, 6, 123.0))
        self.assertEqual(payload['data']['angle'], 123.0)

    def test_zero_command_has_no_data_member(self):
        self.assertEqual(json.loads(protocol.zero_arm_command(9)), {
            'seq': 9,
            'address': 1,
            'funcode': 7,
        })

    def test_lay_down_command_uses_measured_pose(self):
        payload = json.loads(protocol.lay_down_command(10))
        self.assertEqual(payload['funcode'], protocol.SET_ALL_JOINTS)
        self.assertEqual(payload['data']['mode'], protocol.TRAJECTORY)
        self.assertEqual(
            tuple(
                payload['data']['angle{}'.format(index)]
                for index in range(protocol.JOINT_COUNT)
            ),
            protocol.LAY_DOWN_ANGLES_DEGREES,
        )

    def test_enable_and_power_commands(self):
        joint = json.loads(
            protocol.set_joint_enabled_command(1, 3, True)
        )
        arm = json.loads(protocol.set_arm_enabled_command(2, False))
        power = json.loads(protocol.set_power_command(3, True))
        self.assertEqual(joint['data'], {'id': 3, 'mode': 1})
        self.assertEqual(arm['data'], {'mode': 0})
        self.assertEqual(power['data'], {'power': 1})

    def test_parse_joint_angles_feedback(self):
        raw = json.dumps({
            'seq': 10,
            'address': 2,
            'funcode': 1,
            'data': {
                'angle0': 0,
                'angle1': -1,
                'angle2': 2,
                'angle3': -3,
                'angle4': 4,
                'angle5': -5,
                'angle6': 6,
            },
        })
        feedback = protocol.parse_feedback(raw)
        self.assertIsInstance(feedback, protocol.JointAnglesFeedback)
        self.assertEqual(
            feedback.angles_degrees, (0, -1, 2, -3, 4, -5, 6)
        )

    def test_parse_status_feedback(self):
        feedback = protocol.parse_feedback(json.dumps({
            'seq': 10,
            'address': 2,
            'funcode': 3,
            'data': {
                'enable_status': 1,
                'power_status': 1,
                'error_status': 0,
            },
        }))
        self.assertIsInstance(feedback, protocol.ArmStatusFeedback)
        self.assertTrue(feedback.enabled)
        self.assertTrue(feedback.powered)
        self.assertFalse(feedback.healthy)

    def test_parse_motor_status_feedback(self):
        data = {
            'motor{}_status'.format(index): int(index != 2)
            for index in range(7)
        }
        feedback = protocol.parse_feedback(json.dumps({
            'seq': 10,
            'address': 2,
            'funcode': 4,
            'data': data,
        }))
        self.assertIsInstance(feedback, protocol.MotorStatusFeedback)
        self.assertEqual(
            feedback.motor_ok,
            (True, True, False, True, True, True, True),
        )

    def test_parse_command_results(self):
        cases = (
            (1, 'recv_status', protocol.RECEIVE_RESULT),
            (2, 'exec_status', protocol.EXECUTE_RESULT),
        )
        for function_code, field_name, stage in cases:
            with self.subTest(function_code=function_code):
                feedback = protocol.parse_feedback(json.dumps({
                    'seq': 42,
                    'address': 3,
                    'funcode': function_code,
                    'data': {field_name: 1},
                }))
                self.assertIsInstance(
                    feedback, protocol.CommandResultFeedback
                )
                self.assertEqual(feedback.stage, stage)
                self.assertTrue(feedback.success)

    def test_unknown_feedback_remains_forward_compatible(self):
        feedback = protocol.parse_feedback(json.dumps({
            'seq': 10,
            'address': 2,
            'funcode': 99,
            'data': {'future': 'field'},
        }))
        self.assertIsInstance(feedback, protocol.UnknownFeedback)
        self.assertEqual(feedback.data, {'future': 'field'})

    def test_malformed_feedback_is_rejected(self):
        with self.assertRaisesRegex(protocol.ProtocolError, 'valid JSON'):
            protocol.parse_feedback('{not-json')

    def test_sequence_generator_wraps_without_zero(self):
        sequences = protocol.SequenceGenerator(0xFFFFFFFF)
        self.assertEqual(sequences.next(), 0xFFFFFFFF)
        self.assertEqual(sequences.next(), 1)


if __name__ == '__main__':
    unittest.main()
