"""Tests for trajectory validation and interpolation."""

import unittest

from unitree_arm_control import trajectory


class Duration:
    """Minimal Duration stand-in for ROS-independent tests."""

    def __init__(self, seconds, nanoseconds=0):
        self.sec = seconds
        self.nanosec = nanoseconds


class Point:
    """Minimal JointTrajectoryPoint stand-in."""

    def __init__(self, positions, seconds):
        self.positions = positions
        self.time_from_start = Duration(seconds)


class TrajectoryTest(unittest.TestCase):
    """Exercise positional trajectory helpers without ROS."""

    JOINTS = tuple('joint_{}'.format(index) for index in range(7))

    def test_validate_accepts_six_arm_joints_without_gripper(self):
        times = trajectory.validate_trajectory(
            self.JOINTS[:6],
            [Point([0.0] * 6, 1), Point([1.0] * 6, 2)],
            self.JOINTS,
        )
        self.assertEqual(times, (1.0, 2.0))

    def test_validate_rejects_missing_arm_joint(self):
        with self.assertRaisesRegex(
                trajectory.TrajectoryError, 'missing'):
            trajectory.validate_trajectory(
                self.JOINTS[:5],
                [Point([0.0] * 5, 1)],
                self.JOINTS,
            )

    def test_validate_rejects_non_increasing_time(self):
        with self.assertRaisesRegex(
                trajectory.TrajectoryError, 'strictly increasing'):
            trajectory.validate_trajectory(
                self.JOINTS,
                [Point([0.0] * 7, 1), Point([1.0] * 7, 1)],
                self.JOINTS,
            )

    def test_expand_positions_preserves_uncommanded_gripper(self):
        expanded = trajectory.expand_positions(
            self.JOINTS[:6],
            [1.0] * 6,
            self.JOINTS,
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02],
        )
        self.assertEqual(expanded, (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.02))

    def test_interpolation_starts_from_feedback_position(self):
        sampled = trajectory.interpolate_positions(
            1.0,
            (2.0, 4.0),
            ((2.0, 4.0), (4.0, 8.0)),
            (0.0, 0.0),
        )
        self.assertEqual(sampled, (1.0, 2.0))

    def test_interpolation_between_waypoints(self):
        sampled = trajectory.interpolate_positions(
            3.0,
            (2.0, 4.0),
            ((2.0, 4.0), (4.0, 8.0)),
            (0.0, 0.0),
        )
        self.assertEqual(sampled, (3.0, 6.0))

    def test_velocity_validation_accepts_slow_segments(self):
        trajectory.validate_segment_velocities(
            (1.0, 3.0),
            ((0.5, 0.0), (1.0, 1.0)),
            (0.0, 0.0),
            ('joint_0', 'joint_1'),
            (1.0, 1.0),
        )

    def test_velocity_validation_rejects_fast_segment(self):
        with self.assertRaisesRegex(
                trajectory.TrajectoryError, 'velocity limit'):
            trajectory.validate_segment_velocities(
                (0.5,),
                ((1.0,),),
                (0.0,),
                ('joint_0',),
                (1.0,),
            )

    def test_position_validation_rejects_motion_beyond_limit(self):
        with self.assertRaisesRegex(
                trajectory.TrajectoryError, 'position limit'):
            trajectory.validate_position_limits(
                ((1.1,),),
                (0.0,),
                ('joint_0',),
                ((-1.0, 1.0),),
            )

    def test_position_validation_allows_outside_start_toward_limit(self):
        trajectory.validate_position_limits(
            ((-1.05,), (-0.9,)),
            (-1.1,),
            ('joint_0',),
            ((-1.0, 1.0),),
        )

    def test_position_validation_rejects_outside_start_moving_farther(self):
        with self.assertRaisesRegex(
                trajectory.TrajectoryError, 'position limit'):
            trajectory.validate_position_limits(
                ((-1.2,),),
                (-1.1,),
                ('joint_0',),
                ((-1.0, 1.0),),
            )


if __name__ == '__main__':
    unittest.main()
