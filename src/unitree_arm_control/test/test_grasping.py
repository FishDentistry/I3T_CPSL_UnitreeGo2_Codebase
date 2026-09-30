"""Tests for ROS-independent semantic grasp helpers."""

import math
from types import SimpleNamespace
import unittest

from unitree_arm_control import grasping


def semantic_object(
        object_id='object_000001',
        label='Cup',
        confidence=0.9,
        observations=4,
        last_observed=99.0,
        status=0):
    """Create a message-like semantic object for helper tests."""
    seconds = int(last_observed)
    nanoseconds = int(round((last_observed - seconds) * 1e9))
    return SimpleNamespace(
        object_id=object_id,
        label=label,
        confidence=confidence,
        observation_count=observations,
        last_observed=SimpleNamespace(
            sec=seconds, nanosec=nanoseconds
        ),
        status=status,
    )


class GraspingTest(unittest.TestCase):
    """Exercise selection and transform behavior without ROS."""

    def test_normalize_label_is_case_and_whitespace_insensitive(self):
        self.assertEqual(
            grasping.normalize_label('  Coffee   CUP '), 'coffee cup'
        )

    def test_eligible_objects_honors_class_and_specific_id(self):
        objects = [
            semantic_object('object_000001'),
            semantic_object('object_000002'),
            semantic_object('object_000003', label='bottle'),
        ]
        matches = grasping.eligible_objects(
            objects,
            ' cup ',
            'object_000002',
            100.0,
            0.65,
            3,
            10.0,
            0,
        )
        self.assertEqual(
            [candidate.object_id for candidate in matches],
            ['object_000002'],
        )

    def test_eligible_objects_rejects_stale_or_uncertain_candidates(self):
        objects = [
            semantic_object('low', confidence=0.4),
            semantic_object('tentative', observations=2),
            semantic_object('old', last_observed=80.0),
            semantic_object('stale_status', status=1),
        ]
        matches = grasping.eligible_objects(
            objects, 'cup', '', 100.0, 0.65, 3, 10.0, 0
        )
        self.assertEqual(matches, [])

    def test_transform_point_applies_rotation_then_translation(self):
        half_angle = math.pi / 4.0
        transformed = grasping.transform_point(
            (1.0, 0.0, 0.0),
            (1.0, 2.0, 3.0),
            (0.0, 0.0, math.sin(half_angle), math.cos(half_angle)),
        )
        self.assertAlmostEqual(transformed[0], 1.0)
        self.assertAlmostEqual(transformed[1], 3.0)
        self.assertAlmostEqual(transformed[2], 3.0)

    def test_default_pregrasp_rotation_points_local_z_down(self):
        quaternion = grasping.quaternion_from_rpy(math.pi, 0.0, 0.0)
        direction = grasping.rotate_vector((0.0, 0.0, 1.0), quaternion)
        self.assertAlmostEqual(direction[0], 0.0)
        self.assertAlmostEqual(direction[1], 0.0)
        self.assertAlmostEqual(direction[2], -1.0)

    def test_quaternion_composition_applies_map_transform(self):
        map_rotation = grasping.quaternion_from_rpy(
            0.0, 0.0, math.pi / 2.0
        )
        tool_rotation = grasping.quaternion_from_rpy(math.pi, 0.0, 0.0)
        composed = grasping.multiply_quaternions(
            map_rotation, tool_rotation
        )
        direction = grasping.rotate_vector((0.0, 0.0, 1.0), composed)
        self.assertAlmostEqual(direction[0], 0.0)
        self.assertAlmostEqual(direction[1], 0.0)
        self.assertAlmostEqual(direction[2], -1.0)


if __name__ == '__main__':
    unittest.main()
