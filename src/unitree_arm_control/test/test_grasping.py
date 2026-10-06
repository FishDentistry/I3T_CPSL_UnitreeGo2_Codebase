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

    def test_safeguard_failures_report_exact_reasons(self):
        candidate = semantic_object(
            confidence=0.4,
            observations=2,
            last_observed=80.0,
            status=1,
        )
        failures = grasping.object_safeguard_failures(
            candidate, 100.0, 0.65, 3, 10.0, 0
        )
        self.assertEqual(len(failures), 4)
        self.assertIn('not ACTIVE', failures[0])
        self.assertIn('confidence 0.400', failures[1])
        self.assertIn('2 observations', failures[2])
        self.assertIn('20.0s old', failures[3])

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

    def test_configured_pregrasp_rotation_points_local_z_down(self):
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

    def test_approach_orientation_points_tool_z_at_object(self):
        quaternion = grasping.quaternion_from_approach((1.0, 0.0, 0.0))
        direction = grasping.rotate_vector((0.0, 0.0, 1.0), quaternion)
        self.assertAlmostEqual(direction[0], 1.0)
        self.assertAlmostEqual(direction[1], 0.0)
        self.assertAlmostEqual(direction[2], 0.0)

    def test_candidate_offsets_pregrasp_toward_arm(self):
        candidate = grasping.generate_approach_candidates(
            (1.0, 0.0, 0.2),
            (0.0, 0.0, 0.0),
            [0.0],
            0.10,
            0.02,
        )[0]
        self.assertEqual(candidate['grasp_point'], (0.98, 0.0, 0.2))
        self.assertEqual(candidate['pregrasp_point'], (0.88, 0.0, 0.2))

    def test_candidates_try_smallest_yaw_offset_first(self):
        candidates = grasping.generate_approach_candidates(
            (0.5, 0.1, 0.2),
            (0.0, 0.0, 0.0),
            [0.8, -0.2, 0.0, 0.4],
            0.07,
            0.0,
        )
        self.assertEqual(
            [candidate['yaw_offset'] for candidate in candidates],
            [0.0, -0.2, 0.4, 0.8],
        )

    def test_candidate_rejects_nonpositive_approach_distance(self):
        with self.assertRaises(ValueError):
            grasping.generate_approach_candidates(
                (0.5, 0.0, 0.2),
                (0.0, 0.0, 0.0),
                [0.0],
                0.0,
                0.0,
            )

    def test_reach_safeguard_accepts_point_inside_arm_envelope(self):
        problem = grasping.reach_safeguard_problem(
            (0.50, 0.10, 0.20), 0.10, 0.65, 'd1_base_link'
        )
        self.assertIsNone(problem)

    def test_reach_safeguard_reports_distance_frame_and_coordinates(self):
        problem = grasping.reach_safeguard_problem(
            (0.60, 0.30, 0.30), 0.10, 0.65, 'd1_base_link'
        )
        self.assertIn('0.735 m from d1_base_link', problem)
        self.assertIn('xyz [0.600, 0.300, 0.300] m', problem)
        self.assertIn('maximum reach safeguard of 0.650 m', problem)


if __name__ == '__main__':
    unittest.main()
