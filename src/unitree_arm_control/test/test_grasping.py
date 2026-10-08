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

    def test_trajectory_endpoint_state_preserves_unplanned_joints(self):
        start = SimpleNamespace(
            joint_state=SimpleNamespace(
                name=['d1_joint_0', 'd1_joint_1', 'd1_gripper_joint'],
                position=[0.0, 0.1, 0.5],
                velocity=[0.0, 0.0, 0.0],
                effort=[0.0, 0.0, 0.0],
            ),
            is_diff=True,
        )
        trajectory = SimpleNamespace(joint_trajectory=SimpleNamespace(
            joint_names=['d1_joint_1', 'd1_joint_0'],
            points=[SimpleNamespace(positions=[0.3, -0.2])],
        ))
        endpoint = grasping.trajectory_endpoint_state(start, trajectory)
        self.assertEqual(endpoint.joint_state.position, [-0.2, 0.3, 0.5])
        self.assertEqual(start.joint_state.position, [0.0, 0.1, 0.5])
        self.assertEqual(endpoint.joint_state.velocity, [])
        self.assertFalse(endpoint.is_diff)

    def test_trajectory_endpoint_state_rejects_incomplete_plan(self):
        start = SimpleNamespace(
            joint_state=SimpleNamespace(
                name=['d1_joint_0'], position=[0.0], velocity=[], effort=[]
            ),
            is_diff=False,
        )
        for names, points in (
                (['unknown_joint'], [SimpleNamespace(positions=[0.1])]),
                (['d1_joint_0'], []),
                (['d1_joint_0'], [SimpleNamespace(positions=[math.nan])])):
            trajectory = SimpleNamespace(joint_trajectory=SimpleNamespace(
                joint_names=names, points=points,
            ))
            with self.subTest(names=names, points=points):
                with self.assertRaises(ValueError):
                    grasping.trajectory_endpoint_state(start, trajectory)

    def test_normalize_label_is_case_and_whitespace_insensitive(self):
        self.assertEqual(
            grasping.normalize_label('  Coffee   CUP '), 'coffee cup'
        )

    def test_class_forward_depth_offsets_are_normalized(self):
        offsets = grasping.parse_forward_grasp_depth_offsets([
            ' Mug = 0.04', 'water bottle=0.03'
        ])
        self.assertEqual(offsets, {'mug': 0.04, 'water bottle': 0.03})
        self.assertEqual(
            grasping.forward_grasp_depth_for_class(' MUG ', 0.05, offsets),
            0.04,
        )
        self.assertEqual(
            grasping.forward_grasp_depth_for_class('box', 0.05, offsets),
            0.05,
        )

    def test_invalid_class_forward_depth_offsets_are_rejected(self):
        for entries in (
                ['mug'], ['=0.04'], ['mug=invalid'], ['mug=-0.01'],
                ['mug=0.04', ' MUG =0.05']):
            with self.subTest(entries=entries):
                with self.assertRaises(ValueError):
                    grasping.parse_forward_grasp_depth_offsets(entries)

    def test_class_grasp_height_offsets_are_normalized_and_signed(self):
        offsets = grasping.parse_grasp_height_offsets([
            ' Mug = 0.02', 'shelf item=-0.01'
        ])
        self.assertEqual(offsets, {'mug': 0.02, 'shelf item': -0.01})
        self.assertEqual(grasping.grasp_height_for_class(' MUG ', offsets), 0.02)
        self.assertEqual(grasping.grasp_height_for_class('box', offsets), 0.0)

    def test_invalid_class_grasp_height_offsets_are_rejected(self):
        for entries in (
                ['mug'], ['=0.02'], ['mug=invalid'], ['mug=nan'],
                ['mug=0.02', ' MUG =0.03']):
            with self.subTest(entries=entries):
                with self.assertRaises(ValueError):
                    grasping.parse_grasp_height_offsets(entries)

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

        tool_up = grasping.rotate_vector((1.0, 0.0, 0.0), quaternion)
        self.assertAlmostEqual(tool_up[0], 0.0)
        self.assertAlmostEqual(tool_up[1], 0.0)
        self.assertAlmostEqual(tool_up[2], 1.0)

    def test_cartesian_orientation_candidates_are_bounded(self):
        reference = grasping.quaternion_from_rpy(0.2, -0.1, 0.3)
        candidates = grasping.approach_orientation_candidates(
            reference, 0.35
        )
        self.assertEqual(len(candidates), 13)
        for actual, expected in zip(candidates[0], reference):
            self.assertAlmostEqual(actual, expected)
        for candidate in candidates:
            self.assertAlmostEqual(sum(value * value for value in candidate), 1.0)
            dot = abs(sum(
                left * right for left, right in zip(reference, candidate)
            ))
            angle = 2.0 * math.acos(min(1.0, dot))
            self.assertLessEqual(angle, 0.35 + 1.0e-8)

    def test_cartesian_orientation_candidates_reject_invalid_input(self):
        with self.assertRaises(ValueError):
            grasping.approach_orientation_candidates((0, 0, 0, 0), 0.35)
        with self.assertRaises(ValueError):
            grasping.approach_orientation_candidates((0, 0, 0, 1), 0.0)

    def test_approach_corridor_aligns_local_z_between_endpoints(self):
        geometry = grasping.approach_corridor_geometry(
            (0.10, 0.20, 0.30), (0.20, 0.20, 0.30), 0.015
        )
        axis = grasping.rotate_vector(
            (0.0, 0.0, 1.0), geometry['orientation']
        )
        self.assertAlmostEqual(axis[0], 1.0)
        self.assertAlmostEqual(axis[1], 0.0)
        self.assertAlmostEqual(axis[2], 0.0)
        for actual, expected in zip(
                geometry['center'], (0.15, 0.20, 0.30)):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(geometry['dimensions'][0], 0.03)
        self.assertAlmostEqual(geometry['dimensions'][1], 0.03)
        self.assertAlmostEqual(geometry['dimensions'][2], 0.13)

    def test_approach_corridor_rejects_invalid_geometry(self):
        with self.assertRaises(ValueError):
            grasping.approach_corridor_geometry(
                (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.015
            )
        with self.assertRaises(ValueError):
            grasping.approach_corridor_geometry(
                (0.0, 0.0, 0.0), (0.1, 0.0, 0.0), 0.0
            )

    def test_candidate_offsets_pregrasp_toward_arm(self):
        candidate = grasping.generate_approach_candidates(
            (1.0, 0.0, 0.2),
            (0.0, 0.0, 0.0),
            [0.0],
            0.10,
            0.02,
        )[0]
        expected_direction = (
            1.0 / math.sqrt(1.04), 0.0, 0.2 / math.sqrt(1.04)
        )
        for actual, expected in zip(
                candidate['approach_direction'], expected_direction):
            self.assertAlmostEqual(actual, expected)
        for axis in range(3):
            expected_grasp = (
                (1.0, 0.0, 0.2)[axis]
                - 0.02 * expected_direction[axis]
            )
            expected_pregrasp = (
                expected_grasp - 0.10 * expected_direction[axis]
            )
            self.assertAlmostEqual(
                candidate['grasp_point'][axis], expected_grasp
            )
            self.assertAlmostEqual(
                candidate['pregrasp_point'][axis], expected_pregrasp
            )

    def test_low_object_pregrasp_remains_above_object(self):
        candidate = grasping.generate_approach_candidates(
            (0.50, 0.10, 0.20),
            (0.20, -0.10, 0.50),
            [0.0],
            0.10,
            0.0,
        )[0]
        self.assertLess(candidate['approach_direction'][2], 0.0)
        self.assertGreater(candidate['pregrasp_point'][2], 0.20)

    def test_measured_band_approach_stays_at_band_height(self):
        candidate = grasping.generate_approach_candidates(
            (0.50, 0.10, 0.08),
            (-0.046, 0.0, 0.083),
            [0.0],
            0.11,
            0.0,
            forward_grasp_depth=0.04,
            depth_direction=(1.0, 0.0, -0.5),
            horizontal_depth=True,
            level_approach=True,
        )[0]
        self.assertAlmostEqual(candidate['pregrasp_point'][2], 0.08)
        self.assertAlmostEqual(candidate['grasp_point'][2], 0.08)
        self.assertAlmostEqual(candidate['approach_direction'][2], 0.0)
        updated = grasping.retarget_approach_candidate(
            candidate, (0.51, 0.11, 0.09), 0.11, 0.0, 0.04,
            (1.0, 0.0, -0.5), 0.0, True,
        )
        self.assertAlmostEqual(updated['pregrasp_point'][2], 0.09)
        self.assertAlmostEqual(updated['grasp_point'][2], 0.09)

    def test_approach_line_error_ignores_progress_but_detects_height(self):
        self.assertAlmostEqual(
            grasping.approach_line_error(
                (0.42, 0.0, 0.08), (0.40, 0.0, 0.08), (1.0, 0.0, 0.0),
            ),
            0.0,
        )
        self.assertAlmostEqual(
            grasping.approach_line_error(
                (0.42, 0.0, 0.065), (0.40, 0.0, 0.08), (1.0, 0.0, 0.0),
            ),
            0.015,
        )

    def test_candidate_applies_forward_depth_without_lowering_target(self):
        candidate = grasping.generate_approach_candidates(
            (0.50, 0.0, 0.20),
            (0.30, -0.20, 0.40),
            [0.0],
            0.10,
            0.0,
            0.0,
            0.04,
            (1.0, 0.0, -1.0),
            0.02,
            True,
        )[0]
        for actual, expected in zip(
                candidate['grasp_point'], (0.54, 0.0, 0.22)):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(
                candidate['pregrasp_point'],
                tuple(
                    value - 0.10 * direction
                    for value, direction in zip(
                        (0.54, 0.0, 0.22),
                        candidate['approach_direction'],
                    )
                )):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(candidate['forward_grasp_depth'], 0.04)
        self.assertEqual(candidate['depth_direction'], (1.0, 0.0, 0.0))
        self.assertEqual(candidate['grasp_height_offset'], 0.02)
        self.assertTrue(candidate['horizontal_depth'])

    def test_other_classes_retain_three_dimensional_depth_correction(self):
        candidate = grasping.generate_approach_candidates(
            (0.50, 0.0, 0.20), (0.30, 0.0, 0.40), [0.0],
            0.10, 0.0, 0.0, 0.04, (1.0, 0.0, -1.0),
        )[0]
        self.assertAlmostEqual(
            candidate['grasp_point'][2], 0.20 - 0.04 / math.sqrt(2.0)
        )

    def test_yaw_candidate_preserves_direct_approach_slope(self):
        candidates = grasping.generate_approach_candidates(
            (0.50, 0.10, 0.20),
            (0.20, -0.10, 0.50),
            [0.0, math.pi / 2.0],
            0.10,
            0.0,
        )
        self.assertAlmostEqual(
            candidates[0]['approach_direction'][2],
            candidates[1]['approach_direction'][2],
        )

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

    def test_retarget_preserves_selected_approach_orientation(self):
        candidate = grasping.generate_approach_candidates(
            (0.5, 0.1, 0.2),
            (0.2, -0.1, 0.3),
            [0.3],
            0.07,
            0.0,
        )[0]
        updated = grasping.retarget_approach_candidate(
            candidate,
            (0.52, 0.09, 0.21),
            0.07,
            0.0,
            0.04,
            (1.0, 0.0, -1.0),
            0.02,
            True,
        )
        self.assertEqual(updated['orientation'], candidate['orientation'])
        for updated_axis, original_axis in zip(
                updated['approach_direction'],
                candidate['approach_direction']):
            self.assertAlmostEqual(updated_axis, original_axis)
        for actual, expected in zip(
                updated['grasp_point'],
                (0.56, 0.09, 0.23)):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(updated['forward_grasp_depth'], 0.04)
        self.assertEqual(updated['depth_direction'], (1.0, 0.0, 0.0))
        self.assertEqual(updated['grasp_height_offset'], 0.02)
        self.assertTrue(updated['horizontal_depth'])

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
