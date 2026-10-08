"""Static consistency tests for the semantic grasp coordinator."""

import ast
from pathlib import Path
import unittest


REPOSITORY_SRC = Path(__file__).resolve().parents[2]
CONTROL_ROOT = REPOSITORY_SRC / 'unitree_arm_control'
INTERFACE_ROOT = REPOSITORY_SRC / 'unitree_arm'
LAUNCHER_ROOT = REPOSITORY_SRC / 'go2_launcher'


class GraspingConfigTest(unittest.TestCase):
    """Keep interfaces, entry points, and launch wiring synchronized."""

    def test_interface_package_generates_grasp_messages(self):
        cmake = (INTERFACE_ROOT / 'CMakeLists.txt').read_text()
        self.assertIn('"msg/GraspCommand.msg"', cmake)
        self.assertIn('"msg/GraspStatus.msg"', cmake)
        self.assertIn('geometry_msgs', cmake)
        self.assertIn('std_msgs', cmake)

    def test_control_package_installs_coordinator(self):
        setup = (CONTROL_ROOT / 'setup.py').read_text()
        self.assertIn('d1_grasp_coordinator', setup)
        self.assertIn('grasp_coordinator:main', setup)

    def test_moveit_launch_starts_guarded_coordinator(self):
        launch = (
            CONTROL_ROOT / 'launch' / 'd1_moveit.launch.py'
        ).read_text()
        config = (CONTROL_ROOT / 'config' / 'grasping.yaml').read_text()
        self.assertIn("executable='d1_grasp_coordinator'", launch)
        self.assertIn("'grasp_execution_enabled'", launch)
        self.assertIn("'execution_enabled': grasp_execution", launch)
        self.assertNotIn('execution_enabled:', config)

    def test_reach_safeguard_uses_arm_base_frame(self):
        config = (CONTROL_ROOT / 'config' / 'grasping.yaml').read_text()
        coordinator = (
            CONTROL_ROOT / 'unitree_arm_control' / 'grasp_coordinator.py'
        ).read_text()
        self.assertIn('reach_reference_frame: d1_base_link', config)
        self.assertIn("'reach_reference_frame': 'd1_base_link'", coordinator)
        self.assertIn('maximum_reach_m: 0.8', config)
        self.assertIn("self.declare_parameter('maximum_reach_m', 0.67)", coordinator)

    def test_grasp_candidate_age_default_is_twenty_seconds(self):
        config = (CONTROL_ROOT / 'config' / 'grasping.yaml').read_text()
        coordinator = (
            CONTROL_ROOT / 'unitree_arm_control' / 'grasp_coordinator.py'
        ).read_text()
        self.assertIn('maximum_object_age_sec: 20.0', config)
        self.assertIn(
            "declare_parameter('maximum_object_age_sec', 20.0)",
            coordinator,
        )

    def test_side_approach_candidates_replace_fixed_orientation(self):
        config = (CONTROL_ROOT / 'config' / 'grasping.yaml').read_text()
        coordinator = (
            CONTROL_ROOT / 'unitree_arm_control' / 'grasp_coordinator.py'
        ).read_text()
        self.assertIn('approach_yaw_offsets_rad:', config)
        self.assertIn('approach_distance_m: 0.11', config)
        self.assertIn('generate_approach_candidates(', coordinator)
        self.assertNotIn('preserve_current_orientation:', config)

    def test_per_class_forward_grasp_depth_configuration(self):
        config = (CONTROL_ROOT / 'config' / 'grasping.yaml').read_text()
        coordinator = (
            CONTROL_ROOT / 'unitree_arm_control' / 'grasp_coordinator.py'
        ).read_text()
        self.assertIn(
            'default_forward_grasp_depth_offset_m: 0.05', config
        )
        self.assertIn('class_forward_grasp_depth_offsets_m:', config)
        self.assertIn('class_grasp_height_offsets_m:', config)
        self.assertIn('horizontal_forward_depth_classes:', config)
        self.assertIn('parse_grasp_height_offsets(', coordinator)
        self.assertIn('grasp_height_for_class(', coordinator)
        self.assertIn('camera_frame: camera_link', config)
        self.assertIn("'camera_frame': 'camera_link'", coordinator)
        self.assertIn("'camera_frame',", coordinator)
        self.assertIn('forward_grasp_depth_for_class(', coordinator)

    def test_contact_sequence_uses_constant_orientation_cartesian_approach(self):
        config = (CONTROL_ROOT / 'config' / 'grasping.yaml').read_text()
        coordinator = (
            CONTROL_ROOT / 'unitree_arm_control' / 'grasp_coordinator.py'
        ).read_text()
        self.assertIn('detections_topic: /grounding_dino/detection_array', config)
        self.assertIn('grasp_position_tolerance_m: 0.005', config)
        self.assertIn('GetCartesianPath', coordinator)
        self.assertIn('GetPositionFK', coordinator)
        self.assertIn('ExecuteTrajectory', coordinator)
        self.assertIn("'reacquiring'", coordinator)
        self.assertIn('retarget_approach_candidate(', coordinator)
        self.assertIn('self._current_tip_orientation()', coordinator)
        self.assertIn(
            "candidate['orientation'] = achieved_orientation", coordinator
        )
        self.assertIn(
            'request.waypoints = [copy.deepcopy(pose.pose)]', coordinator
        )
        self.assertIn('request.avoid_collisions = True', coordinator)
        self.assertIn('self._minimum_cartesian_fraction', coordinator)
        self.assertIn('minimum_cartesian_fraction: 1.0', config)
        self.assertIn('request.start_state = copy.deepcopy(start_state)', coordinator)
        self.assertIn('trajectory_endpoint_state(', coordinator)
        self.assertIn("grasp_pose, 'preflight'", coordinator)
        self.assertIn(
            "if purpose == 'reposition_pregrasp':\n"
            "            self._check_planned_pregrasp(",
            coordinator,
        )
        self.assertIn('straight_final_approach_corridor', coordinator)
        self.assertIn('goal.request.path_constraints', coordinator)
        self.assertIn('approach_corridor_radius_m: 0.015', config)
        self.assertIn("if purpose == 'approach':", coordinator)
        self.assertNotIn('_restart_candidate_screening(', coordinator)

    def test_candidates_are_screened_before_pregrasp_execution(self):
        config = (CONTROL_ROOT / 'config' / 'grasping.yaml').read_text()
        coordinator = (
            CONTROL_ROOT / 'unitree_arm_control' / 'grasp_coordinator.py'
        ).read_text()
        self.assertIn('tip_link: d1_gripper_center', config)
        self.assertIn('PositionConstraint', coordinator)
        self.assertIn('constraints.position_constraints', coordinator)
        self.assertIn('OrientationConstraint', coordinator)
        self.assertIn('constraints.orientation_constraints', coordinator)
        self.assertIn('grasp_orientation_tolerance_rad: 0.35', config)
        self.assertIn('semantic_position_target', coordinator)
        self.assertIn('semantic_pose_target', coordinator)
        self.assertIn('goal.planning_options.plan_only = True', coordinator)
        self.assertIn('result.planned_trajectory', coordinator)
        self.assertIn("result.trajectory_start, trajectory, 'pregrasp'", coordinator)
        self.assertIn('self._active[\'preflight_trajectory\']', coordinator)
        self.assertIn('after full approach validation', coordinator)
        self.assertNotIn('GetPositionIK', coordinator)
        self.assertNotIn('tool_roll_offsets_rad:', config)
        self.assertNotIn("'validated_candidates': []", coordinator)

    def test_pregrasp_fk_callback_captures_request_token(self):
        coordinator = (
            CONTROL_ROOT / 'unitree_arm_control' / 'grasp_coordinator.py'
        ).read_text()
        module = ast.parse(coordinator)
        coordinator_class = next(
            node for node in module.body
            if isinstance(node, ast.ClassDef)
            and node.name == 'D1GraspCoordinator'
        )
        method = next(
            node for node in coordinator_class.body
            if isinstance(node, ast.FunctionDef)
            and node.name == '_check_planned_pregrasp'
        )
        token_assignments = [
            node.lineno for node in ast.walk(method)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == 'token'
                for target in node.targets
            )
        ]
        callback_lines = [
            node.lineno for node in ast.walk(method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'add_done_callback'
        ]
        self.assertEqual(len(token_assignments), 1)
        self.assertEqual(len(callback_lines), 1)
        self.assertLess(token_assignments[0], callback_lines[0])

    def test_primary_approach_uses_arm_mount_position(self):
        coordinator = (
            CONTROL_ROOT / 'unitree_arm_control' / 'grasp_coordinator.py'
        ).read_text()
        self.assertIn(
            'self._planning_frame, self._reach_reference_frame, Time()',
            coordinator,
        )
        self.assertIn(
            'approach_origin = arm_origin', coordinator
        )
        self.assertNotIn(
            'tip_transform = self._tf_buffer.lookup_transform(',
            coordinator,
        )
        self.assertIn(
            'target_point = object_point if band is None else band[0]',
            coordinator,
        )

    def test_narrow_band_target_keeps_semantic_center_for_association(self):
        config = (CONTROL_ROOT / 'config' / 'grasping.yaml').read_text()
        coordinator = (
            CONTROL_ROOT / 'unitree_arm_control' / 'grasp_coordinator.py'
        ).read_text()
        detection = (
            REPOSITORY_SRC / 'intel_realsense_interfaces' / 'msg'
            / 'GroundedDetection.msg'
        ).read_text()
        self.assertIn('use_grasp_band: true', config)
        self.assertIn('maximum_grasp_band_width_m: 0.07', config)
        self.assertIn('bool has_grasp_band', detection)
        self.assertIn('float32 grasp_band_width_m', detection)
        self.assertIn('GraspBand[] grasp_band_candidates', detection)
        self.assertIn(
            'self._initial_grasp_bands(object_point, object_class)',
            coordinator,
        )
        self.assertIn("candidate['band_target_point'] = target_point", coordinator)
        self.assertIn("candidate['band_count'] = len(bands)", coordinator)
        self.assertIn("'object_point': object_point", coordinator)
        self.assertIn("'target_point': target_point", coordinator)

    def test_gripper_sequence_holds_releases_and_does_not_lift(self):
        config = (CONTROL_ROOT / 'config' / 'grasping.yaml').read_text()
        coordinator = (
            CONTROL_ROOT / 'unitree_arm_control' / 'grasp_coordinator.py'
        ).read_text()
        status = (INTERFACE_ROOT / 'msg' / 'GraspStatus.msg').read_text()
        self.assertIn('hold_duration_sec: 3.0', config)
        self.assertIn('gripper_service: /d1_arm_controller/set_joint', config)
        self.assertIn('gripper_open_degrees: 49.0', config)
        self.assertIn('STAGE_CLOSING=10', status)
        self.assertIn('STAGE_RELEASING=12', status)
        self.assertIn('STAGE_RELEASED=14', status)
        self.assertIn('without lifting', coordinator)

    def test_gripper_open_command_matches_feedback_calibration(self):
        coordinator = (
            CONTROL_ROOT / 'unitree_arm_control' / 'grasp_coordinator.py'
        ).read_text()
        controller = (
            CONTROL_ROOT / 'unitree_arm_control' / 'controller.py'
        ).read_text()
        launch = (CONTROL_ROOT / 'launch' / 'd1_arm.launch.py').read_text()
        self.assertIn("declare_parameter('gripper_open_degrees', 49.0)", coordinator)
        self.assertIn("declare_parameter('gripper_open_degrees', 49.0)", controller)
        self.assertIn("'gripper_open_degrees',\n            default_value='49.0'", launch)

    def test_kdl_solver_uses_full_pose_ik(self):
        kinematics = (
            CONTROL_ROOT / 'config' / 'kinematics.yaml'
        ).read_text()
        self.assertIn('position_only_ik: false', kinematics)

    def test_dog_launch_forwards_execution_safeguard(self):
        launch = (
            LAUNCHER_ROOT / 'launch' / 'dog.launch.py'
        ).read_text()
        self.assertIn("'grasp_execution_enabled'", launch)
        self.assertIn(
            "'grasp_execution_enabled': grasp_execution_enabled",
            launch,
        )

    def test_trajectory_mode_defaults_to_firmware_smoothing(self):
        arm_launch = (
            CONTROL_ROOT / 'launch' / 'd1_arm.launch.py'
        ).read_text()
        dog_launch = (
            LAUNCHER_ROOT / 'launch' / 'dog.launch.py'
        ).read_text()
        controller = (
            CONTROL_ROOT / 'unitree_arm_control' / 'controller.py'
        ).read_text()
        self.assertIn("'trajectory_command_mode',\n            default_value='1'", arm_launch)
        self.assertIn("'arm_trajectory_command_mode',\n            default_value='1'", dog_launch)
        self.assertIn("'trajectory_command_mode', protocol.TRAJECTORY", controller)
        self.assertIn(
            "declare_parameter('trajectory_goal_tolerance_radians', 0.01)",
            controller,
        )
        self.assertIn(
            "'arm_trajectory_goal_tolerance_radians'", dog_launch
        )
        self.assertIn(
            "'arm_trajectory_goal_timeout_sec'", dog_launch
        )


if __name__ == '__main__':
    unittest.main()
