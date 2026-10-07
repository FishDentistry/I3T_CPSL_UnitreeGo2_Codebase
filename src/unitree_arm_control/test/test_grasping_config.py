"""Static consistency tests for the semantic grasp coordinator."""

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
        self.assertIn('maximum_reach_m: 0.67', config)
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
        self.assertIn(
            'class_forward_grasp_depth_offsets_m: ["mug=0.04"]', config
        )
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
        self.assertNotIn('GetPositionFK', coordinator)
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
        self.assertIn("if purpose == 'approach':", coordinator)
        self.assertIn('self._restart_candidate_screening(description)', coordinator)

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
        self.assertIn("'pregrasp',", coordinator)
        self.assertIn(
            'executing orientation-tolerant pre-grasp candidate',
            coordinator,
        )
        self.assertNotIn('GetPositionIK', coordinator)
        self.assertNotIn('tool_roll_offsets_rad:', config)
        self.assertNotIn("'validated_candidates': []", coordinator)

    def test_primary_approach_uses_current_gripper_position(self):
        coordinator = (
            CONTROL_ROOT / 'unitree_arm_control' / 'grasp_coordinator.py'
        ).read_text()
        self.assertIn(
            'self._planning_frame, self._tip_link, Time()', coordinator
        )
        self.assertIn(
            'object_point, approach_origin,', coordinator
        )

    def test_gripper_sequence_holds_releases_and_does_not_lift(self):
        config = (CONTROL_ROOT / 'config' / 'grasping.yaml').read_text()
        coordinator = (
            CONTROL_ROOT / 'unitree_arm_control' / 'grasp_coordinator.py'
        ).read_text()
        status = (INTERFACE_ROOT / 'msg' / 'GraspStatus.msg').read_text()
        self.assertIn('hold_duration_sec: 3.0', config)
        self.assertIn('gripper_service: /d1_arm_controller/set_joint', config)
        self.assertIn('gripper_open_degrees: 50.0', config)
        self.assertIn('STAGE_CLOSING=10', status)
        self.assertIn('STAGE_RELEASING=12', status)
        self.assertIn('STAGE_RELEASED=14', status)
        self.assertIn('without lifting', coordinator)

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


if __name__ == '__main__':
    unittest.main()
