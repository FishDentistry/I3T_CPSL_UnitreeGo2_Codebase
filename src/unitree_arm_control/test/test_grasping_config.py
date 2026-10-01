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
        self.assertIn("executable='d1_grasp_coordinator'", launch)
        self.assertIn("'grasp_execution_enabled'", launch)
        self.assertIn("'execution_enabled': grasp_execution", launch)

    def test_reach_safeguard_uses_arm_base_frame(self):
        config = (CONTROL_ROOT / 'config' / 'grasping.yaml').read_text()
        coordinator = (
            CONTROL_ROOT / 'unitree_arm_control' / 'grasp_coordinator.py'
        ).read_text()
        self.assertIn('reach_reference_frame: d1_base_link', config)
        self.assertIn(
            "declare_parameter('reach_reference_frame', 'd1_base_link')",
            coordinator,
        )

    def test_dog_launch_forwards_execution_safeguard(self):
        launch = (
            LAUNCHER_ROOT / 'launch' / 'dog.launch.py'
        ).read_text()
        self.assertIn("'grasp_execution_enabled'", launch)
        self.assertIn(
            "'grasp_execution_enabled': grasp_execution_enabled",
            launch,
        )


if __name__ == '__main__':
    unittest.main()
