"""Consistency tests for the in-package D1 MoveIt configuration."""

from pathlib import Path
import unittest
from xml.etree import ElementTree


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PACKAGE_ROOT / 'config'
ARM_JOINTS = tuple('d1_joint_{}'.format(index) for index in range(6))


class TestMoveItConfig(unittest.TestCase):
    """Keep the semantic model and controller mapping synchronized."""

    def test_srdf_uses_the_mounted_arm_chain(self):
        root = ElementTree.parse(CONFIG_ROOT / 'd1.srdf').getroot()
        self.assertEqual(root.attrib['name'], 'go2_with_realsense_and_arm')
        arm_group = root.find("./group[@name='d1_arm']")
        self.assertIsNotNone(arm_group)
        chain = arm_group.find('chain')
        self.assertEqual(chain.attrib['base_link'], 'd1_base_link')
        self.assertEqual(chain.attrib['tip_link'], 'd1_gripper_center')

    def test_zero_state_contains_all_arm_joints(self):
        root = ElementTree.parse(CONFIG_ROOT / 'd1.srdf').getroot()
        state = root.find(
            "./group_state[@name='zero'][@group='d1_arm']"
        )
        self.assertIsNotNone(state)
        self.assertEqual(
            tuple(joint.attrib['name'] for joint in state.findall('joint')),
            ARM_JOINTS,
        )

    def test_moveit_controller_targets_existing_action(self):
        configuration = (
            CONFIG_ROOT / 'moveit_controllers.yaml'
        ).read_text()
        self.assertIn('action_ns: follow_joint_trajectory', configuration)
        self.assertIn('type: FollowJointTrajectory', configuration)
        for joint_name in ARM_JOINTS:
            self.assertEqual(configuration.count('- ' + joint_name), 1)

    def test_setup_installs_moveit_configuration(self):
        setup_source = (PACKAGE_ROOT / 'setup.py').read_text()
        self.assertIn("glob('config/*')", setup_source)


if __name__ == '__main__':
    unittest.main()
