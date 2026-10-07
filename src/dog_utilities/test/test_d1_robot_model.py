"""Regression tests for the mounted Unitree D1 robot description."""

from pathlib import Path
import unittest
from xml.etree import ElementTree


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / 'urdf'
    / 'go2_with_realsense_and_arm.urdf'
)


class TestD1RobotModel(unittest.TestCase):
    """Keep the D1 SDK convention and grasp frame physically consistent."""

    @classmethod
    def setUpClass(cls):
        cls.root = ElementTree.parse(MODEL_PATH).getroot()

    def _joint(self, name):
        joint = self.root.find("./joint[@name='{}']".format(name))
        self.assertIsNotNone(joint)
        return joint

    def test_arm_axes_match_sdk_positive_directions(self):
        expected = {
            'd1_joint_0': '0 0 -1',
            'd1_joint_1': '0 0 -1',
            'd1_joint_2': '0 0 -1',
            'd1_joint_3': '0 0 -1',
            'd1_joint_4': '0 0 -1',
            'd1_joint_5': '0 0 -1',
        }
        for name, axis in expected.items():
            with self.subTest(joint=name):
                self.assertEqual(
                    self._joint(name).find('axis').attrib['xyz'],
                    axis,
                )

    def test_gripper_center_is_fingertip_tcp(self):
        joint = self._joint('d1_gripper_center_joint')
        self.assertEqual(joint.find('parent').attrib['link'], 'd1_link_6')
        self.assertEqual(
            joint.find('child').attrib['link'],
            'd1_gripper_center',
        )
        self.assertEqual(
            joint.find('origin').attrib['xyz'],
            '0.00038 0 0.1256',
        )

    def test_closed_gripper_origins_have_zero_nominal_aperture(self):
        self.assertEqual(
            self._joint('d1_gripper_joint').find('origin').attrib['xyz'],
            '-0.0056012 -0.02100301 0.0706',
        )
        self.assertEqual(
            self._joint('d1_gripper_mimic_joint')
            .find('origin').attrib['xyz'],
            '-0.0056388 0.02101013 0.0706',
        )


if __name__ == '__main__':
    unittest.main()
