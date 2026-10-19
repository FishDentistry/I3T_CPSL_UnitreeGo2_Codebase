"""Static wiring checks for the RealSense publication timer."""

from pathlib import Path
import unittest


SOURCE_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = SOURCE_ROOT / 'intel_realsense_functions'


class CameraTimerConfigTest(unittest.TestCase):
    """Keep the camera node parameter and dog launch argument synchronized."""

    def test_dog_launch_forwards_camera_timer_period(self):
        node = (
            PACKAGE_ROOT / 'intel_realsense_functions' / 'getCameraFrames.py'
        ).read_text()
        launch = (
            SOURCE_ROOT / 'go2_launcher' / 'launch' / 'dog.launch.py'
        ).read_text()

        self.assertIn(
            "declare_parameter('timer_callback_period_sec', 0.1)", node
        )
        self.assertIn(
            "LaunchConfiguration(\n        'realsense_timer_period_sec'",
            launch,
        )
        self.assertIn(
            "'realsense_timer_period_sec',\n            default_value='0.1'",
            launch,
        )
        self.assertIn("'timer_callback_period_sec': ParameterValue(", launch)
        self.assertIn('value_type=float', launch)


if __name__ == '__main__':
    unittest.main()
