# Copyright 2026 CPSL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for Grounding DINO target parsing and coordinate calculations."""

import math
from pathlib import Path
import unittest

import numpy as np

from intel_realsense_functions.detection_geometry import (
    camera_point_from_depth,
)
from intel_realsense_functions.detection_geometry import (
    narrow_grasp_band_from_depth,
)
from intel_realsense_functions.detection_geometry import (
    optical_point_to_camera_link,
)
from intel_realsense_functions.detection_geometry import (
    parse_detection_targets,
)
from intel_realsense_functions.detection_geometry import transform_point


class DetectionGeometryTest(unittest.TestCase):
    """Exercise perception helpers independently from ROS and torch."""

    def test_parse_plain_and_json_targets(self):
        """Target messages accept documented plain-text and JSON forms."""
        self.assertEqual(
            parse_detection_targets(' Cup, red bottle, cup '),
            ['cup', 'red bottle'],
        )
        self.assertEqual(
            parse_detection_targets('["cup", "red bottle"]'),
            ['cup', 'red bottle'],
        )
        self.assertEqual(
            parse_detection_targets('{"targets": ["cup"]}'),
            ['cup'],
        )

    def test_project_constant_depth_at_principal_point(self):
        """A principal-point depth sample lies on the optical Z axis."""
        depth = np.full((5, 5), 2.0, dtype=np.float32)
        point = camera_point_from_depth(
            depth,
            (1, 1, 3, 3),
            (100.0, 100.0, 2.0, 2.0),
            0.1,
            5.0,
            1.0,
        )
        self.assertAlmostEqual(point['x'], 0.0)
        self.assertAlmostEqual(point['y'], 0.0)
        self.assertAlmostEqual(point['z'], 2.0)
        self.assertGreater(point['depth_sample_count'], 0)

    def test_invalid_depth_returns_no_point(self):
        """Invalid or out-of-range depth does not produce coordinates."""
        depth = np.zeros((5, 5), dtype=np.float32)
        self.assertIsNone(camera_point_from_depth(
            depth,
            (1, 1, 3, 3),
            (100.0, 100.0, 2.0, 2.0),
            0.1,
            5.0,
        ))

    def test_narrow_grasp_band_uses_supported_object_neck(self):
        """A persistent narrow depth silhouette beats the wider body."""
        depth = np.full((100, 100), 2.0, dtype=np.float32)
        depth[20:80, 35:65] = 1.0
        depth[20:50, 35:65] = 2.0
        depth[20:50, 42:58] = 1.0
        reference = {
            'z': 1.0, 'pixel_u': 50.0,
        }
        band = narrow_grasp_band_from_depth(
            depth, (30, 15, 70, 85),
            (1000.0, 1000.0, 50.0, 50.0),
            reference, 0.1, 3.0,
        )
        self.assertIsNotNone(band)
        self.assertAlmostEqual(band['width_m'], 0.016, places=3)
        self.assertLess(band['pixel_v'], 50.0)

    def test_narrow_grasp_band_rejects_unreliable_depth(self):
        depth = np.full((100, 100), 2.0, dtype=np.float32)
        reference = {'z': 1.0, 'pixel_u': 50.0}
        self.assertIsNone(narrow_grasp_band_from_depth(
            depth, (30, 15, 70, 85),
            (1000.0, 1000.0, 50.0, 50.0),
            reference, 0.1, 3.0,
        ))

    def test_optical_point_is_converted_to_camera_link_axes(self):
        """RealSense right/down/forward maps to link forward/left/up."""
        point = optical_point_to_camera_link({
            'x': 0.2,
            'y': -0.1,
            'z': 2.0,
        })
        self.assertEqual(point, {
            'x': 2.0,
            'y': -0.2,
            'z': 0.1,
        })

    def test_detector_and_mapping_launch_default_to_camera_link(self):
        """The converted coordinates and TF source frame stay synchronized."""
        package_root = Path(__file__).resolve().parents[1]
        detector = (
            package_root / 'intel_realsense_functions'
            / 'groundingDinoNode.py'
        ).read_text()
        mapping_launch = (
            package_root.parent / 'semantic_mapping' / 'launch'
            / 'semantic_mapping.launch.py'
        ).read_text()
        expected = "default_value='camera_link'"
        self.assertIn(
            "declare_parameter('camera_frame', 'camera_link')", detector
        )
        self.assertIn(
            'camera_point = optical_point_to_camera_link(optical_point)',
            detector,
        )
        self.assertIn('_map_point(\n                    camera_point', detector)
        self.assertIn(expected, mapping_launch)

    def test_transform_point_rotates_and_translates(self):
        """Quaternion transforms follow geometry_msgs XYZW ordering."""
        half_sqrt = math.sqrt(0.5)
        result = transform_point(
            (1.0, 0.0, 0.0),
            (1.0, 2.0, 3.0),
            (0.0, 0.0, half_sqrt, half_sqrt),
        )
        self.assertAlmostEqual(result[0], 1.0)
        self.assertAlmostEqual(result[1], 3.0)
        self.assertAlmostEqual(result[2], 3.0)


if __name__ == '__main__':
    unittest.main()
