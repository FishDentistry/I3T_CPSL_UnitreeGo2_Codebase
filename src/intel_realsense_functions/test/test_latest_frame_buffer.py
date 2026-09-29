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

"""Tests for the single-slot Grounding DINO camera-frame buffer."""

import unittest

import numpy as np

from intel_realsense_functions.latest_frame_buffer import LatestFrameBuffer


class LatestFrameBufferTest(unittest.TestCase):
    """Verify that inference always claims the newest available pair."""

    def _complete_pair(self, buffer, value, receipt):
        buffer.update_rgb(
            np.full((2, 2, 3), value, dtype=np.uint8), receipt
        )
        buffer.update_depth(
            np.full((2, 2), value, dtype=np.float32), receipt + 0.01
        )
        buffer.update_camera_info(object())

    def test_new_frames_replace_pending_frames_while_busy(self):
        """Frames arriving during inference replace rather than queue."""
        buffer = LatestFrameBuffer()
        self._complete_pair(buffer, 1, 10.0)

        first, status, _ = buffer.claim(10.1, 1.0, 0.25)
        self.assertEqual(status, 'ready')
        self.assertEqual(int(first.rgb[0, 0, 0]), 1)

        self._complete_pair(buffer, 2, 10.2)
        self._complete_pair(buffer, 3, 10.3)
        claimed, status, _ = buffer.claim(10.4, 1.0, 0.25)
        self.assertIsNone(claimed)
        self.assertEqual(status, 'busy')

        buffer.release()
        newest, status, _ = buffer.claim(10.4, 1.0, 0.25)
        self.assertEqual(status, 'ready')
        self.assertEqual(int(newest.rgb[0, 0, 0]), 3)
        self.assertEqual(float(newest.depth[0, 0]), 3.0)

    def test_claim_rejects_stale_and_unpaired_frames(self):
        """Age and pair-offset safeguards remain active."""
        buffer = LatestFrameBuffer()
        self._complete_pair(buffer, 1, 10.0)

        claimed, status, ages = buffer.claim(12.0, 1.0, 0.25)
        self.assertIsNone(claimed)
        self.assertEqual(status, 'stale')
        self.assertGreater(ages[0], 1.0)

        buffer.update_rgb(np.zeros((2, 2, 3)), 20.0)
        buffer.update_depth(np.zeros((2, 2)), 20.5)
        claimed, status, offset = buffer.claim(20.6, 1.0, 0.25)
        self.assertIsNone(claimed)
        self.assertEqual(status, 'unpaired')
        self.assertAlmostEqual(offset, 0.5)


if __name__ == '__main__':
    unittest.main()
