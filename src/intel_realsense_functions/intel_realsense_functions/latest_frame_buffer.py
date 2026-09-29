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

"""Thread-safe single-slot storage for the freshest RGB/depth frame pair."""

from dataclasses import dataclass
import threading


@dataclass(frozen=True)
class FramePair:
    """A stable copy of one RGB/depth pair claimed for inference."""

    rgb: object
    depth: object
    camera_info: object
    stamp: object
    stamp_nanoseconds: int
    rgb_receipt: float
    depth_receipt: float
    sequence: int


class LatestFrameBuffer:
    """Keep only the newest camera data and permit one active claim."""

    def __init__(self):
        self._lock = threading.Lock()
        self._rgb = None
        self._depth = None
        self._camera_info = None
        self._rgb_stamp = None
        self._depth_stamp = None
        self._rgb_stamp_nanoseconds = 0
        self._depth_stamp_nanoseconds = 0
        self._rgb_receipt = None
        self._depth_receipt = None
        self._rgb_sequence = 0
        self._last_claimed_sequence = 0
        self._claim_active = False

    @staticmethod
    def _stamp_nanoseconds(stamp):
        if stamp is None:
            return 0
        return (
            int(getattr(stamp, 'sec', 0)) * 1000000000
            + int(getattr(stamp, 'nanosec', 0))
        )

    def update_rgb(self, image, receipt, stamp=None):
        """Replace the stored RGB image with the newest received image."""
        with self._lock:
            self._rgb = image
            self._rgb_receipt = float(receipt)
            self._rgb_stamp = stamp
            self._rgb_stamp_nanoseconds = self._stamp_nanoseconds(stamp)
            self._rgb_sequence += 1

    def update_depth(self, image, receipt, stamp=None):
        """Replace the stored aligned depth image."""
        with self._lock:
            self._depth = image
            self._depth_receipt = float(receipt)
            self._depth_stamp = stamp
            self._depth_stamp_nanoseconds = self._stamp_nanoseconds(stamp)

    def update_camera_info(self, camera_info):
        """Replace the stored camera calibration message."""
        with self._lock:
            self._camera_info = camera_info

    def claim(self, now, maximum_age, maximum_offset):
        """Claim the newest valid pair, returning status details on failure."""
        with self._lock:
            if self._claim_active:
                return None, 'busy', None
            if (
                    self._rgb is None
                    or self._depth is None
                    or self._camera_info is None):
                return None, 'missing', None

            rgb_age = float(now) - self._rgb_receipt
            depth_age = float(now) - self._depth_receipt
            if rgb_age > maximum_age or depth_age > maximum_age:
                return None, 'stale', (rgb_age, depth_age)

            if (
                    self._rgb_stamp_nanoseconds > 0
                    and self._depth_stamp_nanoseconds > 0):
                pair_offset = abs(
                    self._rgb_stamp_nanoseconds
                    - self._depth_stamp_nanoseconds
                ) / 1000000000.0
            else:
                pair_offset = abs(
                    self._rgb_receipt - self._depth_receipt
                )
            if pair_offset > maximum_offset:
                return None, 'unpaired', pair_offset
            if self._rgb_sequence == self._last_claimed_sequence:
                return None, 'unchanged', None

            pair = FramePair(
                rgb=self._rgb.copy(),
                depth=self._depth.copy(),
                camera_info=self._camera_info,
                stamp=self._rgb_stamp,
                stamp_nanoseconds=self._rgb_stamp_nanoseconds,
                rgb_receipt=self._rgb_receipt,
                depth_receipt=self._depth_receipt,
                sequence=self._rgb_sequence,
            )
            self._last_claimed_sequence = self._rgb_sequence
            self._claim_active = True
            return pair, 'ready', None

    def release(self):
        """Allow the newest pair to be claimed after inference completes."""
        with self._lock:
            self._claim_active = False
