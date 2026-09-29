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

"""Tests for semantic association, fusion, movement, and lifecycle logic."""

import unittest

from semantic_mapping.semantic_tracker import ACTIVE
from semantic_mapping.semantic_tracker import Observation
from semantic_mapping.semantic_tracker import REMOVED
from semantic_mapping.semantic_tracker import SemanticTracker
from semantic_mapping.semantic_tracker import STALE
from semantic_mapping.semantic_tracker import TrackerConfig


def observation(label, x, timestamp, confidence=0.9):
    """Create a simple map-frame observation for a test."""
    return Observation(label, (x, 0.0, 0.5), confidence, timestamp)


def test_config(**overrides):
    """Return short test timeouts while retaining production relationships."""
    values = {
        'minimum_confidence': 0.5,
        'confirmation_observations': 3,
        'observation_merge_distance': 0.2,
        'association_distance': 0.75,
        'position_fusion_alpha': 0.25,
        'movement_distance': 0.35,
        'movement_cluster_distance': 0.25,
        'movement_confirmation_observations': 2,
        'relocation_minimum_age': 2.0,
        'relocation_maximum_distance': 3.0,
        'tentative_timeout': 2.0,
        'stale_after': 2.0,
        'removed_after': 5.0,
        'removed_retention': 3.0,
    }
    values.update(overrides)
    return TrackerConfig(**values)


class SemanticTrackerTest(unittest.TestCase):
    """Verify the externally meaningful semantic-map behaviors."""

    def test_rejects_low_confidence_and_temporary_detections(self):
        tracker = SemanticTracker(test_config())

        summary = tracker.update(
            [observation('cup', 0.0, 0.0, confidence=0.2)], 0.0
        )
        self.assertEqual(summary.rejected_low_confidence, 1)
        self.assertEqual(tracker.snapshot(), ())

        tracker.update([observation('cup', 0.0, 0.1)], 0.1)
        self.assertEqual(tracker.snapshot(), ())
        tracker.refresh(2.2)
        self.assertEqual(tracker.snapshot(), ())

    def test_merges_frame_duplicates_and_confirms_one_identity(self):
        tracker = SemanticTracker(test_config())
        first = tracker.update([
            observation('Cup', 0.00, 0.0, confidence=0.8),
            observation(' cup ', 0.05, 0.0, confidence=0.9),
        ], 0.0)

        self.assertEqual(first.merged_in_frame, 1)
        self.assertEqual(tracker.snapshot(), ())
        tracker.update([observation('cup', 0.04, 0.5)], 0.5)
        tracker.update([observation('cup', 0.02, 1.0)], 1.0)

        tracks = tracker.snapshot()
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].object_id, 'object_000001')
        self.assertEqual(tracks[0].observation_count, 3)
        self.assertEqual(tracks[0].state, ACTIVE)

    def test_keeps_separate_nearby_objects_with_the_same_label(self):
        tracker = SemanticTracker(test_config(
            confirmation_observations=2,
            observation_merge_distance=0.1,
        ))
        tracker.update([
            observation('cup', 0.0, 0.0),
            observation('cup', 1.5, 0.0),
        ], 0.0)
        tracker.update([
            observation('cup', 0.05, 0.5),
            observation('cup', 1.55, 0.5),
        ], 0.5)

        tracks = tracker.snapshot()
        self.assertEqual(len(tracks), 2)
        self.assertNotEqual(tracks[0].object_id, tracks[1].object_id)

    def test_fuses_position_without_jumping_to_each_measurement(self):
        tracker = SemanticTracker(test_config(confirmation_observations=2))
        tracker.update([observation('bottle', 0.0, 0.0)], 0.0)
        tracker.update([observation('bottle', 0.0, 0.5)], 0.5)
        tracker.update([observation('bottle', 0.2, 1.0)], 1.0)

        track = tracker.snapshot()[0]
        self.assertGreater(track.position[0], 0.0)
        self.assertLess(track.position[0], 0.2)

    def test_requires_consistent_observations_before_moving_an_object(self):
        tracker = SemanticTracker(test_config(confirmation_observations=2))
        tracker.update([observation('chair', 0.0, 0.0)], 0.0)
        tracker.update([observation('chair', 0.0, 0.5)], 0.5)
        original = tracker.snapshot()[0]

        tracker.update([observation('chair', 0.6, 1.0)], 1.0)
        pending = tracker.snapshot()[0]
        self.assertEqual(pending.object_id, original.object_id)
        self.assertAlmostEqual(pending.position[0], 0.0)

        summary = tracker.update(
            [observation('chair', 0.62, 1.5)], 1.5
        )
        moved = tracker.snapshot()[0]
        self.assertEqual(summary.moved, 1)
        self.assertEqual(moved.object_id, original.object_id)
        self.assertAlmostEqual(moved.position[0], 0.61)
        self.assertEqual(moved.movement_count, 1)

    def test_relocates_a_unique_unseen_object_without_changing_its_id(self):
        tracker = SemanticTracker(test_config(confirmation_observations=2))
        tracker.update([observation('cup', 0.0, 0.0)], 0.0)
        tracker.update([observation('cup', 0.0, 0.5)], 0.5)
        stable_id = tracker.snapshot()[0].object_id

        tracker.update([observation('cup', 2.0, 3.0)], 3.0)
        summary = tracker.update([observation('cup', 2.1, 3.5)], 3.5)

        tracks = tracker.snapshot()
        self.assertEqual(summary.relocated, 1)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].object_id, stable_id)
        self.assertAlmostEqual(tracks[0].position[0], 2.05)

    def test_transitions_from_active_to_stale_removed_and_pruned(self):
        tracker = SemanticTracker(test_config(confirmation_observations=2))
        tracker.update([observation('table', 0.0, 0.0)], 0.0)
        tracker.update([observation('table', 0.0, 0.5)], 0.5)

        tracker.refresh(2.5)
        self.assertEqual(tracker.snapshot()[0].state, STALE)
        tracker.refresh(5.5)
        self.assertEqual(tracker.snapshot()[0].state, REMOVED)
        tracker.refresh(8.5)
        self.assertEqual(tracker.snapshot(), ())


if __name__ == '__main__':
    unittest.main()
