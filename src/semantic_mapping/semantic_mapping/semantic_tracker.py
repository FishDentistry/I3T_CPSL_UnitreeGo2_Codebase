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

"""ROS-independent semantic object association and position fusion."""

from dataclasses import dataclass, replace
import math


TENTATIVE = 'tentative'
ACTIVE = 'active'
STALE = 'stale'
REMOVED = 'removed'


def normalize_label(label):
    """Return a stable comparison form without changing the display label."""
    return ' '.join(str(label).strip().lower().split())


@dataclass(frozen=True)
class Observation:
    """One map-frame object observation."""

    label: str
    position: tuple
    confidence: float
    timestamp: float


@dataclass
class TrackerConfig:
    """Association, confirmation, lifecycle, and fusion thresholds."""

    minimum_confidence: float = 0.50
    confirmation_observations: int = 3
    observation_merge_distance: float = 0.20
    association_distance: float = 0.75
    position_fusion_alpha: float = 0.25
    movement_distance: float = 0.35
    movement_cluster_distance: float = 0.30
    movement_confirmation_observations: int = 3
    relocation_minimum_age: float = 5.0
    relocation_maximum_distance: float = 3.0
    tentative_timeout: float = 10.0
    stale_after: float = 30.0
    removed_after: float = 300.0
    removed_retention: float = 300.0

    def validate(self):
        """Raise ValueError when thresholds cannot produce valid behavior."""
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError('minimum_confidence must be in [0, 1]')
        if self.confirmation_observations < 1:
            raise ValueError('confirmation_observations must be positive')
        if self.movement_confirmation_observations < 1:
            raise ValueError(
                'movement_confirmation_observations must be positive'
            )
        positive = (
            'observation_merge_distance',
            'association_distance',
            'position_fusion_alpha',
            'movement_distance',
            'movement_cluster_distance',
            'relocation_maximum_distance',
            'tentative_timeout',
            'stale_after',
            'removed_after',
            'removed_retention',
        )
        for name in positive:
            if float(getattr(self, name)) <= 0.0:
                raise ValueError('{} must be positive'.format(name))
        if self.position_fusion_alpha > 1.0:
            raise ValueError('position_fusion_alpha must not exceed 1')
        if self.relocation_minimum_age < 0.0:
            raise ValueError('relocation_minimum_age must not be negative')
        if self.stale_after >= self.removed_after:
            raise ValueError('stale_after must be less than removed_after')


@dataclass
class Track:
    """Internal state for one tentative or confirmed semantic object."""

    numeric_id: int
    object_id: str
    label: str
    normalized_label: str
    position: tuple
    confidence: float
    observation_count: int
    first_observed: float
    last_observed: float
    state: str = TENTATIVE
    confirmed: bool = False
    movement_count: int = 0
    fusion_weight: float = 0.0
    movement_position: tuple = None
    movement_confidence: float = 0.0
    movement_observations: int = 0


@dataclass
class UpdateSummary:
    """Counts describing how an input frame changed the tracker."""

    changed: bool = False
    accepted: int = 0
    rejected_low_confidence: int = 0
    rejected_invalid: int = 0
    merged_in_frame: int = 0
    created: int = 0
    confirmed: int = 0
    moved: int = 0
    relocated: int = 0


class SemanticTracker:
    """Associate detections into stable, lifecycle-aware object tracks."""

    def __init__(self, config=None):
        self.config = config or TrackerConfig()
        self.config.validate()
        self._tracks = {}
        self._next_id = 1
        self.revision = 0

    @staticmethod
    def _distance(first, second):
        return math.sqrt(sum(
            (float(left) - float(right)) ** 2
            for left, right in zip(first, second)
        ))

    @staticmethod
    def _valid_observation(observation):
        return (
            normalize_label(observation.label) != ''
            and len(observation.position) == 3
            and all(
                math.isfinite(float(value))
                for value in observation.position
            )
            and math.isfinite(float(observation.confidence))
            and math.isfinite(float(observation.timestamp))
        )

    @staticmethod
    def _weighted_position(first, first_weight, second, second_weight):
        total = float(first_weight) + float(second_weight)
        if total <= 0.0:
            return tuple(float(value) for value in second)
        return tuple(
            (
                float(left) * float(first_weight)
                + float(right) * float(second_weight)
            ) / total
            for left, right in zip(first, second)
        )

    def _coalesce_frame_observations(self, observations, summary):
        """Merge same-label points that are duplicates within one frame."""
        clusters = []
        for observation in sorted(
                observations, key=lambda item: item.confidence, reverse=True):
            normalized = normalize_label(observation.label)
            match = None
            for cluster in clusters:
                if cluster['normalized_label'] != normalized:
                    continue
                if self._distance(
                        cluster['position'], observation.position
                        ) <= self.config.observation_merge_distance:
                    match = cluster
                    break
            if match is None:
                clusters.append({
                    'label': observation.label.strip(),
                    'normalized_label': normalized,
                    'position': tuple(float(v) for v in observation.position),
                    'confidence': float(observation.confidence),
                    'weight': float(observation.confidence),
                    'timestamp': float(observation.timestamp),
                })
                continue

            weight = max(float(observation.confidence), 0.000001)
            match['position'] = self._weighted_position(
                match['position'],
                match['weight'],
                observation.position,
                weight,
            )
            match['weight'] += weight
            match['confidence'] = max(
                match['confidence'], float(observation.confidence)
            )
            match['timestamp'] = max(
                match['timestamp'], float(observation.timestamp)
            )
            summary.merged_in_frame += 1

        return [
            Observation(
                label=cluster['label'],
                position=cluster['position'],
                confidence=cluster['confidence'],
                timestamp=cluster['timestamp'],
            )
            for cluster in clusters
        ]

    def _new_track(self, observation):
        numeric_id = self._next_id
        self._next_id += 1
        confidence = float(observation.confidence)
        track = Track(
            numeric_id=numeric_id,
            object_id='object_{:06d}'.format(numeric_id),
            label=observation.label.strip(),
            normalized_label=normalize_label(observation.label),
            position=tuple(float(value) for value in observation.position),
            confidence=confidence,
            observation_count=1,
            first_observed=float(observation.timestamp),
            last_observed=float(observation.timestamp),
            fusion_weight=max(confidence, 0.000001),
        )
        self._tracks[numeric_id] = track
        return track

    def _update_tentative(self, track, observation):
        weight = max(float(observation.confidence), 0.000001)
        track.position = self._weighted_position(
            track.position,
            track.fusion_weight,
            observation.position,
            weight,
        )
        track.fusion_weight += weight
        track.confidence = max(track.confidence, float(observation.confidence))
        track.observation_count += 1
        track.last_observed = float(observation.timestamp)

    def _update_confirmed(self, track, observation, summary):
        distance = self._distance(track.position, observation.position)
        track.last_observed = float(observation.timestamp)
        track.observation_count += 1
        track.state = ACTIVE

        if distance > self.config.movement_distance:
            if (
                    track.movement_position is None
                    or self._distance(
                        track.movement_position, observation.position
                    ) > self.config.movement_cluster_distance):
                track.movement_position = tuple(
                    float(value) for value in observation.position
                )
                track.movement_confidence = float(observation.confidence)
                track.movement_observations = 1
            else:
                count = track.movement_observations
                track.movement_position = self._weighted_position(
                    track.movement_position,
                    count,
                    observation.position,
                    1.0,
                )
                track.movement_confidence = max(
                    track.movement_confidence,
                    float(observation.confidence),
                )
                track.movement_observations += 1

            if (
                    track.movement_observations
                    >= self.config.movement_confirmation_observations):
                track.position = track.movement_position
                track.confidence = track.movement_confidence
                track.movement_count += 1
                track.movement_position = None
                track.movement_confidence = 0.0
                track.movement_observations = 0
                summary.moved += 1
            return

        track.movement_position = None
        track.movement_confidence = 0.0
        track.movement_observations = 0
        alpha = min(
            self.config.position_fusion_alpha,
            max(
                0.01,
                self.config.position_fusion_alpha
                * float(observation.confidence),
            ),
        )
        track.position = tuple(
            (1.0 - alpha) * float(old) + alpha * float(new)
            for old, new in zip(track.position, observation.position)
        )
        track.confidence = (
            (1.0 - alpha) * track.confidence
            + alpha * float(observation.confidence)
        )

    def _confirm_or_relocate(self, track, timestamp, matched_ids, summary):
        candidates = [
            existing
            for existing in self._tracks.values()
            if (
                existing.numeric_id != track.numeric_id
                and existing.confirmed
                and existing.normalized_label == track.normalized_label
                and existing.numeric_id not in matched_ids
                and timestamp - existing.last_observed
                >= self.config.relocation_minimum_age
                and self._distance(existing.position, track.position)
                <= self.config.relocation_maximum_distance
            )
        ]
        if len(candidates) == 1:
            existing = candidates[0]
            existing.position = track.position
            existing.confidence = track.confidence
            existing.observation_count += track.observation_count
            existing.last_observed = track.last_observed
            existing.state = ACTIVE
            existing.movement_count += 1
            existing.movement_position = None
            existing.movement_confidence = 0.0
            existing.movement_observations = 0
            del self._tracks[track.numeric_id]
            matched_ids.add(existing.numeric_id)
            summary.relocated += 1
            return existing

        track.confirmed = True
        track.state = ACTIVE
        summary.confirmed += 1
        return track

    def refresh(self, timestamp):
        """Advance lifecycle states and prune expired internal tracks."""
        timestamp = float(timestamp)
        changed = False
        expired = []
        for numeric_id, track in self._tracks.items():
            age = max(0.0, timestamp - track.last_observed)
            if not track.confirmed:
                if age >= self.config.tentative_timeout:
                    expired.append(numeric_id)
                    changed = True
                continue

            previous = track.state
            if age >= (
                    self.config.removed_after
                    + self.config.removed_retention):
                expired.append(numeric_id)
            elif age >= self.config.removed_after:
                track.state = REMOVED
            elif age >= self.config.stale_after:
                track.state = STALE
            else:
                track.state = ACTIVE
            changed = changed or previous != track.state

        for numeric_id in expired:
            del self._tracks[numeric_id]
        if changed:
            self.revision += 1
        return changed

    def update(self, observations, timestamp):
        """Associate one detection frame and update the semantic map."""
        timestamp = float(timestamp)
        summary = UpdateSummary()
        if self.refresh(timestamp):
            summary.changed = True

        accepted = []
        for observation in observations:
            if not self._valid_observation(observation):
                summary.rejected_invalid += 1
                continue
            if float(observation.confidence) < self.config.minimum_confidence:
                summary.rejected_low_confidence += 1
                continue
            accepted.append(observation)
        accepted = self._coalesce_frame_observations(accepted, summary)
        summary.accepted = len(accepted)

        matched_ids = set()
        for observation in sorted(
                accepted, key=lambda item: item.confidence, reverse=True):
            normalized = normalize_label(observation.label)
            candidates = [
                track
                for track in self._tracks.values()
                if (
                    track.state != REMOVED
                    and track.normalized_label == normalized
                    and track.numeric_id not in matched_ids
                    and observation.timestamp >= track.last_observed
                    and self._distance(track.position, observation.position)
                    <= self.config.association_distance
                )
            ]
            if candidates:
                track = min(
                    candidates,
                    key=lambda item: self._distance(
                        item.position, observation.position
                    ),
                )
                if track.confirmed:
                    self._update_confirmed(track, observation, summary)
                else:
                    self._update_tentative(track, observation)
                matched_ids.add(track.numeric_id)
            else:
                track = self._new_track(observation)
                matched_ids.add(track.numeric_id)
                summary.created += 1

            if (
                    not track.confirmed
                    and track.observation_count
                    >= self.config.confirmation_observations
                    and track.numeric_id in self._tracks):
                track = self._confirm_or_relocate(
                    track, timestamp, matched_ids, summary
                )

        if accepted or summary.changed:
            summary.changed = True
            self.revision += 1
        return summary

    def snapshot(self, include_removed=True):
        """Return detached, confirmed tracks ordered by stable numeric ID."""
        return tuple(
            replace(track)
            for track in sorted(
                self._tracks.values(), key=lambda item: item.numeric_id
            )
            if (
                track.confirmed
                and (include_removed or track.state != REMOVED)
            )
        )
