"""Validation and interpolation helpers for D1 joint trajectories."""

import math


_POSITION_LIMIT_EPSILON = 1.0e-6


class TrajectoryError(ValueError):
    """Raised when a trajectory cannot be executed safely."""


def duration_seconds(duration):
    """Convert a builtin_interfaces/Duration-like value to seconds."""
    return float(duration.sec) + float(duration.nanosec) * 1e-9


def validate_trajectory(joint_names, points, required_joint_names):
    """Validate a positional trajectory and return waypoint times."""
    names = tuple(joint_names)
    required = tuple(required_joint_names)
    if not names:
        raise TrajectoryError('trajectory joint_names must not be empty')
    if len(set(names)) != len(names):
        raise TrajectoryError('trajectory joint_names contains duplicates')

    unknown = sorted(set(names) - set(required))
    if unknown:
        raise TrajectoryError(
            'trajectory contains unknown joints: {}'.format(
                ', '.join(unknown)
            )
        )

    missing = [name for name in required[:6] if name not in names]
    if missing:
        raise TrajectoryError(
            'trajectory must command all six arm joints; missing: {}'.format(
                ', '.join(missing)
            )
        )
    if not points:
        raise TrajectoryError('trajectory must contain at least one point')

    times = []
    previous_time = -1.0
    for index, point in enumerate(points):
        if len(point.positions) != len(names):
            raise TrajectoryError(
                'point {} has {} positions for {} joints'.format(
                    index, len(point.positions), len(names)
                )
            )
        if not all(math.isfinite(value) for value in point.positions):
            raise TrajectoryError(
                'point {} contains a non-finite position'.format(index)
            )
        point_time = duration_seconds(point.time_from_start)
        if point_time < 0.0:
            raise TrajectoryError(
                'point {} has a negative time_from_start'.format(index)
            )
        if point_time <= previous_time:
            raise TrajectoryError(
                'trajectory point times must be strictly increasing'
            )
        times.append(point_time)
        previous_time = point_time
    return tuple(times)


def expand_positions(
        commanded_joint_names,
        commanded_positions,
        all_joint_names,
        fallback_positions):
    """Expand a partial joint vector in canonical D1 joint order."""
    if len(fallback_positions) != len(all_joint_names):
        raise TrajectoryError('fallback joint vector has the wrong length')
    by_name = dict(zip(commanded_joint_names, commanded_positions))
    return tuple(
        by_name.get(name, fallback_positions[index])
        for index, name in enumerate(all_joint_names)
    )


def interpolate_positions(
        elapsed, waypoint_times, waypoint_positions, start_positions):
    """Linearly sample a position trajectory at ``elapsed`` seconds."""
    if not waypoint_times or len(waypoint_times) != len(waypoint_positions):
        raise TrajectoryError('waypoint times and positions must match')
    if len(start_positions) != len(waypoint_positions[0]):
        raise TrajectoryError('start position vector has the wrong length')

    if elapsed >= waypoint_times[-1]:
        return tuple(waypoint_positions[-1])

    previous_time = 0.0
    previous_positions = tuple(start_positions)
    for point_time, positions in zip(
            waypoint_times, waypoint_positions):
        positions = tuple(positions)
        if elapsed <= point_time:
            interval = point_time - previous_time
            if interval <= 0.0:
                return positions
            fraction = max(
                0.0, min(1.0, (elapsed - previous_time) / interval)
            )
            return tuple(
                start + fraction * (end - start)
                for start, end in zip(previous_positions, positions)
            )
        previous_time = point_time
        previous_positions = positions

    return tuple(waypoint_positions[-1])


def validate_segment_velocities(
        waypoint_times,
        waypoint_positions,
        start_positions,
        joint_names,
        velocity_limits,
        zero_time_start_tolerances=None):
    """Reject segments whose average velocity exceeds a joint limit."""
    if len(joint_names) != len(start_positions):
        raise TrajectoryError('joint and start position counts must match')
    if len(velocity_limits) != len(joint_names):
        raise TrajectoryError('joint and velocity limit counts must match')
    if (
            zero_time_start_tolerances is not None
            and len(zero_time_start_tolerances) != len(joint_names)):
        raise TrajectoryError(
            'joint and zero-time start tolerance counts must match'
        )

    previous_time = 0.0
    previous_positions = tuple(start_positions)
    for point_index, (point_time, positions) in enumerate(zip(
            waypoint_times, waypoint_positions)):
        duration = point_time - previous_time
        positions = tuple(positions)
        for joint_index, (start, end, limit) in enumerate(zip(
                previous_positions, positions, velocity_limits)):
            movement = abs(end - start)
            if duration <= 0.0 and movement > 0.0:
                if (
                        point_index == 0
                        and zero_time_start_tolerances is not None
                        and movement <= zero_time_start_tolerances[
                            joint_index
                        ]):
                    continue
                raise TrajectoryError(
                    'point {} moves {} with no time available'.format(
                        point_index, joint_names[joint_index]
                    )
                )
            if duration > 0.0 and movement / duration > limit + 1e-9:
                raise TrajectoryError(
                    'point {} exceeds the velocity limit for {}'.format(
                        point_index, joint_names[joint_index]
                    )
                )
        previous_time = point_time
        previous_positions = positions


def validate_position_limits(
        waypoint_positions, start_positions, joint_names, position_limits):
    """Validate limits while allowing an out-of-range joint toward safety.

    A measured starting position can be marginally outside a published limit.
    Such a joint may remain at that position or move toward the valid range,
    but it may not move farther outside the range. Tiny floating-point
    overshoots at the boundary are treated as numerical noise rather than a real
    limit violation.
    """
    if len(joint_names) != len(start_positions):
        raise TrajectoryError('joint and start position counts must match')
    if len(position_limits) != len(joint_names):
        raise TrajectoryError('joint and position limit counts must match')

    epsilon = _POSITION_LIMIT_EPSILON
    for point_index, positions in enumerate(waypoint_positions):
        for joint_index, (value, start, limits) in enumerate(zip(
                positions, start_positions, position_limits)):
            if limits is None:
                continue
            lower, upper = limits
            if lower - epsilon <= value <= upper + epsilon:
                continue
            moving_toward_range = (
                start < lower and start <= value <= lower + epsilon
            ) or (
                start > upper and upper - epsilon <= value <= start
            )
            if not moving_toward_range:
                raise TrajectoryError(
                    'point {} exceeds the position limit for {}'.format(
                        point_index, joint_names[joint_index]
                    )
                )
