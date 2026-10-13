"""Validation and interpolation helpers for D1 joint trajectories."""

import math


_POSITION_LIMIT_EPSILON = 1.0e-6
# Joint feedback jitters by a fraction of a degree. MoveIt's first trajectory
# point is the joint state at planning time, while the controller compares it
# with feedback received just before execution, so a joint resting on a limit
# (the laid-down J1 sits at exactly -90 degrees) can appear to step slightly
# farther outside between the two readings.
_MEASUREMENT_TOLERANCE_RAD = math.radians(1.0)


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


def conservative_waypoint_times(
        waypoint_positions, max_velocity, max_acceleration,
        minimum_interval=0.1):
    """Retimestamp a joint path without changing any waypoint positions.

    Each segment is given at least the time needed for a rest-to-rest move
    under the supplied velocity and acceleration ceilings. This is a slow
    fallback for Cartesian paths with unusable timestamps, not a replacement
    for MoveIt's normal time parameterization.
    """
    if not all(
            math.isfinite(value) and value > 0.0
            for value in (max_velocity, max_acceleration, minimum_interval)):
        raise TrajectoryError('retiming limits must be finite and positive')
    positions = tuple(tuple(point) for point in waypoint_positions)
    if not positions:
        raise TrajectoryError('cannot retime an empty trajectory')
    joint_count = len(positions[0])
    if not joint_count or any(
            len(point) != joint_count
            or not all(math.isfinite(value) for value in point)
            for point in positions):
        raise TrajectoryError('cannot retime invalid joint positions')

    times = [0.0]
    transition_distance = max_velocity * max_velocity / max_acceleration
    for previous, current in zip(positions, positions[1:]):
        interval = minimum_interval
        for start, end in zip(previous, current):
            distance = abs(end - start)
            if distance <= transition_distance:
                required = 2.0 * math.sqrt(distance / max_acceleration)
            else:
                required = (
                    distance / max_velocity
                    + max_velocity / max_acceleration
                )
            interval = max(interval, required)
        times.append(times[-1] + interval)
    return tuple(times)


def smooth_waypoint_times(waypoint_positions, max_velocity, max_acceleration):
    """Time a joint path as one smooth move at a bounded joint speed.

    The path is followed at constant speed with a single acceleration ramp at
    the start and a single deceleration ramp at the end (a trapezoidal profile
    over path length, measured as the largest joint change per segment), so
    no joint exceeds ``max_velocity`` or ``max_acceleration``. Positions are
    unchanged. Unlike ``conservative_waypoint_times`` it does not stop at
    every waypoint.
    """
    if not all(math.isfinite(value) and value > 0.0
               for value in (max_velocity, max_acceleration)):
        raise TrajectoryError('retiming limits must be finite and positive')
    positions = tuple(tuple(float(value) for value in point)
                      for point in waypoint_positions)
    if not positions:
        raise TrajectoryError('cannot retime an empty trajectory')
    joint_count = len(positions[0])
    if not joint_count or any(
            len(point) != joint_count
            or not all(math.isfinite(value) for value in point)
            for point in positions):
        raise TrajectoryError('cannot retime invalid joint positions')
    cumulative = [0.0]
    for previous, current in zip(positions, positions[1:]):
        step = max(abs(end - start) for start, end in zip(previous, current))
        cumulative.append(cumulative[-1] + step)
    total = cumulative[-1]
    ramp = max_velocity * max_velocity / max_acceleration
    if total >= ramp:
        peak = max_velocity
        ramp_time = max_velocity / max_acceleration
        cruise_time = (total - ramp) / max_velocity
    else:
        peak = math.sqrt(max_acceleration * total)
        ramp_time = peak / max_acceleration
        cruise_time = 0.0
    half_ramp = 0.5 * peak * ramp_time

    def time_at(distance):
        if distance <= half_ramp:
            return math.sqrt(2.0 * distance / max_acceleration)
        if distance <= half_ramp + peak * cruise_time:
            return ramp_time + (distance - half_ramp) / peak
        remaining = max(0.0, total - distance)
        return 2.0 * ramp_time + cruise_time - math.sqrt(
            2.0 * remaining / max_acceleration
        )

    # Keep times strictly increasing even across zero-length segments; the
    # first point is the start state, so it takes a small nonzero time.
    times = []
    previous = 0.0
    for distance in cumulative:
        value = max(time_at(distance), previous + 1.0e-3)
        times.append(value)
        previous = value
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
    but it may not move farther outside the range than feedback noise
    (``_MEASUREMENT_TOLERANCE_RAD``) beyond the start or the limit, whichever
    is farther out. Tiny floating-point overshoots at the boundary are treated
    as numerical noise rather than a real limit violation.
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
            noise = _MEASUREMENT_TOLERANCE_RAD
            within_measured_start = (
                value < lower
                and value >= min(start, lower) - noise
            ) or (
                value > upper
                and value <= max(start, upper) + noise
            )
            if not within_measured_start:
                raise TrajectoryError(
                    'point {} exceeds the position limit for {}'.format(
                        point_index, joint_names[joint_index]
                    )
                )
