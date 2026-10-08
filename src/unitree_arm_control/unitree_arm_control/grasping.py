"""ROS-independent geometry and semantic-object selection helpers."""

import copy
import math


def trajectory_endpoint_state(start_state, trajectory):
    """Apply a planned group's final joint positions to its full start state."""
    names = list(start_state.joint_state.name)
    positions = list(start_state.joint_state.position)
    joint_trajectory = trajectory.joint_trajectory
    moving_names = list(joint_trajectory.joint_names)
    if len(names) != len(positions) or len(set(names)) != len(names):
        raise ValueError('planned trajectory has an invalid starting state')
    if not joint_trajectory.points or not moving_names:
        raise ValueError('planned pre-grasp trajectory is empty')
    endpoint = list(joint_trajectory.points[-1].positions)
    if (len(moving_names) != len(endpoint)
            or len(set(moving_names)) != len(moving_names)):
        raise ValueError('planned pre-grasp endpoint has invalid joint positions')
    if not all(math.isfinite(value) for value in endpoint):
        raise ValueError('planned pre-grasp endpoint has non-finite positions')
    by_name = dict(zip(names, range(len(names))))
    if any(name not in by_name for name in moving_names):
        raise ValueError('planned pre-grasp endpoint lacks a full start state')
    state = copy.deepcopy(start_state)
    for name, value in zip(moving_names, endpoint):
        positions[by_name[name]] = value
    state.joint_state.position = positions
    state.joint_state.velocity = []
    state.joint_state.effort = []
    state.is_diff = False
    return state


def normalize_label(value):
    """Normalize a semantic label for exact class matching."""
    return ' '.join(str(value).strip().lower().split())


def parse_forward_grasp_depth_offsets(entries):
    """Parse ``class=metres`` entries into normalized class offsets."""
    offsets = {}
    for entry in entries:
        if not isinstance(entry, str) or '=' not in entry:
            raise ValueError(
                'forward grasp depth entries must use class=metres'
            )
        label_text, value_text = entry.rsplit('=', 1)
        label = normalize_label(label_text)
        if not label:
            raise ValueError('forward grasp depth class must not be empty')
        if label in offsets:
            raise ValueError(
                'duplicate forward grasp depth class: {}'.format(label)
            )
        try:
            value = float(value_text.strip())
        except ValueError:
            raise ValueError(
                'forward grasp depth for {} must be numeric'.format(label)
            )
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                'forward grasp depth for {} must be finite and nonnegative'.format(
                    label
                )
            )
        offsets[label] = value
    return offsets


def forward_grasp_depth_for_class(label, default_offset, class_offsets):
    """Return the class-specific forward depth or the configured default."""
    return float(class_offsets.get(normalize_label(label), default_offset))


def parse_grasp_height_offsets(entries):
    """Parse signed ``class=metres`` grasp-height corrections."""
    offsets = {}
    for entry in entries:
        if not isinstance(entry, str) or '=' not in entry:
            raise ValueError('grasp height entries must use class=metres')
        label_text, value_text = entry.rsplit('=', 1)
        label = normalize_label(label_text)
        if not label:
            raise ValueError('grasp height class must not be empty')
        if label in offsets:
            raise ValueError('duplicate grasp height class: {}'.format(label))
        try:
            value = float(value_text.strip())
        except ValueError:
            raise ValueError('grasp height for {} must be numeric'.format(label))
        if not math.isfinite(value):
            raise ValueError('grasp height for {} must be finite'.format(label))
        offsets[label] = value
    return offsets


def grasp_height_for_class(label, class_offsets):
    """Return the class-specific vertical correction, or zero."""
    return float(class_offsets.get(normalize_label(label), 0.0))


def stamp_seconds(stamp):
    """Convert a ROS Time-like object to floating-point seconds."""
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def eligible_objects(
        objects,
        object_class,
        object_id,
        now_seconds,
        minimum_confidence,
        minimum_observations,
        maximum_age_seconds,
        active_status):
    """Return semantic objects that satisfy the coordinator safeguards."""
    requested_class = normalize_label(object_class)
    requested_id = str(object_id).strip()
    matches = []
    for candidate in objects:
        if normalize_label(candidate.label) != requested_class:
            continue
        if requested_id and candidate.object_id != requested_id:
            continue
        if object_safeguard_failures(
                candidate,
                now_seconds,
                minimum_confidence,
                minimum_observations,
                maximum_age_seconds,
                active_status):
            continue
        matches.append(candidate)
    return matches


def object_safeguard_failures(
        candidate,
        now_seconds,
        minimum_confidence,
        minimum_observations,
        maximum_age_seconds,
        active_status):
    """Describe every safety gate that rejects one semantic object."""
    failures = []
    if candidate.status != active_status:
        failures.append('status={} is not ACTIVE'.format(candidate.status))
    confidence = float(candidate.confidence)
    if confidence < minimum_confidence:
        failures.append(
            'confidence {:.3f} is below {:.3f}'.format(
                confidence, minimum_confidence
            )
        )
    observations = int(candidate.observation_count)
    if observations < minimum_observations:
        failures.append(
            '{} observations is below {}'.format(
                observations, minimum_observations
            )
        )
    age = max(
        0.0, now_seconds - stamp_seconds(candidate.last_observed)
    )
    if age > maximum_age_seconds:
        failures.append(
            'last observation is {:.1f}s old (maximum {:.1f}s)'.format(
                age, maximum_age_seconds
            )
        )
    return failures


def quaternion_from_rpy(roll, pitch, yaw):
    """Return an xyzw quaternion for fixed-axis roll, pitch, and yaw."""
    half_roll = 0.5 * roll
    half_pitch = 0.5 * pitch
    half_yaw = 0.5 * yaw
    cr = math.cos(half_roll)
    sr = math.sin(half_roll)
    cp = math.cos(half_pitch)
    sp = math.sin(half_pitch)
    cy = math.cos(half_yaw)
    sy = math.sin(half_yaw)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def multiply_quaternions(left, right):
    """Compose two xyzw quaternions, applying ``right`` then ``left``."""
    lx, ly, lz, lw = (float(value) for value in left)
    rx, ry, rz, rw = (float(value) for value in right)
    result = (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )
    norm = math.sqrt(sum(value * value for value in result))
    if norm <= 0.0:
        raise ValueError('composed quaternion must not be zero')
    return tuple(value / norm for value in result)


def approach_orientation_candidates(orientation, maximum_adjustment):
    """Try small local wrist rotations while retaining one XYZ target."""
    if (
            not math.isfinite(maximum_adjustment)
            or not 0.0 < maximum_adjustment <= math.pi):
        raise ValueError('maximum orientation adjustment must be in (0, pi]')
    values = tuple(float(value) for value in orientation)
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise ValueError('orientation must be a finite xyzw quaternion')
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 1.0e-9:
        raise ValueError('orientation must have non-zero length')
    reference = tuple(value / norm for value in values)
    candidates = [reference]
    for fraction in (0.5, 1.0):
        for axis in range(3):
            for sign in (1.0, -1.0):
                angles = [0.0, 0.0, 0.0]
                angles[axis] = sign * fraction * maximum_adjustment
                local_rotation = quaternion_from_rpy(*angles)
                candidates.append(multiply_quaternions(
                    reference, local_rotation
                ))
    return tuple(candidates)


def quaternion_angular_distance(first, second):
    """Return the shortest rotation between two xyzw orientations in radians."""
    values = []
    for quaternion in (first, second):
        components = tuple(float(value) for value in quaternion)
        if (len(components) != 4
                or not all(math.isfinite(value) for value in components)):
            raise ValueError('orientation must be a finite xyzw quaternion')
        norm = math.sqrt(sum(value * value for value in components))
        if norm <= 1.0e-9:
            raise ValueError('orientation must have non-zero length')
        values.append(tuple(value / norm for value in components))
    dot = abs(sum(a * b for a, b in zip(*values)))
    return 2.0 * math.acos(min(1.0, dot))


def rotate_vector(vector, quaternion):
    """Rotate a three-vector by an xyzw unit quaternion."""
    x, y, z = (float(value) for value in vector)
    qx, qy, qz, qw = (float(value) for value in quaternion)
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 0.0:
        raise ValueError('transform quaternion must not be zero')
    qx /= norm
    qy /= norm
    qz /= norm
    qw /= norm

    # Equivalent to q * [v, 0] * conjugate(q), without allocations.
    tx = 2.0 * (qy * z - qz * y)
    ty = 2.0 * (qz * x - qx * z)
    tz = 2.0 * (qx * y - qy * x)
    return (
        x + qw * tx + qy * tz - qz * ty,
        y + qw * ty + qz * tx - qx * tz,
        z + qw * tz + qx * ty - qy * tx,
    )


def transform_point(point, translation, quaternion):
    """Apply a source-to-target rigid transform to a three-vector."""
    rotated = rotate_vector(point, quaternion)
    return tuple(
        value + float(offset)
        for value, offset in zip(rotated, translation)
    )


def distance_from_origin(point):
    """Return the Euclidean distance of a three-vector from its origin."""
    return math.sqrt(sum(float(value) ** 2 for value in point))


def distance_between(first, second):
    """Return the Euclidean distance between two three-vectors."""
    return math.sqrt(sum(
        (float(left) - float(right)) ** 2
        for left, right in zip(first, second)
    ))


def approach_line_error(actual_point, pregrasp_point, approach_direction):
    """Distance from a point to a planned approach line."""
    axis = _normalized(approach_direction)
    offset = tuple(
        float(target) - float(actual)
        for target, actual in zip(pregrasp_point, actual_point)
    )
    along = sum(delta * direction for delta, direction in zip(offset, axis))
    return math.sqrt(sum(
        (delta - along * direction) ** 2
        for delta, direction in zip(offset, axis)
    ))


def wrist_point_from_tip(tip_point, orientation, offset=(0.00038, 0.0, 0.1256)):
    """Recover the wrist center from the tool-center point and fixed tool offset."""
    rotated_offset = rotate_vector(offset, orientation)
    return tuple(
        float(tip_value) - float(offset_value)
        for tip_value, offset_value in zip(tip_point, rotated_offset)
    )


def approach_axis_error(orientation, approach_direction):
    """Return the angle between the gripper's forward axis and the approach."""
    forward = rotate_vector((0.0, 0.0, 1.0), orientation)
    direction = _normalized(approach_direction)
    cosine = max(-1.0, min(1.0, _dot(forward, direction)))
    return math.acos(cosine)


def _normalized(vector):
    values = tuple(float(value) for value in vector)
    length = distance_from_origin(values)
    if length <= 1.0e-9:
        raise ValueError('vector must have non-zero length')
    return tuple(value / length for value in values)


def _horizontal_depth_axis(depth_direction, fallback_direction):
    """Keep the camera-derived forward correction level in the frame XY plane."""
    horizontal = (depth_direction[0], depth_direction[1], 0.0)
    if distance_from_origin(horizontal) <= 1.0e-9:
        horizontal = (fallback_direction[0], fallback_direction[1], 0.0)
    return _normalized(horizontal)


def _depth_axis(depth_direction, fallback_direction, horizontal_depth):
    if horizontal_depth:
        return _horizontal_depth_axis(depth_direction, fallback_direction)
    return _normalized(depth_direction)


def _cross(first, second):
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _dot(first, second):
    return sum(left * right for left, right in zip(first, second))


def _quaternion_from_matrix(columns):
    """Convert a three-by-three rotation matrix, supplied by columns."""
    matrix = tuple(zip(*columns))
    m00, m01, m02 = matrix[0]
    m10, m11, m12 = matrix[1]
    m20, m21, m22 = matrix[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = (
            (m21 - m12) / scale,
            (m02 - m20) / scale,
            (m10 - m01) / scale,
            0.25 * scale,
        )
    elif m00 > m11 and m00 > m22:
        scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        quaternion = (
            0.25 * scale,
            (m01 + m10) / scale,
            (m02 + m20) / scale,
            (m21 - m12) / scale,
        )
    elif m11 > m22:
        scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        quaternion = (
            (m01 + m10) / scale,
            0.25 * scale,
            (m12 + m21) / scale,
            (m02 - m20) / scale,
        )
    else:
        scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        quaternion = (
            (m02 + m20) / scale,
            (m12 + m21) / scale,
            0.25 * scale,
            (m10 - m01) / scale,
        )
    norm = math.sqrt(sum(value * value for value in quaternion))
    return tuple(value / norm for value in quaternion)


def quaternion_from_approach(approach_direction, tool_roll=0.0):
    """Orient local +Z along an approach direction with local +X upward.

    ``tool_roll`` rotates the finger arrangement around the approach axis.
    """
    local_z = _normalized(approach_direction)
    up = (0.0, 0.0, 1.0)
    projection = _dot(up, local_z)
    projected_up = tuple(
        up_value - projection * z_value
        for up_value, z_value in zip(up, local_z)
    )
    if distance_from_origin(projected_up) <= 1.0e-6:
        projected_up = (1.0, 0.0, 0.0)
    local_x = _normalized(projected_up)
    local_y = _normalized(_cross(local_z, local_x))

    cosine = math.cos(float(tool_roll))
    sine = math.sin(float(tool_roll))
    rolled_x = tuple(
        cosine * x_value + sine * y_value
        for x_value, y_value in zip(local_x, local_y)
    )
    rolled_y = tuple(
        -sine * x_value + cosine * y_value
        for x_value, y_value in zip(local_x, local_y)
    )
    return _quaternion_from_matrix((rolled_x, rolled_y, local_z))


def level_gripper_opening(orientation):
    """Level the jaw opening without changing the tool's forward direction.

    The jaws are symmetric about local X, so either of the two half-turn-
    equivalent roll solutions is acceptable. Choose the nearer one to avoid
    an unnecessary 180-degree wrist rotation at pre-grasp.
    """
    forward = rotate_vector((0.0, 0.0, 1.0), orientation)
    options = (
        quaternion_from_approach(forward),
        quaternion_from_approach(forward, math.pi),
    )
    return min(
        options,
        key=lambda option: quaternion_angular_distance(option, orientation),
    )


def approach_corridor_geometry(start_point, end_point, radius):
    """Return a box corridor whose local Z axis joins two tool positions."""
    if radius <= 0.0:
        raise ValueError('corridor radius must be positive')
    start = tuple(float(value) for value in start_point)
    end = tuple(float(value) for value in end_point)
    direction = tuple(
        end_value - start_value
        for start_value, end_value in zip(start, end)
    )
    length = distance_from_origin(direction)
    if length <= 1.0e-9:
        raise ValueError('corridor endpoints must be different')
    return {
        'center': tuple(
            0.5 * (start_value + end_value)
            for start_value, end_value in zip(start, end)
        ),
        'orientation': quaternion_from_approach(direction),
        'dimensions': (
            2.0 * float(radius),
            2.0 * float(radius),
            length + 2.0 * float(radius),
        ),
    }


def _corrected_grasp_point(
        object_point, depth_axis, approach_axis, forward_grasp_depth,
        grasp_center_offset, grasp_height_offset):
    corrected = tuple(
        float(value)
        + float(forward_grasp_depth) * depth_value
        - float(grasp_center_offset) * approach_value
        for value, depth_value, approach_value in zip(
            object_point, depth_axis, approach_axis
        )
    )
    return (
        corrected[0], corrected[1],
        corrected[2] + float(grasp_height_offset),
    )


def generate_approach_candidates(
        object_point,
        approach_origin,
        yaw_offsets,
        approach_distance,
        grasp_center_offset,
        tool_roll=0.0,
        forward_grasp_depth=0.0,
        depth_direction=None,
        grasp_height_offset=0.0,
        horizontal_depth=False, level_approach=False):
    """Generate three-dimensional, object-directed pre-grasp candidates.

    The primary approach direction points from a fixed arm reference origin
    toward the object, independent of the current gripper position. Yaw
    alternatives rotate that direction about the planning frame's vertical
    axis while retaining its vertical component.
    The pre-grasp offset is applied along the resulting direction. For a
    measured band, level_approach keeps both tool centers at band height.
    """
    if approach_distance <= 0.0:
        raise ValueError('approach_distance must be positive')
    if grasp_center_offset < 0.0:
        raise ValueError('grasp_center_offset must not be negative')
    if forward_grasp_depth < 0.0:
        raise ValueError('forward_grasp_depth must not be negative')
    if not math.isfinite(grasp_height_offset):
        raise ValueError('grasp_height_offset must be finite')

    radial = (
        float(object_point[0]) - float(approach_origin[0]),
        float(object_point[1]) - float(approach_origin[1]),
        float(object_point[2]) - float(approach_origin[2]),
    )
    if level_approach:
        # The target already identifies the measured grasp band. A sloped
        # approach would put the pre-grasp below/above that band and sweep the
        # fingers vertically through the object during the final advance.
        radial = (radial[0], radial[1], 0.0)
    radial = _normalized(radial)
    depth_axis = _depth_axis(
        radial if depth_direction is None else depth_direction,
        radial, horizontal_depth,
    )
    candidates = []
    for yaw_offset in yaw_offsets:
        cosine = math.cos(float(yaw_offset))
        sine = math.sin(float(yaw_offset))
        direction = (
            cosine * radial[0] - sine * radial[1],
            sine * radial[0] + cosine * radial[1],
            radial[2],
        )
        grasp_point = _corrected_grasp_point(
            object_point, depth_axis, direction, forward_grasp_depth,
            grasp_center_offset, grasp_height_offset,
        )
        pregrasp_point = tuple(
            value - float(approach_distance) * axis
            for value, axis in zip(grasp_point, direction)
        )
        candidates.append({
            'yaw_offset': float(yaw_offset),
            'approach_direction': direction,
            'depth_direction': depth_axis,
            'forward_grasp_depth': float(forward_grasp_depth),
            'grasp_height_offset': float(grasp_height_offset),
            'horizontal_depth': bool(horizontal_depth),
            'pregrasp_point': pregrasp_point,
            'grasp_point': grasp_point,
            'orientation': quaternion_from_approach(
                direction, tool_roll
            ),
        })
    return tuple(sorted(
        candidates, key=lambda item: abs(item['yaw_offset'])
    ))


def retarget_approach_candidate(
        candidate,
        object_point,
        approach_distance,
        grasp_center_offset,
        forward_grasp_depth=0.0,
        depth_direction=None,
        grasp_height_offset=0.0,
        horizontal_depth=None):
    """Move a candidate to a refreshed object point without rotating it."""
    if approach_distance <= 0.0:
        raise ValueError('approach_distance must be positive')
    if grasp_center_offset < 0.0:
        raise ValueError('grasp_center_offset must not be negative')
    if forward_grasp_depth < 0.0:
        raise ValueError('forward_grasp_depth must not be negative')
    if not math.isfinite(grasp_height_offset):
        raise ValueError('grasp_height_offset must be finite')
    direction = _normalized(candidate['approach_direction'])
    if horizontal_depth is None:
        horizontal_depth = candidate.get('horizontal_depth', False)
    depth_axis = _depth_axis(
        candidate.get('depth_direction', direction)
        if depth_direction is None else depth_direction,
        direction,
        horizontal_depth,
    )
    grasp_point = _corrected_grasp_point(
        object_point, depth_axis, direction, forward_grasp_depth,
        grasp_center_offset, grasp_height_offset,
    )
    pregrasp_point = tuple(
        value - float(approach_distance) * axis
        for value, axis in zip(grasp_point, direction)
    )
    updated = dict(candidate)
    updated['approach_direction'] = direction
    updated['depth_direction'] = depth_axis
    updated['forward_grasp_depth'] = float(forward_grasp_depth)
    updated['grasp_height_offset'] = float(grasp_height_offset)
    updated['horizontal_depth'] = bool(horizontal_depth)
    updated['grasp_point'] = grasp_point
    updated['pregrasp_point'] = pregrasp_point
    return updated


def reach_safeguard_problem(
        point,
        minimum_reach,
        maximum_reach,
        reference_frame):
    """Describe a reach-envelope violation in the arm reference frame."""
    coordinates = tuple(float(value) for value in point)
    reach = distance_from_origin(coordinates)
    detail = (
        '{:.3f} m from {} (xyz [{:.3f}, {:.3f}, {:.3f}] m)'.format(
            reach,
            reference_frame,
            coordinates[0],
            coordinates[1],
            coordinates[2],
        )
    )
    if reach < minimum_reach:
        return (
            'pre-grasp target is {}, inside minimum reach safeguard of '
            '{:.3f} m'.format(detail, minimum_reach)
        )
    if reach > maximum_reach:
        return (
            'pre-grasp target is {}, outside maximum reach safeguard of '
            '{:.3f} m'.format(detail, maximum_reach)
        )
    return None
