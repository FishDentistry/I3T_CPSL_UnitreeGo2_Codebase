"""ROS-independent geometry and semantic-object selection helpers."""

import math


def normalize_label(value):
    """Normalize a semantic label for exact class matching."""
    return ' '.join(str(value).strip().lower().split())


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
        if candidate.status != active_status:
            continue
        if float(candidate.confidence) < minimum_confidence:
            continue
        if int(candidate.observation_count) < minimum_observations:
            continue
        age = now_seconds - stamp_seconds(candidate.last_observed)
        if age < 0.0:
            age = 0.0
        if age > maximum_age_seconds:
            continue
        matches.append(candidate)
    return matches


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
