# Copyright 2026 I3T
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

"""Pure helpers for detection target parsing and three-dimensional geometry."""

import json
import math

import numpy as np


def parse_detection_targets(raw_value):
    """Parse a target message into a normalized list of unique phrases."""
    if not isinstance(raw_value, str):
        raise ValueError('detection targets must be provided as a string')

    stripped = raw_value.strip()
    if not stripped:
        return []

    if stripped.startswith('[') or stripped.startswith('{'):
        try:
            decoded = json.loads(stripped)
        except ValueError as error:
            raise ValueError(
                'detection target JSON is invalid: {}'.format(error)
            )
        if isinstance(decoded, dict):
            decoded = decoded.get('targets')
        if isinstance(decoded, str):
            candidates = [decoded]
        elif isinstance(decoded, list):
            candidates = decoded
        else:
            raise ValueError(
                'target JSON must be a list or an object with a targets list'
            )
    else:
        candidates = stripped.split(',')

    targets = []
    for candidate in candidates:
        if not isinstance(candidate, str):
            raise ValueError('every detection target must be a string')
        target = ' '.join(candidate.lower().strip().strip('.').split())
        if target and target not in targets:
            targets.append(target)
    return targets


def camera_point_from_depth(
        depth_image, bounding_box, intrinsics, minimum_depth,
        maximum_depth, center_fraction=0.5):
    """Estimate a point from the center of a depth-aligned box."""
    if depth_image.ndim != 2:
        raise ValueError('depth image must have one channel')
    if not 0.0 < center_fraction <= 1.0:
        raise ValueError('center_fraction must be in (0, 1]')

    image_height, image_width = depth_image.shape
    x_min, y_min, x_max, y_max = bounding_box
    x_min = max(0, min(image_width - 1, int(math.floor(x_min))))
    y_min = max(0, min(image_height - 1, int(math.floor(y_min))))
    x_max = max(0, min(image_width - 1, int(math.ceil(x_max))))
    y_max = max(0, min(image_height - 1, int(math.ceil(y_max))))
    if x_max < x_min or y_max < y_min:
        return None

    center_x = 0.5 * (x_min + x_max)
    center_y = 0.5 * (y_min + y_max)
    sample_width = max(
        1, int(round((x_max - x_min + 1) * center_fraction))
    )
    sample_height = max(
        1, int(round((y_max - y_min + 1) * center_fraction))
    )
    sample_x_min = max(
        x_min, int(math.floor(center_x - sample_width / 2.0 + 0.5))
    )
    sample_y_min = max(
        y_min, int(math.floor(center_y - sample_height / 2.0 + 0.5))
    )
    sample_x_max = min(x_max + 1, sample_x_min + sample_width)
    sample_y_max = min(y_max + 1, sample_y_min + sample_height)

    sample = depth_image[
        sample_y_min:sample_y_max, sample_x_min:sample_x_max
    ]
    valid_mask = (
        np.isfinite(sample)
        & (sample >= minimum_depth)
        & (sample <= maximum_depth)
    )
    if not np.any(valid_mask):
        return None

    sample_rows, sample_columns = np.nonzero(valid_mask)
    depths = sample[valid_mask].astype(np.float64)
    median_depth = float(np.median(depths))
    depth_tolerance = max(0.05, 0.10 * median_depth)
    inlier_mask = np.abs(depths - median_depth) <= depth_tolerance
    if not np.any(inlier_mask):
        return None

    depths = depths[inlier_mask]
    pixel_u = float(
        np.median(sample_columns[inlier_mask] + sample_x_min)
    )
    pixel_v = float(np.median(sample_rows[inlier_mask] + sample_y_min))
    depth = float(np.median(depths))

    focal_x, focal_y, principal_x, principal_y = intrinsics
    if focal_x <= 0.0 or focal_y <= 0.0:
        raise ValueError('camera focal lengths must be positive')

    return {
        'x': (pixel_u - principal_x) * depth / focal_x,
        'y': (pixel_v - principal_y) * depth / focal_y,
        'z': depth,
        'pixel_u': pixel_u,
        'pixel_v': pixel_v,
        'depth_sample_count': int(depths.size),
    }


def narrow_grasp_band_from_depth(
        depth_image, bounding_box, intrinsics, reference_point,
        minimum_depth, maximum_depth, band_height_m=0.015,
        width_tolerance_m=0.01, depth_tolerance_m=0.06):
    """Estimate a narrow, vertically supported band on the visible object.

    This is a depth silhouette estimate, not a full 3-D object model. Only
    depths near the central object sample are used. Edge rows and isolated
    narrow runs are rejected so background gaps cannot become grasp targets.
    """
    if depth_image.ndim != 2:
        raise ValueError('depth image must have one channel')
    if not all(math.isfinite(value) and value > 0.0 for value in (
            band_height_m, depth_tolerance_m)):
        raise ValueError('band height and depth tolerance must be positive')
    if not math.isfinite(width_tolerance_m) or width_tolerance_m < 0.0:
        raise ValueError('band width tolerance must be nonnegative')
    focal_x, focal_y, principal_x, principal_y = intrinsics
    if focal_x <= 0.0 or focal_y <= 0.0:
        raise ValueError('camera focal lengths must be positive')
    reference_depth = float(reference_point['z'])
    reference_u = float(reference_point['pixel_u'])
    if not math.isfinite(reference_depth) or reference_depth <= 0.0:
        return None

    image_height, image_width = depth_image.shape
    x_min, y_min, x_max, y_max = bounding_box
    x_min = max(0, min(image_width - 1, int(math.floor(x_min))))
    y_min = max(0, min(image_height - 1, int(math.floor(y_min))))
    x_max = max(0, min(image_width - 1, int(math.ceil(x_max))))
    y_max = max(0, min(image_height - 1, int(math.ceil(y_max))))
    box_width = x_max - x_min + 1
    box_height = y_max - y_min + 1
    if box_width < 8 or box_height < 12:
        return None

    margin = max(2, int(round(0.15 * box_height)))
    first_row, last_row = y_min + margin, y_max - margin
    band_height = max(3, int(round(focal_y * band_height_m / reference_depth)))
    if last_row - first_row + 1 < 2 * band_height:
        return None
    step = max(1, band_height // 2)
    candidates = []
    for top in range(first_row, last_row - band_height + 2, step):
        sample = depth_image[top:top + band_height, x_min:x_max + 1]
        foreground = (
            np.isfinite(sample)
            & (sample >= minimum_depth)
            & (sample <= maximum_depth)
            & (np.abs(sample - reference_depth) <= depth_tolerance_m)
        )
        supported = np.count_nonzero(foreground, axis=0) >= max(
            2, int(math.ceil(0.7 * band_height))
        )
        padded = np.concatenate(([False], supported, [False]))
        transitions = np.diff(padded.astype(np.int8))
        starts = np.flatnonzero(transitions == 1)
        ends = np.flatnonzero(transitions == -1)
        runs = []
        for start, end in zip(starts, ends):
            width_m = (end - start) * reference_depth / focal_x
            center_u = x_min + 0.5 * (start + end - 1)
            if width_m < 0.01 or (
                    abs(center_u - reference_u) > max(
                        0.2 * box_width, 0.5 * (end - start))):
                continue
            runs.append((abs(center_u - reference_u), start, end, width_m))
        if not runs:
            continue
        _, start, end, width_m = min(runs)
        depths = sample[:, start:end][foreground[:, start:end]]
        if depths.size < band_height * 3:
            continue
        depth = float(np.median(depths))
        pixel_u = x_min + 0.5 * (start + end - 1)
        pixel_v = top + 0.5 * (band_height - 1)
        candidates.append({
            'x': (pixel_u - principal_x) * depth / focal_x,
            'y': (pixel_v - principal_y) * depth / focal_y,
            'z': depth,
            'pixel_u': pixel_u,
            'pixel_v': pixel_v,
            'width_m': (end - start) * depth / focal_x,
            'band_height_m': band_height * depth / focal_y,
        })

    stable = [
        candidate for candidate in candidates
        if any(
            other is not candidate
            and abs(other['pixel_v'] - candidate['pixel_v']) <= band_height
            and abs(other['width_m'] - candidate['width_m'])
            <= width_tolerance_m
            for other in candidates
        )
    ]
    if not stable:
        return None
    minimum_width = min(item['width_m'] for item in stable)
    eligible = [
        item for item in stable
        if item['width_m'] <= minimum_width + width_tolerance_m
    ]
    center_v = 0.5 * (y_min + y_max)
    return min(
        eligible,
        key=lambda item: (abs(item['pixel_v'] - center_v), item['width_m']),
    )


def optical_point_to_camera_link(point):
    """Convert RealSense optical XYZ to the robot camera-link convention.

    RealSense optical coordinates use X right, Y down, and Z forward.
    ``camera_link`` uses X forward, Y left, and Z up.
    """
    optical_x = float(point['x'])
    optical_y = float(point['y'])
    optical_z = float(point['z'])
    if not all(math.isfinite(value) for value in (
            optical_x, optical_y, optical_z)):
        raise ValueError('optical point coordinates must be finite')
    return {
        'x': optical_z,
        'y': -optical_x,
        'z': -optical_y,
    }


def transform_point(point, translation, quaternion):
    """Apply a geometry_msgs-style rigid transform to an XYZ point."""
    vector = np.asarray(point, dtype=np.float64)
    offset = np.asarray(translation, dtype=np.float64)
    rotation = np.asarray(quaternion, dtype=np.float64)
    if vector.shape != (3,) or offset.shape != (3,):
        raise ValueError('point and translation must contain three values')
    if rotation.shape != (4,):
        raise ValueError('quaternion must contain x, y, z, and w')

    norm = float(np.linalg.norm(rotation))
    if norm <= 1.0e-12:
        raise ValueError('transform quaternion has zero length')
    rotation /= norm
    quaternion_vector = rotation[:3]
    quaternion_w = rotation[3]
    doubled_cross = 2.0 * np.cross(quaternion_vector, vector)
    rotated = (
        vector
        + quaternion_w * doubled_cross
        + np.cross(quaternion_vector, doubled_cross)
    )
    result = rotated + offset
    return tuple(float(value) for value in result)
