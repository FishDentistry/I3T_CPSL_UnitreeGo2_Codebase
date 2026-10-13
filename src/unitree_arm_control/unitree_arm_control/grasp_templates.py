"""Object-relative grasp demonstrations and their persistent library.

The reference frame is anchored at the detected object center. Its X axis
points from the fixed arm mount toward the center in the horizontal plane, Z
is up, and Y completes the right-handed frame. It is not an object-pose
estimate, so it intentionally does not encode an object's rotation.
"""

import math
import os
import re
import tempfile

import yaml

from unitree_arm_control import grasping


_NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$')


def _finite_tuple(values, length, description):
    result = tuple(float(value) for value in values)
    if len(result) != length or not all(math.isfinite(value) for value in result):
        raise ValueError('{} must contain {} finite values'.format(
            description, length
        ))
    return result


def _unit_quaternion(values):
    result = _finite_tuple(values, 4, 'orientation')
    norm = math.sqrt(sum(value * value for value in result))
    if norm <= 1.0e-9:
        raise ValueError('orientation quaternion must not be zero')
    return tuple(value / norm for value in result)


def _conjugate(quaternion):
    return (-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3])


def object_reference(object_point, arm_origin):
    """Return the object-center origin and yaw-only base-frame orientation."""
    origin = _finite_tuple(object_point, 3, 'object center')
    arm = _finite_tuple(arm_origin, 3, 'arm origin')
    dx, dy = origin[0] - arm[0], origin[1] - arm[1]
    if math.hypot(dx, dy) <= 1.0e-6:
        raise ValueError('object center is too close to the arm origin in XY')
    half_yaw = 0.5 * math.atan2(dy, dx)
    return origin, (0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw))


def relative_pose(reference, position, orientation):
    origin, rotation = reference
    point = _finite_tuple(position, 3, 'position')
    quaternion = _unit_quaternion(orientation)
    inverse = _conjugate(rotation)
    offset = tuple(value - base for value, base in zip(point, origin))
    return {
        'position': list(grasping.rotate_vector(offset, inverse)),
        'orientation': list(_unit_quaternion(
            grasping.multiply_quaternions(inverse, quaternion)
        )),
    }


def absolute_pose(reference, pose):
    origin, rotation = reference
    point = _finite_tuple(pose['position'], 3, 'relative position')
    quaternion = _unit_quaternion(pose['orientation'])
    rotated = grasping.rotate_vector(point, rotation)
    return (
        tuple(value + base for value, base in zip(rotated, origin)),
        _unit_quaternion(grasping.multiply_quaternions(rotation, quaternion)),
    )


def validate_template(template):
    if not isinstance(template, dict):
        raise ValueError('grasp template must be a mapping')
    if (not isinstance(template.get('name'), str)
            or not _NAME.fullmatch(template['name'])):
        raise ValueError('template name must be 1-64 letters, digits, _ or -')
    if template.get('reference') != 'target_object_center':
        raise ValueError('template reference must be target_object_center')
    for key in ('pregrasp', 'contact'):
        pose = template[key]
        if not isinstance(pose, dict):
            raise ValueError('{} must be a pose mapping'.format(key))
        _finite_tuple(pose['position'], 3, key + ' position')
        _unit_quaternion(pose['orientation'])
    pre = template['pregrasp']
    contact = template['contact']
    pre_point = _finite_tuple(pre['position'], 3, 'pregrasp position')
    contact_point = _finite_tuple(contact['position'], 3, 'contact position')
    if (grasping.distance_from_origin(pre_point) > 0.40
            or grasping.distance_from_origin(contact_point) > 0.15):
        raise ValueError('demonstrated poses are too far from the object center')
    delta = tuple(
        target - start for start, target in zip(pre_point, contact_point)
    )
    distance = grasping.distance_from_origin(delta)
    if not 0.02 <= distance <= 0.25 or delta[0] <= 0.01:
        raise ValueError(
            'contact must be 2-25 cm ahead of pre-grasp toward the object'
        )
    if math.hypot(delta[1], delta[2]) > 0.03:
        raise ValueError('demonstrated contact must be a nearly straight approach')
    if grasping.approach_axis_error(pre['orientation'], delta) > 0.35:
        raise ValueError('gripper forward axis must face the contact point')
    if grasping.quaternion_angular_distance(
            pre['orientation'], contact['orientation']) > 0.10:
        raise ValueError('pre-grasp and contact wrist orientations differ')
    open_angle = float(template['gripper_open_degrees'])
    if not math.isfinite(open_angle) or open_angle <= 0.0:
        raise ValueError('recorded gripper opening must be positive and finite')
    if not isinstance(template.get('enabled', True), bool):
        raise ValueError('template enabled must be boolean')


def candidate_from_template(template, reference):
    """Place a demonstrated straight approach at a new object center."""
    validate_template(template)
    pre_point, pre_orientation = absolute_pose(reference, template['pregrasp'])
    contact_point, _ = absolute_pose(reference, template['contact'])
    direction_delta = tuple(
        end - start for start, end in zip(pre_point, contact_point)
    )
    distance = grasping.distance_from_origin(direction_delta)
    direction = tuple(value / distance for value in direction_delta)
    return {
        'taught': True,
        'template_name': template['name'],
        'template': template,
        'yaw_offset': 0.0,
        'approach_direction': direction,
        'pregrasp_point': pre_point,
        'grasp_point': contact_point,
        'orientation': pre_orientation,
        'taught_orientation': pre_orientation,
        'band_target_point': reference[0],
        'band_width': None,
    }


class GraspTemplateLibrary:
    """A versioned YAML library with atomic writes to a user-selected path."""

    def __init__(self, path):
        self.path = os.path.abspath(os.path.expanduser(path))
        self.classes = {}
        self.load()

    def load(self):
        if not os.path.exists(self.path):
            self.classes = {}
            return
        try:
            with open(self.path, 'r', encoding='utf-8') as stream:
                document = yaml.safe_load(stream) or {}
        except yaml.YAMLError as error:
            raise ValueError('invalid grasp template YAML: {}'.format(error))
        if not isinstance(document, dict) or document.get('version') != 2:
            if isinstance(document, dict) and document.get('version') == 1:
                raise ValueError(
                    'grasp template file uses obsolete version 1 band '
                    'references; move it aside and reteach object-center grasps'
                )
            raise ValueError('grasp template file must have version: 2')
        classes = document.get('classes', {})
        if not isinstance(classes, dict):
            raise ValueError('grasp template classes must be a mapping')
        checked = {}
        for label, templates in classes.items():
            key = grasping.normalize_label(label)
            if not key or key in checked or not isinstance(templates, list):
                raise ValueError('invalid or duplicate grasp template class')
            names = set()
            for template in templates:
                validate_template(template)
                if template['name'] in names:
                    raise ValueError('duplicate grasp template name for ' + key)
                names.add(template['name'])
            checked[key] = templates
        self.classes = checked

    def templates_for(self, object_class):
        return tuple(
            template for template in self.classes.get(
                grasping.normalize_label(object_class), ()
            ) if template.get('enabled', True)
        )

    def add(self, object_class, template):
        key = grasping.normalize_label(object_class)
        if not key:
            raise ValueError('object class must not be empty')
        validate_template(template)
        if any(item['name'] == template['name']
               for item in self.classes.get(key, ())):
            raise ValueError('template name already exists for ' + key)
        updated = {label: list(items) for label, items in self.classes.items()}
        updated.setdefault(key, []).append(template)
        parent = os.path.dirname(self.path)
        os.makedirs(parent, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                    'w', encoding='utf-8', dir=parent, prefix='.grasp-',
                    suffix='.yaml', delete=False) as stream:
                temporary = stream.name
                yaml.safe_dump(
                    {'version': 2, 'classes': updated}, stream,
                    sort_keys=True, default_flow_style=False,
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)
        self.classes = updated
