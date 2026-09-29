"""Encode D1 commands and decode feedback carried by ArmString."""

from dataclasses import dataclass
import json
import math
import threading


COMMAND_ADDRESS = 1
FEEDBACK_ADDRESS = 2
RESULT_ADDRESS = 3

SET_JOINT = 1
SET_ALL_JOINTS = 2
SET_JOINT_ENABLED = 4
SET_ALL_ENABLED = 5
SET_POWER = 6
ZERO_ARM = 7

JOINT_ANGLES = 1
ARM_STATUS = 3
MOTOR_STATUS = 4

RECEIVE_RESULT = 1
EXECUTE_RESULT = 2

JOINT_COUNT = 7
SMOOTH_10_HZ = 0
TRAJECTORY = 1

# Mechanical angle limits published for D1-550 joints J0 through J5.
# J6 is the gripper and is deliberately left unbounded here because its
# command units/range differ between D1 variants.
JOINT_LIMITS_DEGREES = (
    (-135.0, 135.0),
    (-90.0, 90.0),
    (-90.0, 90.0),
    (-135.0, 135.0),
    (-90.0, 90.0),
    (-135.0, 135.0),
    None,
)


class ProtocolError(ValueError):
    """Raised when a D1 command or feedback payload is invalid."""


class SequenceGenerator:
    """Generate process-local uint32 command sequence numbers."""

    def __init__(self, start=1):
        if not isinstance(start, int) or isinstance(start, bool):
            raise ProtocolError('sequence start must be an integer')
        if start < 1 or start > 0xFFFFFFFF:
            raise ProtocolError('sequence start must be in [1, 4294967295]')
        self._next_value = start
        self._lock = threading.Lock()

    def next(self):
        """Return the next sequence number, wrapping to one."""
        with self._lock:
            value = self._next_value
            self._next_value = 1 if value == 0xFFFFFFFF else value + 1
            return value


@dataclass(frozen=True)
class JointAnglesFeedback:
    """Decoded joint-angle feedback."""

    sequence: int
    angles_degrees: tuple
    raw_json: str


@dataclass(frozen=True)
class ArmStatusFeedback:
    """Decoded overall arm status feedback."""

    sequence: int
    enabled: bool
    powered: bool
    healthy: bool
    raw_json: str


@dataclass(frozen=True)
class MotorStatusFeedback:
    """Decoded per-motor status feedback."""

    sequence: int
    motor_ok: tuple
    raw_json: str


@dataclass(frozen=True)
class CommandResultFeedback:
    """Decoded command receipt or execution result."""

    sequence: int
    stage: int
    success: bool
    raw_json: str


@dataclass(frozen=True)
class UnknownFeedback:
    """A valid envelope whose address/function is not yet supported."""

    sequence: int
    address: int
    function_code: int
    data: dict
    raw_json: str


def _require_integer(value, field_name, minimum=None, maximum=None):
    if not isinstance(value, int) or isinstance(value, bool):
        raise ProtocolError('{} must be an integer'.format(field_name))
    if minimum is not None and value < minimum:
        raise ProtocolError('{} is below {}'.format(field_name, minimum))
    if maximum is not None and value > maximum:
        raise ProtocolError('{} is above {}'.format(field_name, maximum))
    return value


def _require_number(value, field_name):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ProtocolError('{} must be numeric'.format(field_name))
    value = float(value)
    if not math.isfinite(value):
        raise ProtocolError('{} must be finite'.format(field_name))
    return value


def _require_flag(value, field_name):
    if value not in (0, 1) or isinstance(value, bool):
        raise ProtocolError('{} must be 0 or 1'.format(field_name))
    return bool(value)


def _validate_sequence(sequence):
    return _require_integer(sequence, 'sequence', 1, 0xFFFFFFFF)


def _validate_joint_id(joint_id):
    return _require_integer(joint_id, 'joint_id', 0, JOINT_COUNT - 1)


def _validate_angle(joint_id, angle_degrees, enforce_limits):
    angle = _require_number(angle_degrees, 'angle_degrees')
    if enforce_limits:
        limits = JOINT_LIMITS_DEGREES[joint_id]
        if limits is not None and not limits[0] <= angle <= limits[1]:
            raise ProtocolError(
                'joint {} angle {} is outside [{}, {}] degrees'.format(
                    joint_id, angle, limits[0], limits[1]
                )
            )
    return angle


def _encode(sequence, function_code, data=None):
    payload = {
        'seq': _validate_sequence(sequence),
        'address': COMMAND_ADDRESS,
        'funcode': function_code,
    }
    if data is not None:
        payload['data'] = data
    return json.dumps(payload, separators=(',', ':'), allow_nan=False)


def set_joint_command(
        sequence, joint_id, angle_degrees, enforce_limits=True):
    """Build a single-joint command; D1 requires delay_ms to be zero."""
    joint_id = _validate_joint_id(joint_id)
    angle = _validate_angle(joint_id, angle_degrees, enforce_limits)
    return _encode(
        sequence,
        SET_JOINT,
        {'id': joint_id, 'angle': angle, 'delay_ms': 0},
    )


def set_joint_angles_command(
        sequence, angles_degrees, mode=TRAJECTORY, enforce_limits=True):
    """Build a seven-joint command in D1 joint order."""
    if len(angles_degrees) != JOINT_COUNT:
        raise ProtocolError('exactly seven joint angles are required')
    mode = _require_integer(mode, 'mode', SMOOTH_10_HZ, TRAJECTORY)
    angles = [
        _validate_angle(index, value, enforce_limits)
        for index, value in enumerate(angles_degrees)
    ]
    data = {'mode': mode}
    data.update({
        'angle{}'.format(index): angle
        for index, angle in enumerate(angles)
    })
    return _encode(sequence, SET_ALL_JOINTS, data)


def set_joint_enabled_command(sequence, joint_id, enabled):
    """Build a single-joint enable/release command."""
    joint_id = _validate_joint_id(joint_id)
    if not isinstance(enabled, bool):
        raise ProtocolError('enabled must be a boolean')
    return _encode(
        sequence,
        SET_JOINT_ENABLED,
        {'id': joint_id, 'mode': int(enabled)},
    )


def set_arm_enabled_command(sequence, enabled):
    """Build an all-joint enable/release command."""
    if not isinstance(enabled, bool):
        raise ProtocolError('enabled must be a boolean')
    return _encode(sequence, SET_ALL_ENABLED, {'mode': int(enabled)})


def set_power_command(sequence, powered):
    """Build a motor-power command."""
    if not isinstance(powered, bool):
        raise ProtocolError('powered must be a boolean')
    return _encode(sequence, SET_POWER, {'power': int(powered)})


def zero_arm_command(sequence):
    """Build a return-to-zero command."""
    return _encode(sequence, ZERO_ARM)


def _load_envelope(raw_json):
    if not isinstance(raw_json, str):
        raise ProtocolError('feedback must be a string')
    try:
        envelope = json.loads(raw_json)
    except (TypeError, ValueError) as error:
        raise ProtocolError('feedback is not valid JSON: {}'.format(error))
    if not isinstance(envelope, dict):
        raise ProtocolError('feedback JSON must be an object')
    sequence = _require_integer(
        envelope.get('seq'), 'seq', 0, 0xFFFFFFFF
    )
    address = _require_integer(envelope.get('address'), 'address')
    function_code = _require_integer(envelope.get('funcode'), 'funcode')
    data = envelope.get('data', {})
    if not isinstance(data, dict):
        raise ProtocolError('data must be an object')
    return sequence, address, function_code, data


def parse_feedback(raw_json):
    """Decode one JSON payload received from /arm_Feedback."""
    sequence, address, function_code, data = _load_envelope(raw_json)

    if address == FEEDBACK_ADDRESS and function_code == JOINT_ANGLES:
        angles = tuple(
            _require_number(data.get('angle{}'.format(index)),
                            'data.angle{}'.format(index))
            for index in range(JOINT_COUNT)
        )
        return JointAnglesFeedback(sequence, angles, raw_json)

    if address == FEEDBACK_ADDRESS and function_code == ARM_STATUS:
        return ArmStatusFeedback(
            sequence,
            _require_flag(data.get('enable_status'), 'data.enable_status'),
            _require_flag(data.get('power_status'), 'data.power_status'),
            _require_flag(data.get('error_status'), 'data.error_status'),
            raw_json,
        )

    if address == FEEDBACK_ADDRESS and function_code == MOTOR_STATUS:
        status = tuple(
            _require_flag(data.get('motor{}_status'.format(index)),
                          'data.motor{}_status'.format(index))
            for index in range(JOINT_COUNT)
        )
        return MotorStatusFeedback(sequence, status, raw_json)

    if address == RESULT_ADDRESS and function_code in (
            RECEIVE_RESULT, EXECUTE_RESULT):
        field_name = (
            'recv_status'
            if function_code == RECEIVE_RESULT
            else 'exec_status'
        )
        return CommandResultFeedback(
            sequence,
            function_code,
            _require_flag(data.get(field_name), 'data.' + field_name),
            raw_json,
        )

    return UnknownFeedback(
        sequence, address, function_code, data, raw_json
    )
