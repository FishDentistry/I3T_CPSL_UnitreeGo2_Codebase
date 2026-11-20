"""Classify the Go2's posture from its sport-mode state.

The Go2 publishes ``unitree_go/msg/SportModeState`` on ``sportmodestate`` and
(at lower rate) ``lf/sportmodestate``. Its ``mode`` field is documented by
Unitree as::

    0 idle (default stand)   5 lieDown          10 sit
    1 balanceStand           6 jointLock        11 frontFlip
    2 pose                   7 damping          12 frontJump
    3 locomotion             8 recoveryStand    13 frontPounce
    4 reserve                9 reserve

``StandDown`` (the "lay down" action) and ``StandUp`` both leave the robot
in a joint-locked pose, low and high respectively, so ``jointLock`` is
split into lying or standing using ``body_height``.
"""

from __future__ import annotations

STANDING = 'standing'
SITTING = 'sitting'
LYING = 'lying'
DAMPED = 'damped'
OTHER = 'other'
UNKNOWN = 'unknown'

#: Postures a command's ``from_posture`` table may name.
POSTURES = (STANDING, SITTING, LYING, DAMPED, OTHER)

MODE_IDLE = 0
MODE_BALANCE_STAND = 1
MODE_POSE = 2
MODE_LOCOMOTION = 3
MODE_LIE_DOWN = 5
MODE_JOINT_LOCK = 6
MODE_DAMPING = 7
MODE_SIT = 10

_STANDING_MODES = {MODE_IDLE, MODE_BALANCE_STAND, MODE_POSE,
                   MODE_LOCOMOTION}

#: Default body height (m) below which a joint-locked robot counts as lying.
#: The Go2 stands at roughly 0.3 m and lies at roughly 0.1 m.
DEFAULT_LYING_HEIGHT_M = 0.2


def classify(mode: int, body_height: float,
             lying_height_m: float = DEFAULT_LYING_HEIGHT_M) -> str:
    """Return the posture name for a ``SportModeState`` reading."""
    if mode in _STANDING_MODES:
        return STANDING
    if mode == MODE_SIT:
        return SITTING
    if mode == MODE_LIE_DOWN:
        return LYING
    if mode == MODE_JOINT_LOCK:
        return LYING if body_height < lying_height_m else STANDING
    if mode == MODE_DAMPING:
        return DAMPED
    return OTHER
