import pytest

from dog_voice_commands.command_parser import CommandVocabulary
from dog_voice_commands.posture import (DAMPED, LYING, OTHER, SITTING,
                                        STANDING, classify)


@pytest.mark.parametrize('mode,height,expected', [
    (0, 0.32, STANDING),
    (0, 0.0816, LYING),    # Observed on a Go2 lying down: idle mode, low body
    (1, 0.08, LYING),
    (1, 0.32, STANDING),
    (3, 0.30, STANDING),
    (10, 0.20, SITTING),
    (5, 0.08, LYING),
    (6, 0.08, LYING),      # StandDown: joints locked, low
    (6, 0.32, STANDING),   # StandUp: joints locked, high
    (7, 0.05, DAMPED),
    (12, 0.30, OTHER),
])
def test_classify(mode, height, expected):
    assert classify(mode, height) == expected


def test_lying_height_threshold_configurable():
    assert classify(6, 0.25, lying_height_m=0.3) == LYING


def _vocab(stand_up):
    return CommandVocabulary.from_dict({
        'wake_phrases': ['hey dog'],
        'commands': {'stand_up': dict(phrases=['stand up'], **stand_up)},
    })


def test_stand_up_steps():
    vocab = CommandVocabulary.from_yaml(
        __file__.replace('test/test_posture.py', 'config/commands.yaml'))
    stand_up = {c.name: c for c in vocab.commands}['stand_up']
    assert stand_up.needs_posture
    assert stand_up.steps_for(SITTING) == (1010,)
    assert stand_up.steps_for(LYING) == (1004, 1002)
    assert stand_up.steps_for(STANDING) is None
    assert stand_up.steps_for(DAMPED) is None
    assert stand_up.steps_for(None) is None


def test_unconditional_command_ignores_posture():
    vocab = CommandVocabulary.from_dict({
        'wake_phrases': ['hey dog'],
        'commands': {'sit': {'api_id': 1009, 'phrases': ['sit']}},
    })
    assert vocab.commands[0].steps_for(None) == (1009,)
    assert not vocab.commands[0].needs_posture


def test_single_int_step_allowed():
    cmd = _vocab({'from_posture': {'sitting': 1010}}).commands[0]
    assert cmd.steps_for(SITTING) == (1010,)


@pytest.mark.parametrize('spec', [
    {'api_id': 1, 'from_posture': {'sitting': [1010]}},  # both
    {},                                                 # neither
    {'from_posture': {}},                               # empty table
    {'from_posture': {'flying': [1]}},                  # unknown posture
    {'from_posture': {'sitting': []}},                  # no steps
])
def test_invalid_from_posture(spec):
    with pytest.raises(ValueError):
        _vocab(spec)
