import os

import pytest

from dog_voice_commands.command_parser import (CommandParser,
                                               CommandVocabulary, normalize)

CONFIG = os.path.join(os.path.dirname(__file__), '..', 'config',
                      'commands.yaml')


@pytest.fixture
def vocab():
    return CommandVocabulary.from_yaml(CONFIG)


@pytest.fixture
def parser(vocab):
    return CommandParser(vocab, arm_timeout_sec=4.0)


def test_normalize():
    assert normalize('  Hey, Yeller!  SIT. ') == 'hey yeller sit'


def test_packaged_vocabulary(vocab):
    names = {c.name: c.api_id for c in vocab.commands}
    assert names == {'sit': 1009, 'shake': 1016, 'lay_down': 1005,
                     'stand_up': None}
    assert 'hey yeller' in vocab.wake_phrases


def test_grammar_contents(vocab):
    grammar = vocab.grammar()
    assert grammar[-1] == '[unk]'
    assert 'hey yeller sit' in grammar
    assert 'hey yeller lay down' in grammar
    assert 'hey yeller' in grammar
    assert 'shake' in grammar
    assert len(grammar) == len(set(grammar))


@pytest.mark.parametrize('text,name', [
    ('hey yeller sit', 'sit'),
    ('Hey Yeller, shake!', 'shake'),
    ('hey yeller lay down', 'lay_down'),
    ('hey yeller lie down', 'lay_down'),
    ('hey yellow sit down', 'sit'),
])
def test_single_utterance(parser, text, name):
    result = parser.feed(text, now=0.0)
    assert result.kind == 'command'
    assert result.command.name == name
    assert not result.continues_wake


def test_command_without_wake_is_ignored(parser):
    assert parser.feed('sit', now=0.0).kind == 'ignored'


def test_two_part_interaction(parser):
    assert parser.feed('hey yeller', now=10.0).kind == 'armed'
    result = parser.feed('shake', now=12.0)
    assert result.kind == 'command'
    assert result.command.name == 'shake'
    assert result.continues_wake
    # Arming is consumed by the command.
    assert parser.feed('sit', now=13.0).kind == 'ignored'


def test_arm_window_expires(parser):
    parser.feed('hey yeller', now=0.0)
    assert parser.feed('sit', now=4.5).kind == 'expired'
    assert not parser.armed


def test_unrelated_speech_disarms(parser):
    parser.feed('hey yeller', now=0.0)
    parser.feed('hey yeller banana', now=1.0)
    assert parser.feed('sit', now=1.5).kind == 'ignored'


def test_wake_must_be_prefix(parser):
    assert parser.feed('sit hey yeller', now=0.0).kind == 'ignored'


def test_without_words_drops_phrases(vocab):
    reduced = vocab.without_words(['yeller'])
    assert reduced.wake_phrases == ('hey yellow', 'hey yell er')
    reduced = vocab.without_words(['shake'])
    assert {c.name for c in reduced.commands} == {'sit', 'lay_down',
                                                  'stand_up'}


def test_without_words_all_wake_unknown(vocab):
    with pytest.raises(ValueError):
        vocab.without_words(['hey'])


def test_duplicate_phrase_rejected():
    with pytest.raises(ValueError):
        CommandVocabulary.from_dict({
            'wake_phrases': ['hey dog'],
            'commands': {
                'a': {'api_id': 1, 'phrases': ['down']},
                'b': {'api_id': 2, 'phrases': ['Down']},
            },
        })


def test_missing_api_id_rejected():
    with pytest.raises(ValueError):
        CommandVocabulary.from_dict({
            'wake_phrases': ['hey dog'],
            'commands': {'a': {'phrases': ['sit']}},
        })


def test_stand_up_phrase(parser):
    result = parser.feed('hey yeller stand up', now=0.0)
    assert result.kind == 'command'
    assert result.command.name == 'stand_up'
