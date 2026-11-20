"""Exercise GrammarRecognizer against a stand-in for the vosk module.

The stand-in mirrors the real vosk Python API (vosk/__init__.py): Model has
vosk_model_find_word() returning -1 for unknown words, and KaldiRecognizer
takes (model, sample_rate, grammar_json).
"""

import json
import os
import sys
import types

import pytest

from dog_voice_commands.command_parser import CommandVocabulary

CONFIG = os.path.join(os.path.dirname(__file__), '..', 'config',
                      'commands.yaml')


@pytest.fixture
def fake_vosk(monkeypatch):
    module = types.ModuleType('vosk')
    known = {'hey', 'yellow', 'yell', 'er', 'sit', 'down', 'shake', 'hands',
             'lay', 'lie', 'stand', 'up'}  # 'yeller' deliberately unknown
    created = {}

    class Model:
        def __init__(self, model_path=None, model_name=None, lang=None):
            self.path = model_path

        def vosk_model_find_word(self, word):
            return 1 if word in known else -1

    class KaldiRecognizer:
        def __init__(self, *args):
            created['args'] = args
            self.words = False

        def SetWords(self, enable_words):
            self.words = enable_words

        def AcceptWaveform(self, data):
            return True

        def Result(self):
            return json.dumps({'text': 'hey yellow sit', 'result': [
                {'word': 'hey', 'start': 1.0, 'end': 1.2, 'conf': 1.0},
                {'word': 'yellow', 'start': 1.2, 'end': 1.6, 'conf': 0.9},
                {'word': 'sit', 'start': 1.7, 'end': 2.0, 'conf': 0.8}]})

        def FinalResult(self):
            return json.dumps({'text': ''})

        def Reset(self):
            pass

    module.Model = Model
    module.KaldiRecognizer = KaldiRecognizer
    module.SetLogLevel = lambda level: None
    monkeypatch.setitem(sys.modules, 'vosk', module)
    return created


def test_recognizer_filters_unknown_words(fake_vosk, tmp_path):
    from dog_voice_commands.recognizer import GrammarRecognizer

    warnings = []
    logger = types.SimpleNamespace(warn=warnings.append)
    vocab = CommandVocabulary.from_yaml(CONFIG)
    rec = GrammarRecognizer(str(tmp_path), vocab, logger=logger)

    assert rec.unknown_words == ['yeller']
    assert 'hey yeller' not in rec.vocabulary.wake_phrases
    assert 'hey yellow' in rec.vocabulary.wake_phrases
    assert warnings and 'yeller' in warnings[0]

    model, rate, grammar = fake_vosk['args']
    assert rate == 16000
    grammar = json.loads(grammar)
    assert 'hey yellow sit' in grammar and '[unk]' in grammar
    assert not any('yeller' in phrase for phrase in grammar)


def test_recognizer_parses_result(fake_vosk, tmp_path):
    import numpy as np
    from dog_voice_commands.recognizer import GrammarRecognizer

    vocab = CommandVocabulary.from_yaml(CONFIG)
    rec = GrammarRecognizer(str(tmp_path), vocab)
    transcript = rec.accept(np.zeros(1600, dtype=np.int16))
    assert transcript.text == 'hey yellow sit'
    assert transcript.span == (1.0, 2.0)
    assert abs(transcript.mean_confidence - 0.9) < 1e-9


def test_recognizer_missing_model(fake_vosk, tmp_path):
    from dog_voice_commands.recognizer import GrammarRecognizer

    vocab = CommandVocabulary.from_yaml(CONFIG)
    with pytest.raises(FileNotFoundError):
        GrammarRecognizer(str(tmp_path / 'missing'), vocab)
