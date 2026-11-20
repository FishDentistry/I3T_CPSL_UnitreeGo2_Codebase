"""Grammar-constrained offline speech recognition using Vosk."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .audio import TARGET_SAMPLE_RATE
from .command_parser import CommandVocabulary

DEFAULT_MODEL_PATH = os.path.expanduser(
    '~/.local/share/dog_voice_commands/models/vosk-model-small-en-us-0.15')


@dataclass
class Word:
    word: str
    start: float  # Seconds since the start of the audio stream.
    end: float
    conf: float


@dataclass
class Transcript:
    text: str
    words: List[Word] = field(default_factory=list)

    @property
    def mean_confidence(self) -> float:
        if not self.words:
            return 0.0
        return float(np.mean([w.conf for w in self.words]))

    @property
    def span(self) -> Optional[Tuple[float, float]]:
        if not self.words:
            return None
        return self.words[0].start, self.words[-1].end


class GrammarRecognizer:
    """Vosk recognizer restricted to the command vocabulary.

    Restricting the decoder to a short list of phrases (plus ``[unk]``) makes
    recognition far more robust than open-vocabulary transcription and keeps
    CPU use low enough for the Go2 expansion board. Runtime grammars require
    a Vosk model with a dynamic graph, such as the *small* models.
    """

    def __init__(self, model_path: str, vocabulary: CommandVocabulary,
                 logger=None) -> None:
        from vosk import KaldiRecognizer, Model, SetLogLevel

        SetLogLevel(-1)
        model_path = os.path.expanduser(model_path)
        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f'Vosk model not found at {model_path}. See the package '
                f'README for download instructions.')

        self._model = Model(model_path)
        # Returns the word's symbol ID, or -1 if the model cannot recognize
        # it. (The Python binding names it vosk_model_find_word.)
        unknown = [w for w in vocabulary.words()
                   if self._model.vosk_model_find_word(w) < 0]
        if unknown and logger is not None:
            logger.warn(
                'The speech model does not know these words, so phrases '
                f'using them are disabled: {", ".join(unknown)}')
        self.vocabulary = (vocabulary.without_words(unknown)
                           if unknown else vocabulary)
        self.unknown_words = unknown

        self._recognizer = KaldiRecognizer(
            self._model, TARGET_SAMPLE_RATE,
            json.dumps(self.vocabulary.grammar()))
        self._recognizer.SetWords(True)

    def accept(self, audio: np.ndarray) -> Optional[Transcript]:
        """Feed audio; return a transcript when an utterance ends."""
        if self._recognizer.AcceptWaveform(
                np.asarray(audio, dtype=np.int16).tobytes()):
            return self._parse(self._recognizer.Result())
        return None

    def flush(self) -> Optional[Transcript]:
        return self._parse(self._recognizer.FinalResult())

    def reset(self) -> None:
        self._recognizer.Reset()

    @staticmethod
    def _parse(raw: str) -> Optional[Transcript]:
        result = json.loads(raw)
        text = result.get('text', '').strip()
        if not text or text == '[unk]':
            return None
        words = [Word(w['word'], float(w['start']), float(w['end']),
                      float(w.get('conf', 1.0)))
                 for w in result.get('result', [])]
        return Transcript(text=text, words=words)
