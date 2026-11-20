"""Wake-phrase and command parsing for transcribed speech.

This module is deliberately free of ROS, audio, and model dependencies so the
matching rules can be unit tested in isolation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import yaml

from .posture import POSTURES


def normalize(text: str) -> str:
    """Lower-case text, strip punctuation, and collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z' ]+", ' ', text)
    return ' '.join(text.split())


@dataclass(frozen=True)
class CommandSpec:
    """One spoken command and the Go2 sport-mode action(s) it triggers.

    A command is either *unconditional* (``api_id`` set: always sends that
    one action) or *posture-dependent* (``from_posture`` set: runs only when
    the robot is in one of the listed postures, sending that posture's
    sequence of API IDs in order).
    """

    name: str
    api_id: Optional[int]
    phrases: Tuple[str, ...]
    from_posture: Optional[Tuple[Tuple[str, Tuple[int, ...]], ...]] = None

    @property
    def needs_posture(self) -> bool:
        return self.from_posture is not None

    @property
    def allowed_postures(self) -> Tuple[str, ...]:
        return tuple(p for p, _ in (self.from_posture or ()))

    def steps_for(self, posture: Optional[str]) -> Optional[Tuple[int, ...]]:
        """Return the API IDs to send, or ``None`` if not allowed now."""
        if self.from_posture is None:
            return (self.api_id,)
        for name, steps in self.from_posture:
            if name == posture:
                return steps
        return None


@dataclass(frozen=True)
class ParseResult:
    """Outcome of feeding one transcribed utterance to the parser.

    ``kind`` is one of:

    * ``'ignored'``  - no wake phrase and not armed, or unrecognized text.
    * ``'armed'``    - the wake phrase was heard on its own; a command is
      expected within the arm timeout.
    * ``'command'``  - a complete command was recognized.
    * ``'expired'``  - a command was heard, but the arm window had lapsed.
    """

    kind: str
    command: Optional[CommandSpec] = None
    text: str = ''
    #: True when the command completes a previously armed wake phrase, so
    #: the caller should include the wake-phrase audio for verification.
    continues_wake: bool = False


@dataclass
class CommandVocabulary:
    """Wake phrases and commands loaded from ``commands.yaml``."""

    wake_phrases: Tuple[str, ...]
    commands: Tuple[CommandSpec, ...]

    @classmethod
    def from_dict(cls, data: dict) -> 'CommandVocabulary':
        wake = tuple(normalize(p) for p in data.get('wake_phrases', []))
        wake = tuple(p for p in wake if p)
        if not wake:
            raise ValueError('commands file must define at least one '
                             'wake phrase')

        raw_commands = data.get('commands') or {}
        if not raw_commands:
            raise ValueError('commands file must define at least one command')

        commands: List[CommandSpec] = []
        seen_phrases: Dict[str, str] = {}
        for name, spec in raw_commands.items():
            has_id = 'api_id' in spec
            has_posture = 'from_posture' in spec
            if has_id == has_posture:
                raise ValueError(f'command {name!r} must define exactly one '
                                 f'of api_id or from_posture')
            from_posture = None
            if has_posture:
                table = spec['from_posture'] or {}
                if not table:
                    raise ValueError(f'command {name!r} has an empty '
                                     f'from_posture table')
                entries = []
                for posture, steps in table.items():
                    if posture not in POSTURES:
                        raise ValueError(
                            f'command {name!r}: unknown posture '
                            f'{posture!r} (expected one of {POSTURES})')
                    if isinstance(steps, int):
                        steps = [steps]
                    if not steps:
                        raise ValueError(f'command {name!r}: no actions '
                                         f'for posture {posture!r}')
                    entries.append((posture, tuple(int(x) for x in steps)))
                from_posture = tuple(entries)
            phrases = tuple(normalize(p) for p in spec.get('phrases', []))
            phrases = tuple(p for p in phrases if p)
            if not phrases:
                raise ValueError(f'command {name!r} has no phrases')
            for phrase in phrases:
                if phrase in seen_phrases:
                    raise ValueError(
                        f'phrase {phrase!r} is used by both '
                        f'{seen_phrases[phrase]!r} and {name!r}')
                seen_phrases[phrase] = name
            commands.append(CommandSpec(
                name=str(name),
                api_id=int(spec['api_id']) if has_id else None,
                phrases=phrases, from_posture=from_posture))

        return cls(wake_phrases=wake, commands=tuple(commands))

    @classmethod
    def from_yaml(cls, path: str) -> 'CommandVocabulary':
        with open(path, 'r') as handle:
            return cls.from_dict(yaml.safe_load(handle) or {})

    def grammar(self) -> List[str]:
        """Return every phrase the recognizer should be allowed to output.

        Includes the wake phrase alone, the wake phrase followed by each
        command phrase, each command phrase alone (for the two-part
        "Hey Yeller ... sit" interaction), and ``[unk]`` so that unrelated
        speech is rejected instead of being forced onto the closest phrase.
        """
        phrases: List[str] = []
        for wake in self.wake_phrases:
            phrases.append(wake)
            for command in self.commands:
                for phrase in command.phrases:
                    phrases.append(f'{wake} {phrase}')
        for command in self.commands:
            phrases.extend(command.phrases)
        phrases.append('[unk]')
        # Preserve order while removing duplicates.
        return list(dict.fromkeys(phrases))

    def words(self) -> List[str]:
        """Return the unique words used by the vocabulary."""
        words: List[str] = []
        for phrase in self.grammar():
            if phrase == '[unk]':
                continue
            words.extend(phrase.split())
        return list(dict.fromkeys(words))

    def without_words(self, unknown: Iterable[str]) -> 'CommandVocabulary':
        """Return a copy without phrases that use any of ``unknown``.

        Used to drop phrases containing words the speech model cannot
        recognize. Raises ``ValueError`` if nothing usable remains.
        """
        unknown = set(unknown)

        def usable(phrase: str) -> bool:
            return not (set(phrase.split()) & unknown)

        wake = tuple(p for p in self.wake_phrases if usable(p))
        commands = []
        for command in self.commands:
            phrases = tuple(p for p in command.phrases if usable(p))
            if phrases:
                commands.append(CommandSpec(
                    command.name, command.api_id, phrases,
                    command.from_posture))
        if not wake:
            raise ValueError(
                'none of the wake phrases can be recognized by the speech '
                f'model (unknown words: {sorted(unknown)})')
        if not commands:
            raise ValueError(
                'none of the command phrases can be recognized by the '
                f'speech model (unknown words: {sorted(unknown)})')
        return CommandVocabulary(wake_phrases=wake, commands=tuple(commands))


@dataclass
class CommandParser:
    """Turn transcribed utterances into commands.

    Accepts either a single utterance ("hey yeller sit") or a wake phrase
    followed by a command in a separate utterance within
    ``arm_timeout_sec`` seconds.
    """

    vocabulary: CommandVocabulary
    arm_timeout_sec: float = 4.0
    _armed_until: Optional[float] = field(default=None, init=False)

    def __post_init__(self) -> None:
        # Longest wake phrase first so "hey yeller" wins over "hey".
        self._wake = sorted(self.vocabulary.wake_phrases,
                            key=len, reverse=True)
        self._lookup: Dict[str, CommandSpec] = {}
        for command in self.vocabulary.commands:
            for phrase in command.phrases:
                self._lookup[phrase] = command

    @property
    def armed(self) -> bool:
        return self._armed_until is not None

    def is_armed(self, now: float) -> bool:
        return self._armed_until is not None and now <= self._armed_until

    def disarm(self) -> None:
        self._armed_until = None

    def _split_wake(self, text: str) -> Tuple[bool, str]:
        for wake in self._wake:
            if text == wake:
                return True, ''
            if text.startswith(wake + ' '):
                return True, text[len(wake) + 1:]
        return False, text

    def feed(self, text: str, now: float) -> ParseResult:
        """Process one final transcription at monotonic time ``now``."""
        text = normalize(text)
        if not text:
            return ParseResult('ignored', text=text)

        has_wake, remainder = self._split_wake(text)

        if has_wake:
            if not remainder:
                self._armed_until = now + self.arm_timeout_sec
                return ParseResult('armed', text=text)
            command = self._lookup.get(remainder)
            self.disarm()
            if command is None:
                return ParseResult('ignored', text=text)
            return ParseResult('command', command=command, text=text)

        command = self._lookup.get(text)
        if command is None:
            # Unrelated speech cancels a pending wake phrase.
            self.disarm()
            return ParseResult('ignored', text=text)

        if self._armed_until is None:
            return ParseResult('ignored', command=command, text=text)

        in_window = now <= self._armed_until
        self.disarm()
        if not in_window:
            return ParseResult('expired', command=command, text=text)
        return ParseResult('command', command=command, text=text,
                           continues_wake=True)
