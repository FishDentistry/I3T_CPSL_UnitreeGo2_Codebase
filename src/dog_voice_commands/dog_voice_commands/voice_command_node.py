"""ROS 2 node that turns owner-verified voice commands into Go2 actions.

Pipeline::

    microphone -> Vosk (grammar-restricted) -> wake/command parser
               -> speaker verification -> go2_interfaces/Go2Cmd

The node never talks to the robot directly. It publishes the sport-mode API
ID of the recognized action on ``command_topic`` (default
``/go2/sport_cmd``); the node that owns the robot connection forwards it.

Posture-dependent commands (such as "stand up") use the robot's
``SportModeState`` to decide whether, and how, to run.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Optional

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from go2_interfaces.msg import Go2Cmd
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String
from unitree_go.msg import SportModeState

from .audio import (TARGET_SAMPLE_RATE, MicrophoneStream, RingBuffer,
                    parse_device, rms_dbfs)
from .command_parser import (CommandParser, CommandVocabulary,
                             ParseResult)
from .posture import DEFAULT_LYING_HEIGHT_M, UNKNOWN, classify
from .recognizer import DEFAULT_MODEL_PATH, GrammarRecognizer, Transcript
from .speaker_verifier import (DEFAULT_PROFILE_PATH, DEFAULT_THRESHOLDS,
                               SpeakerProfile, SpeakerVerifier, load_backend)

#: Audio kept around word boundaries when cutting an utterance for
#: speaker verification.
SEGMENT_PAD_SEC = 0.15

#: Input quieter than this for SILENCE_WARN_SEC triggers a warning that the
#: microphone may be muted or the wrong device.
SILENCE_DBFS = -60.0
SILENCE_WARN_SEC = 10.0

#: Report an error if the microphone delivers nothing this long after opening.
NO_AUDIO_SEC = 5.0


def _default_commands_file() -> str:
    return os.path.join(get_package_share_directory('dog_voice_commands'),
                        'config', 'commands.yaml')


class VoiceCommandNode(Node):

    def __init__(self) -> None:
        super().__init__('voice_command_node')

        p = self.declare_parameter
        p('commands_file', '')
        p('vosk_model_path', DEFAULT_MODEL_PATH)
        p('audio_device', '')
        p('capture_sample_rate', TARGET_SAMPLE_RATE)
        p('command_topic', '/go2/sport_cmd')
        p('arm_timeout_sec', 4.0)
        p('cooldown_sec', 3.0)
        p('min_word_confidence', 0.6)
        p('speaker_verification', True)
        p('speaker_backend', 'speechbrain')
        p('speaker_profile_path', DEFAULT_PROFILE_PATH)
        p('speaker_threshold', -1.0)
        p('compute_device', 'cpu')
        p('start_enabled', True)
        p('dry_run', False)
        p('sport_state_topic', 'lf/sportmodestate')
        p('state_timeout_sec', 2.0)
        p('lying_height_m', DEFAULT_LYING_HEIGHT_M)
        p('step_delay_sec', 2.5)

        g = self.get_parameter
        commands_file = g('commands_file').value or _default_commands_file()
        self._cooldown = float(g('cooldown_sec').value)
        self._min_conf = float(g('min_word_confidence').value)
        self._dry_run = bool(g('dry_run').value)
        self._enabled = bool(g('start_enabled').value)
        self._state_timeout = float(g('state_timeout_sec').value)
        self._lying_height = float(g('lying_height_m').value)
        self._step_delay = float(g('step_delay_sec').value)
        self._posture = UNKNOWN
        self._posture_time = -float('inf')
        self._sequence_lock = threading.Lock()
        self._pending_steps = []  # Timers for queued sequence steps.

        # Speech recognition.
        vocabulary = CommandVocabulary.from_yaml(commands_file)
        self._recognizer = GrammarRecognizer(
            g('vosk_model_path').value, vocabulary,
            logger=self.get_logger())
        self._parser = CommandParser(self._recognizer.vocabulary,
                                     float(g('arm_timeout_sec').value))

        # Speaker verification.
        self._verifier: Optional[SpeakerVerifier] = None
        if bool(g('speaker_verification').value):
            self._verifier = self._load_verifier()
        else:
            self.get_logger().warn(
                'Speaker verification is DISABLED: anyone who says the wake '
                'phrase can command the robot.')

        # Audio.
        self._buffer = RingBuffer(seconds=20.0)
        self._mic = MicrophoneStream(
            device=parse_device(g('audio_device').value),
            capture_rate=int(g('capture_sample_rate').value))
        self._wake_audio: Optional[np.ndarray] = None
        self._utterance_start = 0
        self._last_command_time = -float('inf')

        # ROS interfaces.
        self._cmd_pub = self.create_publisher(
            Go2Cmd, g('command_topic').value, 10)
        self._event_pub = self.create_publisher(String, '~/events', 10)
        self.create_subscription(Bool, '~/enable', self._on_enable, 10)
        self.create_subscription(SportModeState,
                                 g('sport_state_topic').value,
                                 self._on_sport_state, 10)

        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._run, daemon=True)

        # Audio level monitoring, for diagnosing microphone problems.
        self._level_lock = threading.Lock()
        self._level_peak = float('-inf')
        self._quiet_since = None
        self._silence_warned = False
        self._level_pub = self.create_publisher(Float32, '~/audio_level', 10)
        self.create_timer(0.5, self._publish_level)

        self._log_vocabulary(vocabulary)
        self.get_logger().info(
            f'Publishing commands to {g("command_topic").value}'
            f'{" (DRY RUN: nothing will be sent)" if self._dry_run else ""}.')

    def _log_vocabulary(self, configured: CommandVocabulary) -> None:
        active = self._recognizer.vocabulary
        wake = ', '.join(f'"{w}"' for w in active.wake_phrases)
        dropped = [w for w in configured.wake_phrases
                   if w not in active.wake_phrases]
        if dropped:
            self.get_logger().info(
                'The speech model does not know every word in '
                + ', '.join(f'"{w}"' for w in dropped)
                + f', so it listens for the similar-sounding {wake} instead. '
                'Say the wake phrase normally; these are what the model '
                'hears it as.')
        else:
            self.get_logger().info(f'Wake phrases: {wake}.')
        self.get_logger().info(
            'Commands: ' + ', '.join(
                f'"{c.phrases[0]}"' for c in active.commands) + '.')

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def _load_verifier(self) -> SpeakerVerifier:
        g = self.get_parameter
        profile_path = os.path.expanduser(g('speaker_profile_path').value)
        if not os.path.isfile(profile_path):
            raise FileNotFoundError(
                f'No speaker profile at {profile_path}. Run '
                f'"ros2 run dog_voice_commands enroll_speaker" first, or set '
                f'speaker_verification:=false to accept any voice.')
        backend_name = g('speaker_backend').value
        threshold = float(g('speaker_threshold').value)
        if threshold < 0.0:
            threshold = DEFAULT_THRESHOLDS[backend_name]
        self.get_logger().info(
            f'Loading {backend_name} speaker model (threshold '
            f'{threshold:.2f})...')
        profile = SpeakerProfile.load(profile_path)
        backend = load_backend(backend_name, device=g('compute_device').value)
        self.get_logger().info(
            f'Speaker profile loaded ({profile.num_samples} samples).')
        return SpeakerVerifier(profile, backend, threshold)

    def start(self) -> None:
        self._mic.start()
        self.get_logger().info(
            f'Microphone: "{self._mic.device_name}" at '
            f'{self._mic.capture_rate} Hz. Input level is published on '
            f'{self.get_name()}/audio_level (dBFS).')
        if self._mic.capture_rate != self._mic.requested_rate:
            self.get_logger().info(
                f'Microphone does not support {self._mic.requested_rate} Hz; '
                f'capturing at {self._mic.capture_rate} Hz and resampling.')
        self._worker.start()

    def _track_level(self, chunk: np.ndarray) -> None:
        level = rms_dbfs(chunk)
        now = time.monotonic()
        with self._level_lock:
            self._level_peak = max(self._level_peak, level)
        if level >= SILENCE_DBFS:
            if self._silence_warned:
                self.get_logger().info('Microphone input detected again.')
            self._quiet_since = None
            self._silence_warned = False
            return
        if self._quiet_since is None:
            self._quiet_since = now
        elif (not self._silence_warned
              and now - self._quiet_since >= SILENCE_WARN_SEC):
            self._silence_warned = True
            self.get_logger().warn(
                f'Microphone input has been below {SILENCE_DBFS:.0f} dBFS '
                f'for {SILENCE_WARN_SEC:.0f} s. Check that the right device '
                f'is selected ("{self._mic.device_name}") and that it is not '
                f'muted (alsamixer).')

    def _publish_level(self) -> None:
        with self._level_lock:
            peak, self._level_peak = self._level_peak, float('-inf')
        if peak != float('-inf'):
            self._level_pub.publish(Float32(data=float(round(peak, 1))))

    def shutdown(self) -> None:
        self._stop.set()
        self._cancel_pending_steps()
        if self._worker.is_alive():
            self._worker.join(timeout=2.0)
        self._mic.stop()

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _on_sport_state(self, msg: SportModeState) -> None:
        self._posture = classify(int(msg.mode), float(msg.body_height),
                                 self._lying_height)
        self._posture_time = time.monotonic()

    def _current_posture(self) -> str:
        if time.monotonic() - self._posture_time > self._state_timeout:
            return UNKNOWN
        return self._posture

    def _on_enable(self, msg: Bool) -> None:
        if msg.data != self._enabled:
            self._enabled = msg.data
            self._parser.disarm()
            self._wake_audio = None
            if not msg.data:
                self._cancel_pending_steps()
            self.get_logger().info(
                'Voice commands ' + ('enabled' if msg.data else 'muted'))

    # ------------------------------------------------------------------
    # Audio processing
    # ------------------------------------------------------------------
    def _run(self) -> None:
        started = time.monotonic()
        received = 0
        warned_no_audio = False
        last_error = None
        while not self._stop.is_set():
            try:
                chunk = self._mic.read(timeout=0.5)
                if chunk is None:
                    if (received == 0 and not warned_no_audio
                            and time.monotonic() - started > NO_AUDIO_SEC):
                        warned_no_audio = True
                        self.get_logger().error(
                            f'No audio received from "{self._mic.device_name}" '
                            f'{NO_AUDIO_SEC:.0f} s after opening it. The '
                            f'device opened but is not delivering samples; '
                            f'see "No audio received" in the README '
                            f'troubleshooting section.')
                    continue
                if received == 0:
                    self.get_logger().info('Receiving audio from microphone.')
                received += 1
                self._buffer.append(chunk)
                self._track_level(chunk)
                transcript = self._recognizer.accept(chunk)
                if transcript is not None:
                    self._handle(transcript)
            except Exception as exc:  # Keep listening on any failure.
                message = f'{type(exc).__name__}: {exc}'
                if message != last_error:  # Do not flood the log.
                    self.get_logger().error(
                        f'Error in audio processing: {message}')
                    last_error = message

    def _segment(self, transcript: Transcript) -> np.ndarray:
        """Cut the audio for ``transcript`` out of the ring buffer."""
        total = self._buffer.total_samples
        utt_start = self._utterance_start
        self._utterance_start = total
        span = transcript.span
        if span is None:
            return self._buffer.extract(utt_start, total)
        start, end = span
        rate = TARGET_SAMPLE_RATE
        # Vosk reports times from the start of the stream. Fall back to
        # utterance-relative times if they do not fit the current utterance.
        if start * rate < utt_start - rate:
            start += utt_start / rate
            end += utt_start / rate
        return self._buffer.extract_seconds(start - SEGMENT_PAD_SEC,
                                            end + SEGMENT_PAD_SEC)

    def _handle(self, transcript: Transcript) -> None:
        now = time.monotonic()
        audio = self._segment(transcript)

        if not self._enabled:
            return

        # Vosk reports speech outside the grammar as "[unk]". Keep it in the
        # text, so "hey yeller [unk]" is not mistaken for a bare wake phrase,
        # but leave it out of the confidence score.
        if all(token == '[unk]' for token in transcript.text.split()):
            self._event('unrecognized', transcript.text)
            return
        known = [w for w in transcript.words if w.word != '[unk]']
        transcript = Transcript(text=transcript.text,
                                words=known or transcript.words)

        conf = transcript.mean_confidence
        if transcript.words and conf < self._min_conf:
            self._event('low_confidence', transcript.text, confidence=conf)
            return

        result: ParseResult = self._parser.feed(transcript.text, now)

        if result.kind == 'armed':
            self._wake_audio = audio
            self._event('armed', result.text, confidence=conf)
            return

        if result.kind != 'command':
            if result.kind == 'expired':
                self._event('expired', result.text, confidence=conf)
            elif result.command is not None:
                self._event('ignored', result.text, confidence=conf,
                            reason='command without the wake phrase')
            else:
                self._event('ignored', result.text, confidence=conf,
                            reason='not a recognized command')
            self._wake_audio = None
            return

        if result.continues_wake and self._wake_audio is not None:
            audio = np.concatenate([self._wake_audio, audio])
        self._wake_audio = None
        command = result.command

        score = None
        if self._verifier is not None:
            verification = self._verifier.verify(audio)
            score = verification.score
            if not verification.accepted:
                self.get_logger().info(
                    f'Rejected "{result.text}": {verification.reason}')
                self._event('rejected_speaker', result.text,
                            command=command.name, confidence=conf,
                            speaker_score=score,
                            reason=verification.reason)
                return

        if now < self._last_command_time + self._cooldown:
            self._event('cooldown', result.text, command=command.name,
                        confidence=conf, speaker_score=score)
            return

        posture = None
        if command.needs_posture:
            posture = self._current_posture()
            if posture == UNKNOWN:
                self.get_logger().warn(
                    f'Ignoring "{command.name}": no recent robot state on '
                    f'{self.get_parameter("sport_state_topic").value}')
                self._event('no_robot_state', result.text,
                            command=command.name, confidence=conf,
                            speaker_score=score)
                return
        steps = command.steps_for(posture)
        if steps is None:
            allowed = ', '.join(command.allowed_postures)
            self.get_logger().info(
                f'Ignoring "{command.name}": robot is {posture}; it only '
                f'runs when the robot is {allowed}.')
            self._event('wrong_posture', result.text, command=command.name,
                        posture=posture, confidence=conf,
                        speaker_score=score)
            return

        # The cooldown starts after the last step of a sequence.
        self._last_command_time = now + self._step_delay * (len(steps) - 1)

        score_txt = '' if score is None else f', speaker {score:.2f}'
        posture_txt = '' if posture is None else f', from {posture}'
        self.get_logger().info(
            f'Command "{command.name}" (api_ids '
            f'{", ".join(str(s) for s in steps)}, conf {conf:.2f}'
            f'{score_txt}{posture_txt})'
            f'{" [dry run]" if self._dry_run else ""}')
        self._run_steps(steps)
        self._event('dry_run' if self._dry_run else 'accepted', result.text,
                    command=command.name, api_ids=list(steps),
                    posture=posture, confidence=conf, speaker_score=score)

    def _publish_step(self, api_id: int) -> None:
        if self._dry_run or not self._enabled:
            return
        msg = Go2Cmd()
        msg.cmd = api_id
        self._cmd_pub.publish(msg)

    def _run_steps(self, steps) -> None:
        """Send the first step now and the rest ``step_delay_sec`` apart."""
        self._cancel_pending_steps()
        self._publish_step(steps[0])
        with self._sequence_lock:
            for i, api_id in enumerate(steps[1:], start=1):
                timer = threading.Timer(self._step_delay * i,
                                        self._publish_step, args=(api_id,))
                timer.daemon = True
                self._pending_steps.append(timer)
                timer.start()

    def _cancel_pending_steps(self) -> None:
        with self._sequence_lock:
            for timer in self._pending_steps:
                timer.cancel()
            self._pending_steps = []

    def _event(self, kind: str, text: str, **fields) -> None:
        payload = {'event': kind, 'text': text}
        payload.update({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in fields.items() if v is not None})
        self._event_pub.publish(String(data=json.dumps(payload)))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = VoiceCommandNode()
        node.start()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
