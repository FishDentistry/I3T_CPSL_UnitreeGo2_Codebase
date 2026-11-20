"""Command-line tools: speaker enrollment, verification, and device listing.

These tools do not need a running ROS graph; they are installed as ROS
executables only for convenience (``ros2 run dog_voice_commands ...``).
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from typing import List

import numpy as np
import yaml

from .audio import (parse_device, read_wav, record_seconds, rms_dbfs,
                    trim_silence, write_wav)
from .speaker_verifier import (DEFAULT_PROFILE_PATH, DEFAULT_THRESHOLDS,
                               SpeakerProfile, SpeakerVerifier, load_backend)

DEFAULT_PROMPTS = [
    'Hey Yeller, sit',
    'Hey Yeller, shake',
    'Hey Yeller, lay down',
    'Hey Yeller, stand up',
    'Hey Yeller',
]

#: Recordings quieter than this are rejected as "no speech captured".
MIN_LEVEL_DBFS = -45.0

#: Per-machine parameter overrides, loaded by voice_commands.launch.py after
#: the packaged defaults. Kept outside the repository.
USER_PARAMS_PATH = os.path.expanduser(
    '~/.local/share/dog_voice_commands/params.yaml')
NODE_NAME = 'voice_command_node'


def update_user_params(values: dict, path: str = USER_PARAMS_PATH) -> None:
    """Merge ``values`` into the per-machine parameter file."""
    data = {}
    if os.path.isfile(path):
        with open(path, 'r') as handle:
            data = yaml.safe_load(handle) or {}
    params = (data.setdefault(NODE_NAME, {})
              .setdefault('ros__parameters', {}))
    params.update(values)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as handle:
        handle.write('# Per-machine settings for dog_voice_commands, written '
                     'by enroll_speaker /\n# verify_speaker. Loaded after the '
                     'packaged defaults; edit freely.\n')
        yaml.safe_dump(data, handle, default_flow_style=False)
    print(f'Saved {", ".join(sorted(values))} to {path}')


def _add_audio_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '--device', default='',
        help='Input device index or name substring (see list_audio_devices).'
             ' Default: system default input.')
    parser.add_argument(
        '--capture-rate', type=int, default=16000,
        help='Rate to open the microphone at; audio is resampled to 16 kHz. '
             'Use 44100 or 48000 if the device rejects 16000.')


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--backend', default='speechbrain',
                        choices=sorted(DEFAULT_THRESHOLDS),
                        help='Speaker-embedding backend (default: '
                             'speechbrain).')
    parser.add_argument('--profile', default=DEFAULT_PROFILE_PATH,
                        help=f'Profile path (default: {DEFAULT_PROFILE_PATH}).')
    parser.add_argument('--compute-device', default='cpu',
                        help='Torch device for the embedding model, e.g. '
                             '"cpu" or "cuda".')


def _record_prompted(prompt: str, seconds: float, device, capture_rate: int
                     ) -> np.ndarray:
    input(f'\n  Say: "{prompt}"  -- press Enter, then speak... ')
    print('  Recording', end='', flush=True)
    audio = record_seconds(seconds, device=device, capture_rate=capture_rate)
    print(' done.')
    return audio


def _summarize(scores: List[float], backend: str) -> None:
    if not scores:
        return
    arr = np.array(scores)
    print('\nSelf-similarity of enrollment samples (leave-one-out):')
    print(f'  min {arr.min():.3f}   mean {arr.mean():.3f}   '
          f'max {arr.max():.3f}')
    default = DEFAULT_THRESHOLDS[backend]
    if arr.min() < default:
        print(f'  WARNING: some samples score below the default threshold '
              f'({default:.2f}). Re-record in a quieter room, keep the same '
              f'distance from the microphone, or lower speaker_threshold.')
    print('  Run "verify_speaker" to check how you and other people score '
          'before choosing a threshold.')


def enroll_main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog='enroll_speaker',
        description='Record voice samples and build the speaker profile the '
                    'voice command node uses to recognize its owner.')
    _add_audio_args(parser)
    _add_model_args(parser)
    parser.add_argument('--samples', type=int, default=12,
                        help='Number of recordings to make (default: 12).')
    parser.add_argument('--seconds', type=float, default=3.0,
                        help='Length of each recording (default: 3.0).')
    parser.add_argument('--from-dir', default='',
                        help='Enroll from existing 16-bit PCM WAV files in '
                             'this directory instead of recording.')
    parser.add_argument('--save-audio', default='',
                        help='Also save each recording as a WAV file here.')
    parser.add_argument('--append', action='store_true',
                        help='Add samples to an existing profile instead of '
                             'replacing it.')
    parser.add_argument('--yes', action='store_true',
                        help='Overwrite an existing profile without asking.')
    args = parser.parse_args(argv)

    profile_path = os.path.expanduser(args.profile)
    existing = None
    if os.path.isfile(profile_path):
        if args.append:
            existing = SpeakerProfile.load(profile_path)
            if existing.backend != args.backend:
                print(f'Existing profile uses {existing.backend!r}; cannot '
                      f'append {args.backend!r} samples.', file=sys.stderr)
                return 1
        elif not args.yes:
            answer = input(f'{profile_path} exists. Overwrite? [y/N] ')
            if answer.strip().lower() != 'y':
                print('Aborted.')
                return 1

    print(f'Loading {args.backend} speaker model (first run downloads it)...')
    backend = load_backend(args.backend, device=args.compute_device)

    embeddings = list(existing.embeddings) if existing is not None else []

    if args.from_dir:
        files = sorted(glob.glob(os.path.join(args.from_dir, '*.wav')))
        if not files:
            print(f'No .wav files found in {args.from_dir}', file=sys.stderr)
            return 1
        for path in files:
            audio = trim_silence(read_wav(path))
            embeddings.append(backend.embed(audio))
            print(f'  {os.path.basename(path)}: {len(audio) / 16000:.2f}s')
    else:
        device = parse_device(args.device)
        if args.save_audio:
            os.makedirs(args.save_audio, exist_ok=True)
        print(
            '\nYou will record {n} short phrases. Speak at a normal volume '
            'from where you would usually\ngive commands. Vary your tone '
            'slightly between takes. Only the owner should speak.'
            .format(n=args.samples))
        i = 0
        while i < args.samples:
            prompt = DEFAULT_PROMPTS[i % len(DEFAULT_PROMPTS)]
            print(f'\n[{i + 1}/{args.samples}]', end='')
            audio = _record_prompted(prompt, args.seconds, device,
                                     args.capture_rate)
            level = rms_dbfs(audio)
            if level < MIN_LEVEL_DBFS:
                print(f'  Too quiet ({level:.1f} dBFS); check the microphone '
                      f'and try again.')
                continue
            trimmed = trim_silence(audio)
            if args.save_audio:
                write_wav(os.path.join(
                    args.save_audio,
                    f'enroll_{int(time.time())}_{i:02d}.wav'), trimmed)
            embeddings.append(backend.embed(trimmed))
            i += 1

    profile = SpeakerProfile.from_embeddings(backend.name, embeddings)
    profile.save(profile_path)
    print(f'\nSaved profile with {profile.num_samples} samples to '
          f'{profile_path}')
    _summarize(profile.leave_one_out_scores(), backend.name)

    settings = {
        'speaker_backend': backend.name,
        'speaker_profile_path': profile_path,
    }
    if not args.from_dir:
        # Save the microphone used for enrollment so launches use it too.
        settings['audio_device'] = args.device
        settings['capture_sample_rate'] = args.capture_rate
    update_user_params(settings)
    return 0


def verify_main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog='verify_speaker',
        description='Score live recordings or WAV files against the speaker '
                    'profile to choose a threshold.')
    _add_audio_args(parser)
    _add_model_args(parser)
    parser.add_argument('--threshold', type=float, default=-1.0,
                        help='Threshold to report pass/fail against '
                             '(default: backend default).')
    parser.add_argument('--seconds', type=float, default=3.0)
    parser.add_argument('--save-threshold', type=float, default=None,
                        metavar='VALUE',
                        help='Save VALUE as speaker_threshold in the '
                             'per-machine settings file and exit.')
    parser.add_argument('wav', nargs='*',
                        help='WAV files to score. Omit to record live.')
    args = parser.parse_args(argv)

    if args.save_threshold is not None:
        update_user_params({'speaker_threshold': float(args.save_threshold)})
        return 0

    threshold = (args.threshold if args.threshold >= 0
                 else DEFAULT_THRESHOLDS[args.backend])
    profile = SpeakerProfile.load(args.profile)
    backend = load_backend(args.backend, device=args.compute_device)
    verifier = SpeakerVerifier(profile, backend, threshold)

    def report(label: str, audio: np.ndarray) -> None:
        result = verifier.verify(trim_silence(audio))
        verdict = 'ACCEPT' if result.accepted else 'REJECT'
        print(f'  {label}: score {result.score:.3f} -> {verdict}'
              f'{"" if result.accepted else " (" + result.reason + ")"}')

    print(f'Threshold: {threshold:.3f}')
    if args.wav:
        for path in args.wav:
            report(os.path.basename(path), read_wav(path))
        return 0

    device = parse_device(args.device)
    print('Press Ctrl+C to stop. Try your own voice and other people\'s.')
    try:
        n = 1
        while True:
            audio = _record_prompted('Hey Yeller, sit', args.seconds, device,
                                     args.capture_rate)
            report(f'take {n}', audio)
            n += 1
    except (KeyboardInterrupt, EOFError):
        print()
    return 0


def list_devices_main(argv=None) -> int:
    import sounddevice as sd

    print('Input devices (use the index or part of the name as '
          'audio_device):\n')
    default_in = sd.default.device[0]
    for index, dev in enumerate(sd.query_devices()):
        if dev['max_input_channels'] < 1:
            continue
        marker = '*' if index == default_in else ' '
        print(f' {marker} {index:3d}  {dev["name"]}  '
              f'({dev["max_input_channels"]} ch, '
              f'{int(dev["default_samplerate"])} Hz default)')
    print('\n * = system default')
    return 0


def enroll() -> None:
    sys.exit(enroll_main())


def verify() -> None:
    sys.exit(verify_main())


def list_devices() -> None:
    sys.exit(list_devices_main())
