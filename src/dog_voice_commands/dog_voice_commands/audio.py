"""Audio capture, buffering, and file helpers.

All audio handed to the recognizer and speaker verifier is mono, 16 kHz,
signed 16-bit PCM (``int16``). Microphones that cannot capture at 16 kHz are
captured at their native rate and resampled.
"""

from __future__ import annotations

import queue
import threading
import wave
from math import gcd
from typing import Optional, Union

import numpy as np

TARGET_SAMPLE_RATE = 16000


def resample(audio: np.ndarray, source_rate: int,
             target_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Resample ``int16`` or ``float`` audio with a polyphase filter."""
    if source_rate == target_rate:
        return audio
    from scipy.signal import resample_poly

    divisor = gcd(int(source_rate), int(target_rate))
    up = target_rate // divisor
    down = source_rate // divisor
    out = resample_poly(audio.astype(np.float32), up, down)
    if audio.dtype == np.int16:
        out = np.clip(np.round(out), -32768, 32767).astype(np.int16)
    return out


class StreamResampler:
    """Resample a continuous ``int16`` stream chunk by chunk.

    Resampling each chunk independently (as :func:`resample` would) treats
    every chunk edge as silence, which adds a click at each chunk boundary
    and degrades speech recognition. This resampler keeps its filter state
    and fractional read position between chunks, so the output is the same
    whether the stream arrives in one piece or many.

    An 8th-order Butterworth low-pass at 90% of the target Nyquist frequency
    removes content that would alias, then samples are linearly
    interpolated at the target rate.
    """

    def __init__(self, source_rate: int,
                 target_rate: int = TARGET_SAMPLE_RATE) -> None:
        self.source_rate = int(source_rate)
        self.target_rate = int(target_rate)
        self.passthrough = self.source_rate == self.target_rate
        if self.passthrough:
            return
        from scipy.signal import butter

        if self.target_rate > self.source_rate:
            raise ValueError('StreamResampler only downsamples')
        cutoff = 0.9 * (self.target_rate / 2.0) / (self.source_rate / 2.0)
        self._sos = butter(8, cutoff, output='sos')
        self._zi = np.zeros((self._sos.shape[0], 2))
        self._step = self.source_rate / self.target_rate
        self._pos = 0.0          # Next output position in the buffer below.
        self._prev = None        # Last filtered sample of the previous chunk.

    def process(self, chunk: np.ndarray) -> np.ndarray:
        if self.passthrough:
            return np.asarray(chunk, dtype=np.int16)
        from scipy.signal import sosfilt

        filtered, self._zi = sosfilt(
            self._sos, np.asarray(chunk, dtype=np.float64), zi=self._zi)
        if self._prev is not None:
            # Index 0 is the last sample of the previous chunk, so
            # interpolation is continuous across the boundary.
            filtered = np.concatenate(([self._prev], filtered))
        last = len(filtered) - 1
        if last < 0:
            return np.zeros(0, dtype=np.int16)
        positions = np.arange(self._pos, last + 1e-9, self._step)
        out = np.interp(positions, np.arange(len(filtered)), filtered)
        self._pos = (positions[-1] + self._step - last) if positions.size \
            else self._pos - last
        self._prev = filtered[-1]
        return np.clip(np.round(out), -32768, 32767).astype(np.int16)


def int16_to_float(audio: np.ndarray) -> np.ndarray:
    """Convert ``int16`` PCM to ``float32`` in ``[-1, 1)``."""
    return audio.astype(np.float32) / 32768.0


def rms_dbfs(audio: np.ndarray) -> float:
    """Return the RMS level of ``int16`` audio in dBFS."""
    if audio.size == 0:
        return float('-inf')
    rms = np.sqrt(np.mean(int16_to_float(audio) ** 2))
    return float(20.0 * np.log10(max(rms, 1e-10)))


def trim_silence(audio: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE,
                 frame_ms: int = 20, threshold_db: float = 30.0,
                 pad_ms: int = 150) -> np.ndarray:
    """Trim leading and trailing frames quieter than the loudest frame.

    Frames more than ``threshold_db`` below the loudest frame are treated
    as silence. ``pad_ms`` of audio is kept on each side of the speech.
    """
    frame = max(1, int(sample_rate * frame_ms / 1000))
    n_frames = len(audio) // frame
    if n_frames == 0:
        return audio
    frames = int16_to_float(audio[:n_frames * frame]).reshape(n_frames, frame)
    energy_db = 10.0 * np.log10(np.mean(frames ** 2, axis=1) + 1e-12)
    voiced = np.flatnonzero(energy_db >= energy_db.max() - threshold_db)
    if voiced.size == 0:
        return audio
    pad = int(sample_rate * pad_ms / 1000)
    start = max(0, voiced[0] * frame - pad)
    end = min(len(audio), (voiced[-1] + 1) * frame + pad)
    return audio[start:end]


def read_wav(path: str) -> np.ndarray:
    """Read a PCM WAV file as mono 16 kHz ``int16``."""
    with wave.open(path, 'rb') as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())
    if width != 2:
        raise ValueError(f'{path}: only 16-bit PCM WAV files are supported '
                         f'(got {8 * width}-bit)')
    audio = np.frombuffer(raw, dtype=np.int16)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1).astype(np.int16)
    return resample(audio, rate)


def write_wav(path: str, audio: np.ndarray,
              sample_rate: int = TARGET_SAMPLE_RATE) -> None:
    """Write mono ``int16`` audio to a WAV file."""
    with wave.open(path, 'wb') as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(np.asarray(audio, dtype=np.int16).tobytes())


class RingBuffer:
    """Fixed-length history of audio indexed by absolute sample number.

    Lets callers extract the audio for a word span reported by the
    recognizer (whose timestamps are relative to the start of the stream).
    """

    def __init__(self, seconds: float,
                 sample_rate: int = TARGET_SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate
        self.capacity = int(seconds * sample_rate)
        self._data = np.zeros(self.capacity, dtype=np.int16)
        self._total = 0  # Samples written since creation.
        self._lock = threading.Lock()

    @property
    def total_samples(self) -> int:
        return self._total

    def append(self, chunk: np.ndarray) -> None:
        chunk = np.asarray(chunk, dtype=np.int16)
        with self._lock:
            if chunk.size >= self.capacity:
                end_total = self._total + chunk.size
                idx = np.arange(end_total - self.capacity,
                                end_total) % self.capacity
                self._data[idx] = chunk[-self.capacity:]
                self._total = end_total
                return
            start = self._total % self.capacity
            end = start + chunk.size
            if end <= self.capacity:
                self._data[start:end] = chunk
            else:
                split = self.capacity - start
                self._data[start:] = chunk[:split]
                self._data[:end - self.capacity] = chunk[split:]
            self._total += chunk.size

    def extract(self, start_sample: int, end_sample: int) -> np.ndarray:
        """Return samples ``[start_sample, end_sample)``, clipped to history."""
        with self._lock:
            oldest = max(0, self._total - self.capacity)
            start = max(int(start_sample), oldest)
            end = min(int(end_sample), self._total)
            if end <= start:
                return np.zeros(0, dtype=np.int16)
            idx = np.arange(start, end) % self.capacity
            return self._data[idx].copy()

    def extract_seconds(self, start_sec: float, end_sec: float) -> np.ndarray:
        return self.extract(int(start_sec * self.sample_rate),
                            int(end_sec * self.sample_rate))


DeviceSpec = Optional[Union[int, str]]


def parse_device(device: str) -> DeviceSpec:
    """Interpret a device parameter: empty -> default, digits -> index."""
    device = (device or '').strip()
    if not device:
        return None
    if device.isdigit():
        return int(device)
    return device


def supported_capture_rate(device: DeviceSpec = None,
                           requested: int = TARGET_SAMPLE_RATE,
                           sd=None) -> int:
    """Return a capture rate the input device accepts.

    Tries ``requested`` first, then the device's default rate, then 48000 and
    44100 Hz. Many USB microphones opened directly through ALSA only accept
    their native rate (usually 44100 or 48000 Hz), not 16000 Hz; audio
    captured at another rate is resampled to 16 kHz by this module.
    """
    if sd is None:
        import sounddevice as sd

    candidates = [int(requested)]
    try:
        info = sd.query_devices(device, 'input')
        candidates.append(int(info['default_samplerate']))
    except Exception:
        pass
    candidates += [48000, 44100]

    last_error = None
    for rate in dict.fromkeys(candidates):
        try:
            sd.check_input_settings(device=device, channels=1,
                                    dtype='int16', samplerate=rate)
            return rate
        except Exception as exc:  # PortAudioError, ValueError
            last_error = exc
    raise last_error


class MicrophoneStream:
    """Microphone capture that yields 16 kHz ``int16`` chunks.

    Audio is read by PortAudio (via ``sounddevice``) on its own thread and
    passed through a bounded queue so a slow consumer drops old audio rather
    than growing memory without bound.

    By default PortAudio chooses the buffer size (``blocksize=0``), as
    ``sounddevice.rec`` does. Forcing a fixed block size on a raw ALSA
    hardware device can make some USB microphones stop delivering audio
    shortly after the stream starts.
    """

    def __init__(self, device: DeviceSpec = None,
                 capture_rate: int = TARGET_SAMPLE_RATE,
                 blocksize: int = 0, max_queue_chunks: int = 200) -> None:
        self.device = device
        self.capture_rate = int(capture_rate)
        self.requested_rate = self.capture_rate
        self.blocksize = int(blocksize)
        self.device_name = ''
        self._resampler = StreamResampler(self.capture_rate)
        self._queue: 'queue.Queue[bytes]' = queue.Queue(max_queue_chunks)
        self._stream = None
        self.overflows = 0
        self.dropped_chunks = 0
        self.samples_received = 0     # At the capture rate.
        self.restarts = 0
        #: Last exception raised inside the audio callback, if any. The
        #: callback never lets an exception escape, because sounddevice
        #: aborts the stream permanently when one does.
        self.callback_error: Optional[str] = None

    def _callback(self, indata, frames, time_info, status) -> None:
        try:
            if status and status.input_overflow:
                self.overflows += 1
            data = bytes(indata)
            self.samples_received += frames
            try:
                self._queue.put_nowait(data)
            except queue.Full:
                self.dropped_chunks += 1
                try:
                    self._queue.get_nowait()
                    self._queue.put_nowait(data)
                except (queue.Empty, queue.Full):
                    pass
        except Exception as exc:  # Never abort the stream.
            self.callback_error = f'{type(exc).__name__}: {exc}'

    def _open(self, sd) -> None:
        self._stream = sd.RawInputStream(
            samplerate=self.capture_rate, blocksize=self.blocksize,
            device=self.device, channels=1, dtype='int16',
            callback=self._callback)
        try:
            self.device_name = sd.query_devices(
                self._stream.device, 'input')['name']
        except Exception:
            self.device_name = str(self._stream.device)
        self._stream.start()

    def start(self) -> None:
        import sounddevice as sd

        rate = supported_capture_rate(self.device, self.capture_rate, sd)
        if rate != self.capture_rate:
            self.requested_rate = self.capture_rate
            self.capture_rate = rate
        self._resampler = StreamResampler(self.capture_rate)
        self._open(sd)

    def state(self) -> str:
        """Describe the PortAudio stream state, for diagnostics."""
        if self._stream is None:
            return 'not open'
        try:
            return ('active' if self._stream.active else
                    'stopped' if self._stream.stopped else 'inactive')
        except Exception as exc:
            return f'unknown ({exc})'

    def restart(self) -> None:
        """Close and reopen the stream, keeping the same settings."""
        import sounddevice as sd

        self._close()
        self.flush()
        self._resampler = StreamResampler(self.capture_rate)
        self.restarts += 1
        self._open(sd)

    def _close(self) -> None:
        if self._stream is not None:
            stream, self._stream = self._stream, None
            for action in (stream.abort, stream.close):
                try:
                    action()
                except Exception:
                    pass

    def stop(self) -> None:
        self._close()

    def read(self, timeout: float = 0.5) -> Optional[np.ndarray]:
        """Return the next 16 kHz chunk, or ``None`` on timeout."""
        try:
            raw = self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
        audio = np.frombuffer(raw, dtype=np.int16)
        return self._resampler.process(audio)

    def flush(self) -> None:
        """Discard any queued audio."""
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def __enter__(self) -> 'MicrophoneStream':
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


def record_seconds(seconds: float, device: DeviceSpec = None,
                   capture_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Record a fixed-length clip and return 16 kHz ``int16`` audio."""
    import sounddevice as sd

    frames = int(seconds * capture_rate)
    audio = sd.rec(frames, samplerate=capture_rate, channels=1,
                   dtype='int16', device=device)
    sd.wait()
    return resample(audio[:, 0], capture_rate)
