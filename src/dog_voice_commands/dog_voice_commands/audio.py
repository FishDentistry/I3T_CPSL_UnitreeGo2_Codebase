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


class MicrophoneStream:
    """Microphone capture that yields 16 kHz ``int16`` chunks.

    Audio is read by PortAudio (via ``sounddevice``) on its own thread and
    passed through a bounded queue so a slow consumer drops old audio rather
    than growing memory without bound.
    """

    def __init__(self, device: DeviceSpec = None,
                 capture_rate: int = TARGET_SAMPLE_RATE,
                 chunk_ms: int = 100, max_queue_chunks: int = 100) -> None:
        self.device = device
        self.capture_rate = int(capture_rate)
        self.blocksize = int(self.capture_rate * chunk_ms / 1000)
        self._queue: 'queue.Queue[bytes]' = queue.Queue(max_queue_chunks)
        self._stream = None
        self.overflows = 0
        self.dropped_chunks = 0

    def _callback(self, indata, frames, time_info, status) -> None:
        if status and status.input_overflow:
            self.overflows += 1
        try:
            self._queue.put_nowait(bytes(indata))
        except queue.Full:
            self.dropped_chunks += 1
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(bytes(indata))
            except (queue.Empty, queue.Full):
                pass

    def start(self) -> None:
        import sounddevice as sd

        self._stream = sd.RawInputStream(
            samplerate=self.capture_rate, blocksize=self.blocksize,
            device=self.device, channels=1, dtype='int16',
            callback=self._callback)
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def read(self, timeout: float = 0.5) -> Optional[np.ndarray]:
        """Return the next 16 kHz chunk, or ``None`` on timeout."""
        try:
            raw = self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
        audio = np.frombuffer(raw, dtype=np.int16)
        return resample(audio, self.capture_rate)

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
