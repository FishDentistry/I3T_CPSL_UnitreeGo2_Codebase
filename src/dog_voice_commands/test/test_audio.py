import numpy as np
import pytest

from dog_voice_commands.audio import (RingBuffer, read_wav, resample,
                                      rms_dbfs, trim_silence, write_wav)


def test_ring_buffer_extract_absolute():
    buf = RingBuffer(seconds=1.0, sample_rate=10)  # capacity 10
    buf.append(np.arange(0, 7, dtype=np.int16))
    buf.append(np.arange(7, 15, dtype=np.int16))
    assert buf.total_samples == 15
    # Oldest retained sample is 5.
    np.testing.assert_array_equal(buf.extract(0, 15), np.arange(5, 15))
    np.testing.assert_array_equal(buf.extract(8, 12), np.arange(8, 12))
    assert buf.extract(20, 30).size == 0


def test_ring_buffer_oversized_chunk():
    buf = RingBuffer(seconds=1.0, sample_rate=4)
    buf.append(np.arange(10, dtype=np.int16))
    np.testing.assert_array_equal(buf.extract(0, 10), np.arange(6, 10))


def test_resample_length():
    audio = np.zeros(48000, dtype=np.int16)
    out = resample(audio, 48000, 16000)
    assert out.dtype == np.int16
    assert len(out) == 16000


def test_trim_silence():
    rate = 16000
    t = np.arange(rate // 2) / rate
    tone = (0.5 * 32767 * np.sin(2 * np.pi * 220 * t)).astype(np.int16)
    silence = np.zeros(rate, dtype=np.int16)
    audio = np.concatenate([silence, tone, silence])
    trimmed = trim_silence(audio, rate, pad_ms=0)
    assert abs(len(trimmed) - len(tone)) <= rate * 0.04


def test_rms_dbfs():
    assert rms_dbfs(np.zeros(100, dtype=np.int16)) < -150
    full = np.full(100, 32767, dtype=np.int16)
    assert abs(rms_dbfs(full)) < 0.01


def test_wav_round_trip(tmp_path):
    audio = (np.random.default_rng(0).standard_normal(8000) * 1000
             ).astype(np.int16)
    path = str(tmp_path / 'a.wav')
    write_wav(path, audio)
    np.testing.assert_array_equal(read_wav(path), audio)


class _FakeSd:
    """Mimics sounddevice for a mic that only accepts some rates."""

    def __init__(self, accepted, default_rate):
        self.accepted = set(accepted)
        self.default_rate = default_rate
        self.tried = []

    def query_devices(self, device, kind):
        return {'default_samplerate': float(self.default_rate)}

    def check_input_settings(self, device, channels, dtype, samplerate):
        self.tried.append(samplerate)
        if samplerate not in self.accepted:
            raise RuntimeError('Invalid sample rate [PaErrorCode -9997]')


def test_capture_rate_keeps_supported_request():
    from dog_voice_commands.audio import supported_capture_rate
    sd = _FakeSd(accepted={16000, 48000}, default_rate=48000)
    assert supported_capture_rate('usb', 16000, sd) == 16000


def test_capture_rate_falls_back_to_device_default():
    from dog_voice_commands.audio import supported_capture_rate
    sd = _FakeSd(accepted={44100}, default_rate=44100)
    assert supported_capture_rate('usb', 16000, sd) == 44100
    assert sd.tried == [16000, 44100]


def test_capture_rate_raises_when_nothing_works():
    import pytest
    from dog_voice_commands.audio import supported_capture_rate
    sd = _FakeSd(accepted=set(), default_rate=22050)
    with pytest.raises(RuntimeError):
        supported_capture_rate('usb', 16000, sd)


def test_resample_44100_to_16000():
    from dog_voice_commands.audio import resample
    out = resample(np.zeros(44100, dtype=np.int16), 44100, 16000)
    assert len(out) == 16000


def _tone(rate, seconds=1.0, freq=440.0):
    t = np.arange(int(rate * seconds)) / rate
    return (0.5 * 32767 * np.sin(2 * np.pi * freq * t)).astype(np.int16)


@pytest.mark.parametrize('rate', [48000, 44100])
def test_stream_resampler_chunking_invariant(rate):
    from dog_voice_commands.audio import StreamResampler
    audio = _tone(rate)
    whole = StreamResampler(rate).process(audio)
    stream = StreamResampler(rate)
    block = rate // 10  # 100 ms, as the microphone delivers it
    chunked = np.concatenate([stream.process(audio[i:i + block])
                              for i in range(0, len(audio), block)])
    assert len(chunked) == len(whole)
    assert np.max(np.abs(chunked.astype(int) - whole.astype(int))) <= 1
    assert abs(len(whole) - 16000) <= 1


@pytest.mark.parametrize('rate', [48000, 44100])
def test_stream_resampler_preserves_tone(rate):
    from dog_voice_commands.audio import StreamResampler
    out = StreamResampler(rate).process(_tone(rate, freq=1000.0))
    spectrum = np.abs(np.fft.rfft(out[1600:].astype(float)))
    peak_hz = np.argmax(spectrum) * 16000 / len(out[1600:])
    assert abs(peak_hz - 1000.0) < 5.0


def test_stream_resampler_passthrough():
    from dog_voice_commands.audio import StreamResampler
    audio = _tone(16000, 0.1)
    np.testing.assert_array_equal(StreamResampler(16000).process(audio), audio)


def test_microphone_callback_never_raises():
    from dog_voice_commands.audio import MicrophoneStream

    class BadBuffer:
        def __bytes__(self):
            raise ValueError('bad buffer')

    mic = MicrophoneStream(capture_rate=16000)
    mic._callback(BadBuffer(), 10, None, None)  # Must not raise.
    assert mic.callback_error == 'ValueError: bad buffer'


def test_microphone_callback_queues_audio():
    from dog_voice_commands.audio import MicrophoneStream

    mic = MicrophoneStream(capture_rate=16000)
    block = (np.arange(160) * 10).astype(np.int16)
    mic._callback(memoryview(block.tobytes()), 160, None, None)
    assert mic.samples_received == 160
    np.testing.assert_array_equal(mic.read(timeout=0.1), block)
    assert mic.read(timeout=0.01) is None


def test_microphone_defaults_to_portaudio_blocksize():
    from dog_voice_commands.audio import MicrophoneStream
    assert MicrophoneStream().blocksize == 0
