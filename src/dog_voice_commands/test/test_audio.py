import numpy as np

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
