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
