import numpy as np
import pytest

from dog_voice_commands.speaker_verifier import (EmbeddingBackend,
                                                 SpeakerProfile,
                                                 SpeakerVerifier,
                                                 cosine_similarity)


class FakeBackend(EmbeddingBackend):
    """Embeds audio as a fixed vector chosen by its first sample."""

    name = 'fake'

    def __init__(self, vectors):
        self.vectors = vectors

    def embed(self, audio):
        return self.vectors[int(audio[0])]


def test_cosine_similarity():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([0, 0], [1, 0]) == 0.0


def test_profile_save_load(tmp_path):
    emb = np.random.default_rng(1).standard_normal((5, 192)).astype(
        np.float32)
    profile = SpeakerProfile('speechbrain', emb)
    path = str(tmp_path / 'p.npz')
    profile.save(path)
    loaded = SpeakerProfile.load(path)
    assert loaded.backend == 'speechbrain'
    np.testing.assert_allclose(loaded.embeddings, emb)
    assert len(loaded.leave_one_out_scores()) == 5


def test_verifier_accepts_owner_rejects_other():
    owner = np.array([1.0, 0.1, 0.0])
    other = np.array([0.0, 0.2, 1.0])
    backend = FakeBackend({0: owner, 1: other})
    profile = SpeakerProfile.from_embeddings(
        'fake', [owner, owner * 1.1 + 0.01])
    verifier = SpeakerVerifier(profile, backend, threshold=0.8)

    owner_audio = np.zeros(16000, dtype=np.int16)
    other_audio = np.ones(16000, dtype=np.int16)
    assert verifier.verify(owner_audio).accepted
    result = verifier.verify(other_audio)
    assert not result.accepted
    assert 'below threshold' in result.reason


def test_verifier_rejects_short_audio():
    backend = FakeBackend({0: np.ones(3)})
    profile = SpeakerProfile.from_embeddings('fake', [np.ones(3)])
    verifier = SpeakerVerifier(profile, backend, threshold=0.5)
    result = verifier.verify(np.zeros(1000, dtype=np.int16))
    assert not result.accepted
    assert 'too short' in result.reason


def test_backend_mismatch():
    backend = FakeBackend({})
    profile = SpeakerProfile.from_embeddings('speechbrain', [np.ones(3)])
    with pytest.raises(ValueError):
        SpeakerVerifier(profile, backend, threshold=0.5)


def test_update_user_params(tmp_path):
    import yaml
    from dog_voice_commands.tools import update_user_params

    path = str(tmp_path / 'sub' / 'params.yaml')
    update_user_params({'audio_device': 'USB'}, path)
    update_user_params({'speaker_threshold': 0.5}, path)
    with open(path) as handle:
        params = yaml.safe_load(handle)['voice_command_node'][
            'ros__parameters']
    assert params == {'audio_device': 'USB', 'speaker_threshold': 0.5}
