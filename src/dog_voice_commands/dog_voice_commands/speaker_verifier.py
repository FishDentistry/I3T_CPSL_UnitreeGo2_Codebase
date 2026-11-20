"""Speaker verification against an enrolled voice profile.

A speaker-embedding model maps an utterance to a fixed-length vector. During
enrollment the owner records several phrases; their embeddings are averaged
into a *centroid*. At run time an utterance is accepted only when the cosine
similarity between its embedding and the centroid meets a threshold.

Two embedding backends are supported:

* ``speechbrain`` (default) - ECAPA-TDNN trained on VoxCeleb
  (``speechbrain/spkrec-ecapa-voxceleb``). More accurate; needs PyTorch and
  torchaudio.
* ``resemblyzer`` - GE2E encoder. Lighter and simpler to install, but less
  discriminative, so it needs a higher threshold.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .audio import TARGET_SAMPLE_RATE, int16_to_float

DEFAULT_DATA_DIR = os.path.expanduser('~/.local/share/dog_voice_commands')
DEFAULT_PROFILE_PATH = os.path.join(DEFAULT_DATA_DIR, 'speaker_profile.npz')

#: Reasonable starting thresholds per backend. Tune with ``verify_speaker``.
DEFAULT_THRESHOLDS = {
    'speechbrain': 0.45,
    'resemblyzer': 0.75,
}

#: Utterances shorter than this are rejected without scoring; embeddings
#: from very short clips are unreliable.
MIN_VERIFY_SECONDS = 0.5


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector if norm == 0.0 else vector / norm


class EmbeddingBackend:
    """Interface for speaker-embedding models."""

    name = 'base'

    def embed(self, audio: np.ndarray) -> np.ndarray:
        """Return an embedding for 16 kHz ``int16`` audio."""
        raise NotImplementedError


class SpeechBrainBackend(EmbeddingBackend):
    name = 'speechbrain'
    SOURCE = 'speechbrain/spkrec-ecapa-voxceleb'

    def __init__(self, model_dir: Optional[str] = None,
                 device: str = 'cpu') -> None:
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:  # SpeechBrain < 1.0
            from speechbrain.pretrained import EncoderClassifier
        import torch

        self._torch = torch
        model_dir = model_dir or os.path.join(
            DEFAULT_DATA_DIR, 'models', 'spkrec-ecapa-voxceleb')
        self._model = EncoderClassifier.from_hparams(
            source=self.SOURCE, savedir=model_dir,
            run_opts={'device': device})
        self._model.eval()

    def embed(self, audio: np.ndarray) -> np.ndarray:
        wav = self._torch.from_numpy(int16_to_float(audio)).unsqueeze(0)
        with self._torch.no_grad():
            emb = self._model.encode_batch(wav)
        return emb.squeeze().cpu().numpy().astype(np.float32)


class ResemblyzerBackend(EmbeddingBackend):
    name = 'resemblyzer'

    def __init__(self, device: str = 'cpu') -> None:
        from resemblyzer import VoiceEncoder

        self._encoder = VoiceEncoder(device=device, verbose=False)

    def embed(self, audio: np.ndarray) -> np.ndarray:
        from resemblyzer import preprocess_wav

        wav = preprocess_wav(int16_to_float(audio),
                             source_sr=TARGET_SAMPLE_RATE)
        if wav.size == 0:
            raise ValueError('no speech found in audio')
        return self._encoder.embed_utterance(wav).astype(np.float32)


def load_backend(name: str, device: str = 'cpu',
                 model_dir: Optional[str] = None) -> EmbeddingBackend:
    name = name.strip().lower()
    if name == 'speechbrain':
        return SpeechBrainBackend(model_dir=model_dir, device=device)
    if name == 'resemblyzer':
        return ResemblyzerBackend(device=device)
    raise ValueError(f'unknown speaker backend {name!r} '
                     f'(expected "speechbrain" or "resemblyzer")')


@dataclass
class SpeakerProfile:
    """Enrolled embeddings for one speaker."""

    backend: str
    embeddings: np.ndarray  # Shape (N, D).

    @property
    def centroid(self) -> np.ndarray:
        normed = np.stack([l2_normalize(e) for e in self.embeddings])
        return l2_normalize(normed.mean(axis=0))

    @property
    def num_samples(self) -> int:
        return int(self.embeddings.shape[0])

    def score(self, embedding: np.ndarray) -> float:
        return cosine_similarity(embedding, self.centroid)

    def leave_one_out_scores(self) -> List[float]:
        """Score each enrollment sample against the others' centroid.

        A rough estimate of how the owner's own voice will score at run time.
        """
        if self.num_samples < 2:
            return []
        scores = []
        for i in range(self.num_samples):
            rest = np.delete(self.embeddings, i, axis=0)
            centroid = l2_normalize(
                np.stack([l2_normalize(e) for e in rest]).mean(axis=0))
            scores.append(cosine_similarity(self.embeddings[i], centroid))
        return scores

    def save(self, path: str) -> None:
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        np.savez(path, backend=np.array(self.backend),
                 embeddings=self.embeddings.astype(np.float32),
                 created_unix=np.array(time.time()))
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass

    @classmethod
    def load(cls, path: str) -> 'SpeakerProfile':
        with np.load(os.path.expanduser(path), allow_pickle=False) as data:
            return cls(backend=str(data['backend']),
                       embeddings=np.array(data['embeddings']))

    @classmethod
    def from_embeddings(cls, backend: str,
                        embeddings: Sequence[np.ndarray]) -> 'SpeakerProfile':
        if not embeddings:
            raise ValueError('at least one embedding is required')
        return cls(backend=backend, embeddings=np.stack(embeddings))


@dataclass
class VerificationResult:
    accepted: bool
    score: float
    reason: str = ''


class SpeakerVerifier:
    """Accept or reject utterances against an enrolled profile."""

    def __init__(self, profile: SpeakerProfile, backend: EmbeddingBackend,
                 threshold: float) -> None:
        if profile.backend != backend.name:
            raise ValueError(
                f'speaker profile was enrolled with {profile.backend!r} but '
                f'the {backend.name!r} backend is loaded; re-enroll or change '
                f'speaker_backend')
        self.profile = profile
        self.backend = backend
        self.threshold = float(threshold)

    def verify(self, audio: np.ndarray) -> VerificationResult:
        seconds = len(audio) / TARGET_SAMPLE_RATE
        if seconds < MIN_VERIFY_SECONDS:
            return VerificationResult(False, 0.0,
                                      f'utterance too short ({seconds:.2f}s)')
        try:
            embedding = self.backend.embed(audio)
        except ValueError as exc:
            return VerificationResult(False, 0.0, str(exc))
        score = self.profile.score(embedding)
        if score >= self.threshold:
            return VerificationResult(True, score)
        return VerificationResult(
            False, score,
            f'score {score:.3f} below threshold {self.threshold:.3f}')
