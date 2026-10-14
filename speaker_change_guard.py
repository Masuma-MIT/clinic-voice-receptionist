# SPEAKER_STRONG_TURN_ROUTING_V1_20261010
# SPEAKER_HANDOVER_REFERENCE_V2_20261010
# SPEAKER_REFERENCE_QUALITY_V1_20261004
# WHOLE_CALL_SPEAKER_EVIDENCE_V1_20261003
"""Conservative, in-session speaker-change detection for live PCM16 audio.

The guard treats a voice embedding as a safety signal, never as proof of a
person's identity. A new speaker is reported only after two good-quality,
mutually consistent turns are both far from every enrolled session speaker.
"""

from __future__ import annotations

import math
import os
from array import array
from threading import Lock

import numpy as np
import torch
import torchaudio

if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: []

from speechbrain.inference.speaker import SpeakerRecognition
from speechbrain.utils.fetching import LocalStrategy


class SpeakerChangeGuard:
    """Track anonymous speakers during one call without permanent voice IDs."""

    _model = None
    _model_lock = Lock()
    SAMPLE_RATE = 16000
    MINIMUM_SECONDS = 1.50
    SINGLE_TURN_CONFIRMATION_SECONDS = 3.00
    MINIMUM_RMS = 220.0
    SAME_SPEAKER_THRESHOLD = float(
        os.getenv("SPEAKER_SAME_THRESHOLD", "0.62")
    )
    DIFFERENT_SPEAKER_THRESHOLD = float(
        os.getenv("SPEAKER_DIFFERENT_THRESHOLD", "0.42")
    )
    CANDIDATE_MATCH_THRESHOLD = float(
        os.getenv("SPEAKER_CANDIDATE_THRESHOLD", "0.62")
    )
    REQUIRED_DIFFERENT_TURNS = 2
    MAX_SESSION_SPEAKERS = 4
    MAX_REFERENCE_SECONDS = 12.0
    MAX_EMBEDDING_SECONDS = 8.0

    def __init__(self, embedding_provider=None):
        self.embedding_provider = embedding_provider
        self.speakers: dict[str, torch.Tensor] = {}
        self.active_speaker_id: str | None = None
        self.pending_embedding: torch.Tensor | None = None
        self.pending_count = 0
        self.turn_number = 0
        self.last_embedding: torch.Tensor | None = None
        # Short anonymous reference audio for optional multi-model confirmation.
        # It is scoped to this call and is never treated as identity proof.
        self.speaker_audio: dict[str, bytes] = {}

    def _snapshot_before_evidence_repair(self) -> dict:
        """Capture mutable call state so a provisional change can be rolled back."""
        return {
            "speakers": {key: value.clone() for key, value in self.speakers.items()},
            "speaker_audio": dict(self.speaker_audio),
            "active_speaker_id": self.active_speaker_id,
            "pending_embedding": (
                None if self.pending_embedding is None else self.pending_embedding.clone()
            ),
            "pending_count": self.pending_count,
            "turn_number": self.turn_number,
            "last_embedding": (
                None if self.last_embedding is None else self.last_embedding.clone()
            ),
        }

    def _restore_before_evidence_repair(self, state: dict) -> None:
        """Restore state when the multi-model verifier rejects a change."""
        self.speakers = {key: value.clone() for key, value in state["speakers"].items()}
        self.speaker_audio = dict(state["speaker_audio"])
        self.active_speaker_id = state["active_speaker_id"]
        self.pending_embedding = state["pending_embedding"]
        self.pending_count = state["pending_count"]
        self.turn_number = state["turn_number"]
        self.last_embedding = state["last_embedding"]

    @staticmethod
    def _speech_pcm16(pcm):
        y=np.frombuffer(pcm,dtype='<i2').astype(np.float32)/32768.0
        if not len(y): return b''
        padded=np.pad(y,(1024,1024))
        frames=np.lib.stride_tricks.sliding_window_view(padded,2048)[::512]
        rms=np.sqrt(np.mean(frames*frames,axis=1))
        db=20*np.log10(np.maximum(1e-5,rms))-20*np.log10(max(1e-5,float(np.max(rms))))
        mask=db > -30
        edges=np.flatnonzero(np.diff(mask.astype(int)))+1
        if mask[0]: edges=np.r_[0,edges]
        if mask[-1]: edges=np.r_[edges,len(mask)]
        edges=np.minimum(edges*512,len(y)).reshape(-1,2)
        return b''.join(pcm[int(a)*2:int(b)*2] for a,b in edges)

    def _remember_audio(self, speaker_id, pcm16_bytes):
        # Preserve established, matched speech; short replies cannot evict it.
        # Unknown candidates use a separate buffer and cannot enter this bank.
        maximum = int(self.SAMPLE_RATE * self.MAX_REFERENCE_SECONDS) * 2
        clean = self._speech_pcm16(bytes(pcm16_bytes))
        existing = self.speaker_audio.get(speaker_id, b'')
        self.speaker_audio[speaker_id] = (existing + clean)[:maximum]

    def _pair_before_evidence_repair(self, assessment: dict, current_pcm16: bytes) -> tuple[bytes, bytes] | None:
        """Return prior-speaker and candidate audio for independent verification."""
        previous = assessment.get("previous_speaker_id")
        if not previous:
            return None
        reference = self.speaker_audio.get(previous, b"")
        if not reference or not current_pcm16:
            return None
        return reference, bytes(current_pcm16)

    @classmethod
    def _get_model(cls):
        with cls._model_lock:
            if cls._model is None:
                cls._model = SpeakerRecognition.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir="pretrained_models/speaker_ecapa",
                    local_strategy=LocalStrategy.COPY,
                )
                cls._model.eval()
            return cls._model

    @staticmethod
    def _similarity(left: torch.Tensor, right: torch.Tensor) -> float:
        return float(torch.dot(left, right).item())

    @staticmethod
    def _normalize(embedding) -> torch.Tensor:
        tensor = torch.as_tensor(embedding, dtype=torch.float32).squeeze().cpu()
        return torch.nn.functional.normalize(tensor, dim=0)

    def _encode(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.embedding_provider is not None:
            return self._normalize(self.embedding_provider(waveform))
        maximum_samples = int(self.SAMPLE_RATE * self.MAX_EMBEDDING_SECONDS)
        waveform = waveform[..., -maximum_samples:]
        model = self._get_model()
        with torch.inference_mode():
            embedding = model.encode_batch(waveform).squeeze().cpu()
        return self._normalize(embedding)

    def _unavailable(self, reason: str, **details) -> dict:
        return {
            "available": False,
            "decision": "unavailable",
            "possible_change": False,
            "reason": reason,
            **details,
        }

    def _assess_before_evidence_repair(self, pcm16_bytes: bytes | bytearray | None) -> dict:
        """Return same/uncertain/different for one completed caller turn."""
        self.turn_number += 1
        if not pcm16_bytes:
            return self._unavailable("No completed speech audio.")

        samples = array("h")
        samples.frombytes(bytes(pcm16_bytes))
        minimum_samples = int(self.SAMPLE_RATE * self.MINIMUM_SECONDS)
        if len(samples) < minimum_samples:
            return self._unavailable(
                "Speech turn too short for reliable comparison.",
                sample_count=len(samples),
            )

        rms = math.sqrt(sum(float(value) ** 2 for value in samples) / len(samples))
        if rms < self.MINIMUM_RMS:
            return self._unavailable(
                "Speech energy too low for reliable comparison.",
                rms=round(rms, 2),
            )

        # A short greeting can pass the fast 1.5-second gate but can never
        # supply the heavy verifier's five-second change-evidence requirement.
        # Wait for one longer turn instead of silently replacing a weak enrolled
        # voice with an uncertain later speaker, or combining unrelated voices.
        if not self.speakers:
            frame_samples = int(self.SAMPLE_RATE * 0.02)
            usable_samples = 0
            for offset in range(0, len(samples), frame_samples):
                frame = samples[offset:offset + frame_samples]
                frame_rms = math.sqrt(sum(float(v) ** 2 for v in frame) / len(frame))
                if frame_rms >= self.MINIMUM_RMS:
                    usable_samples += len(frame)
            usable_seconds = min(usable_samples / self.SAMPLE_RATE,
                                 len(self._speech_pcm16(bytes(pcm16_bytes))) / (self.SAMPLE_RATE * 2))
            if usable_seconds < 5.0:
                result = self._unavailable(
                    "Initial voice reference needs a longer usable speech turn; no speaker or identity claim.",
                    reference_energy_seconds=round(usable_seconds, 3),
                    required_reference_seconds=5.0,
                )
                result["decision"] = "reference_pending"
                return result

        waveform = torch.tensor(samples, dtype=torch.float32).unsqueeze(0) / 32768.0
        try:
            embedding = self._encode(waveform)
            self.last_embedding = embedding
        except Exception as error:
            return self._unavailable(
                "Speaker model unavailable; conversation continued without a voice claim.",
                error_type=type(error).__name__,
            )

        if not self.speakers:
            self.speakers["speaker_1"] = embedding
            self._remember_audio("speaker_1", pcm16_bytes)
            self.active_speaker_id = "speaker_1"
            return {
                "available": True,
                "decision": "enrolled_session_baseline",
                "baseline_created": True,
                "possible_change": False,
                "speaker_id": "speaker_1",
                "reference_speech_seconds": round(len(self.speaker_audio["speaker_1"]) / 32000, 3),
                "reason": "Anonymous in-call baseline created; identity not established.",
            }

        similarities = {
            speaker_id: self._similarity(reference, embedding)
            for speaker_id, reference in self.speakers.items()
        }
        best_speaker_id = max(similarities, key=similarities.get)
        best_similarity = similarities[best_speaker_id]

        if best_similarity >= self.SAME_SPEAKER_THRESHOLD:
            previous_speaker_id = self.active_speaker_id
            self.speakers[best_speaker_id] = self._normalize(
                (0.90 * self.speakers[best_speaker_id]) + (0.10 * embedding)
            )
            self.active_speaker_id = best_speaker_id
            self._remember_audio(best_speaker_id, pcm16_bytes)
            self.pending_embedding = None
            self.pending_count = 0
            changed_to_known = (
                previous_speaker_id is not None
                and previous_speaker_id != best_speaker_id
            )
            return {
                "available": True,
                "decision": "different" if changed_to_known else "same",
                "possible_change": changed_to_known,
                "speaker_id": best_speaker_id,
                "previous_speaker_id": previous_speaker_id,
                "similarity": round(best_similarity, 4),
                "reason": (
                    "High-confidence transition to a previously observed session speaker."
                    if changed_to_known
                    else "High-confidence match to the active session speaker."
                ),
            }

        if best_similarity > self.DIFFERENT_SPEAKER_THRESHOLD:
            self.pending_embedding = None
            self.pending_count = 0
            return {
                "available": True,
                "decision": "uncertain",
                "possible_change": False,
                "speaker_id": self.active_speaker_id,
                "similarity": round(best_similarity, 4),
                "reason": "Voice evidence is uncertain; no clarification requested.",
            }

        # When a sufficiently long turn is already far from every enrolled
        # speaker, compare two independent halves. This can confirm a genuine
        # transition in one caller turn without lowering the normal threshold
        # or asking routine clarification questions. Short turns still require
        # two mutually consistent completed turns below.
        single_turn_samples = int(
            self.SAMPLE_RATE * self.SINGLE_TURN_CONFIRMATION_SECONDS
        )
        speech_samples = array("h")
        speech_samples.frombytes(self._speech_pcm16(bytes(pcm16_bytes)))
        if len(speech_samples) >= single_turn_samples:
            midpoint = len(speech_samples) // 2
            first_waveform = (
                torch.tensor(speech_samples[:midpoint], dtype=torch.float32)
                .unsqueeze(0) / 32768.0
            )
            second_waveform = (
                torch.tensor(speech_samples[midpoint:], dtype=torch.float32)
                .unsqueeze(0) / 32768.0
            )
            try:
                first_embedding = self._encode(first_waveform)
                second_embedding = self._encode(second_waveform)
                first_known_similarity = max(
                    self._similarity(reference, first_embedding)
                    for reference in self.speakers.values()
                )
                second_known_similarity = max(
                    self._similarity(reference, second_embedding)
                    for reference in self.speakers.values()
                )
                within_turn_similarity = self._similarity(
                    first_embedding,
                    second_embedding,
                )
            except Exception:
                first_known_similarity = 1.0
                second_known_similarity = 1.0
                within_turn_similarity = -1.0

            if (
                first_known_similarity <= self.DIFFERENT_SPEAKER_THRESHOLD
                and second_known_similarity <= self.DIFFERENT_SPEAKER_THRESHOLD
                and within_turn_similarity >= self.CANDIDATE_MATCH_THRESHOLD
                and len(self.speakers) < self.MAX_SESSION_SPEAKERS
            ):
                previous_speaker_id = self.active_speaker_id
                new_speaker_id = f"speaker_{len(self.speakers) + 1}"
                self.speakers[new_speaker_id] = self._normalize(
                    first_embedding + second_embedding
                )
                self._remember_audio(new_speaker_id, pcm16_bytes)
                self.active_speaker_id = new_speaker_id
                self.pending_embedding = None
                self.pending_count = 0
                return {
                    "available": True,
                    "decision": "different",
                    "possible_change": True,
                    "speaker_id": new_speaker_id,
                    "previous_speaker_id": previous_speaker_id,
                    "similarity": round(best_similarity, 4),
                    "within_turn_similarity": round(within_turn_similarity, 4),
                    "evidence_sections": 2,
                    "reason": (
                        "Two consistent sections of one long turn support a "
                        "speaker transition; request context clarification."
                    ),
                }

        if (
            len(speech_samples) / self.SAMPLE_RATE >= 3.0
            and best_similarity <= min(0.30, self.DIFFERENT_SPEAKER_THRESHOLD)
            and len(self.speakers) < self.MAX_SESSION_SPEAKERS
        ):
            previous_speaker_id = self.active_speaker_id
            new_speaker_id = f"speaker_{len(self.speakers) + 1}"
            self.speakers[new_speaker_id] = embedding.clone()
            self._remember_audio(new_speaker_id, pcm16_bytes)
            self.active_speaker_id = new_speaker_id
            self.pending_embedding = None
            self.pending_count = 0
            return {
                "available": True,
                "decision": "different",
                "possible_change": True,
                "speaker_id": new_speaker_id,
                "previous_speaker_id": previous_speaker_id,
                "similarity": round(best_similarity, 4),
                "verification_route": "strong_first_turn_requires_independent_models",
                "within_turn_similarity": round(within_turn_similarity, 4),
                "reason": "Strong voice difference requests independent verification before responding; provisional evidence only.",
            }

        candidate_similarity = None
        if self.pending_embedding is not None:
            candidate_similarity = self._similarity(self.pending_embedding, embedding)

        if (
            self.pending_embedding is None
            or candidate_similarity < self.CANDIDATE_MATCH_THRESHOLD
        ):
            self.pending_embedding = embedding
            self.pending_count = 1
        else:
            self.pending_embedding = self._normalize(
                (0.50 * self.pending_embedding) + (0.50 * embedding)
            )
            self.pending_count += 1

        if (
            self.pending_count < self.REQUIRED_DIFFERENT_TURNS
            or len(self.speakers) >= self.MAX_SESSION_SPEAKERS
        ):
            return {
                "available": True,
                "decision": "uncertain",
                "possible_change": False,
                "speaker_id": self.active_speaker_id,
                "similarity": round(best_similarity, 4),
                "candidate_evidence_turns": self.pending_count,
                "reason": "Possible change needs another consistent voice turn.",
            }

        previous_speaker_id = self.active_speaker_id
        new_speaker_id = f"speaker_{len(self.speakers) + 1}"
        self.speakers[new_speaker_id] = self.pending_embedding
        self._remember_audio(new_speaker_id, pcm16_bytes)
        self.active_speaker_id = new_speaker_id
        self.pending_embedding = None
        self.pending_count = 0
        return {
            "available": True,
            "decision": "different",
            "possible_change": True,
            "speaker_id": new_speaker_id,
            "previous_speaker_id": previous_speaker_id,
            "similarity": round(best_similarity, 4),
            "candidate_similarity": round(candidate_similarity, 4),
            "evidence_turns": self.REQUIRED_DIFFERENT_TURNS,
            "reason": "Two consistent turns support a speaker transition; request context clarification.",
        }

    def _clear_candidate_audio(self):
        self._candidate_audio = b''
        self._candidate_embedding = None
        self._candidate_count = 0
        self._candidate_previous = None
        self._candidate_turn = None

    def snapshot_state(self):
        state = self._snapshot_before_evidence_repair()
        state['candidate_evidence'] = {
            'audio': getattr(self, '_candidate_audio', b''),
            'embedding': (getattr(self, '_candidate_embedding', None).clone()
                          if getattr(self, '_candidate_embedding', None) is not None else None),
            'count': getattr(self, '_candidate_count', 0),
            'previous': getattr(self, '_candidate_previous', None),
            'turn': getattr(self, '_candidate_turn', None),
        }
        return state

    def restore_state(self, state):
        self._restore_before_evidence_repair(state)
        evidence = state.get('candidate_evidence', {})
        self._candidate_audio = evidence.get('audio', b'')
        self._candidate_embedding = evidence.get('embedding')
        self._candidate_count = evidence.get('count', 0)
        self._candidate_previous = evidence.get('previous')
        self._candidate_turn = evidence.get('turn')

    def assess_pcm16(self, pcm16_bytes):
        previous = self.active_speaker_id
        result = self._assess_before_evidence_repair(pcm16_bytes)
        suspected = result.get('possible_change') or result.get('candidate_evidence_turns')
        if not result.get('available') or not suspected or self.last_embedding is None:
            self._clear_candidate_audio()
            # A missing/ambiguous intervening turn breaks a candidate sequence.
            # This prevents accumulation across unrelated or low-quality turns.
            if not result.get('available'):
                self.pending_embedding = None
                self.pending_count = 0
            return result
        embedding = self.last_embedding
        old_embedding = getattr(self, '_candidate_embedding', None)
        consistent = (
            old_embedding is not None
            and getattr(self, '_candidate_previous', None) == previous
            and getattr(self, '_candidate_turn', None) == self.turn_number - 1
            and self._similarity(old_embedding, embedding) >= self.CANDIDATE_MATCH_THRESHOLD
        )
        if not consistent:
            self._clear_candidate_audio()
        maximum = int(self.SAMPLE_RATE * self.MAX_REFERENCE_SECONDS) * 2
        self._candidate_audio = (getattr(self, '_candidate_audio', b'') + self._speech_pcm16(bytes(pcm16_bytes)))[-maximum:]
        self._candidate_embedding = (self._normalize(old_embedding + embedding)
                                     if consistent else embedding.clone())
        self._candidate_count = getattr(self, '_candidate_count', 0) + 1
        self._candidate_previous = previous
        self._candidate_turn = self.turn_number
        result['candidate_audio_turns'] = self._candidate_count
        result['candidate_audio_seconds'] = round(len(self._candidate_audio) / (self.SAMPLE_RATE * 2), 3)
        return result

    def multimodel_pair(self, assessment, current_pcm16):
        previous = assessment.get('previous_speaker_id')
        reference = self.speaker_audio.get(previous, b'')
        candidate = bytes(current_pcm16 or b'')
        if (getattr(self, '_candidate_previous', None) == previous
                and getattr(self, '_candidate_turn', None) == self.turn_number):
            candidate = getattr(self, '_candidate_audio', b'') or candidate
        if not reference or not candidate:
            return None
        return reference, candidate

    def resolve_multimodel(self, before, assessment, verification):
        verification = verification or {'available': False, 'status': 'no_verification_result'}
        confirmed = (verification.get('available') is True and verification.get('status')
                     == 'possible_speaker_change_needs_human_confirmation')
        assessment['multimodel_verification'] = verification
        assessment['multimodel_confirmed'] = confirmed
        if confirmed:
            current = assessment.get('speaker_id')
            if current not in before['speakers'] and getattr(self, '_candidate_audio', b''):
                self.speaker_audio[current] = self._candidate_audio
            self._clear_candidate_audio()
            return assessment
        # Restore enrolled speakers and active context, but keep anonymous,
        # coherent candidate evidence for a later independent comparison.
        evidence = self.snapshot_state()['candidate_evidence']
        current_turn = self.turn_number
        last_embedding = self.last_embedding
        provisional_id = assessment.get('speaker_id')
        self.restore_state(before)
        self.turn_number = current_turn
        self.last_embedding = last_embedding
        if verification.get('status') == 'same_speaker_evidence_false_split_suspected':
            self._clear_candidate_audio()
            self.pending_embedding = None
            self.pending_count = 0
        elif evidence.get('embedding') is not None:
            self._candidate_audio = evidence['audio']
            self._candidate_embedding = evidence['embedding']
            self._candidate_count = evidence['count']
            self._candidate_previous = evidence['previous']
            self._candidate_turn = evidence['turn']
            self.pending_embedding = evidence['embedding'].clone()
            self.pending_count = evidence['count']
        assessment['provisional_speaker_id'] = provisional_id
        assessment['speaker_id'] = self.active_speaker_id
        assessment['decision'] = 'uncertain'
        assessment['possible_change'] = False
        assessment['reason'] = 'Multi-model verification did not support a change; patient context retained and coherent candidate evidence preserved.'
        return assessment

