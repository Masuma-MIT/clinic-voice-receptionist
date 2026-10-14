# SPEAKER_SHORT_CONSENSUS_CALIBRATION_V2_20261010
# SPEAKER_BOUNDED_CLEAN_COMPARISON_V1_20261010
"""Persistent Pyannote + TitaNet + ECAPA worker for live confirmation.

The worker runs in .venv_speaker_lab, loads models once, and accepts JSON lines
from the phone gateway. It never receives transcripts, names, or database data.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio

if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: []

from conversation_speaker_guard_v3 import conservative_assessment, load_ecapa


PREFIX = "V3_RESULT "
SAMPLE_RATE = 16000


class Models:
    def __init__(self):
        self.pyannote = None
        self.titanet = None
        self.ecapa = None

    def load(self) -> None:
        if self.pyannote is not None:
            return
        from pyannote.audio import Pipeline
        from nemo.collections.asr.models import EncDecSpeakerLabelModel

        root = Path(__file__).resolve().parent
        self.pyannote = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1", token=True
        )
        self.titanet = EncDecSpeakerLabelModel.from_pretrained(
            "nvidia/speakerverification_en_titanet_large", map_location="cpu"
        )
        self.ecapa = load_ecapa(
            root / "speaker_models" / "speechbrain_ecapa_local",
            root / "speaker_models" / "speechbrain_ecapa_runtime_live_v3",
        )


MODELS = Models()


def finite_cosine(left, right) -> float:
    score = float(torch.nn.functional.cosine_similarity(
        left.reshape(1, -1), right.reshape(1, -1)
    ).item())
    if not math.isfinite(score):
        raise ValueError("Non-finite speaker similarity")
    return round(score, 4)


def speech_only(path: Path) -> tuple[np.ndarray, float]:
    waveform = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)[0].astype(np.float32)
    intervals = librosa.effects.split(waveform, top_db=30)
    if not len(intervals):
        return np.zeros(0, dtype=np.float32), 0.0
    clean = np.concatenate([waveform[start:end] for start, end in intervals])
    return clean.astype(np.float32), len(clean) / SAMPLE_RATE


def verify(reference: Path, candidate: Path) -> dict:
    MODELS.load()
    left, seconds_left = speech_only(reference)
    right, seconds_right = speech_only(candidate)
    # Bound compute using clear speech, not a silence-filled trailing window.
    left = left[:8 * SAMPLE_RATE]
    right = right[-8 * SAMPLE_RATE:]
    seconds_left = len(left) / SAMPLE_RATE
    seconds_right = len(right) / SAMPLE_RATE
    if min(seconds_left, seconds_right) < 3.0:
        return {
            "available": True,
            **conservative_assessment(0.0, 0.0, seconds_left, seconds_right),
            "pyannote_proposed_speakers": None,
            "reference_speech_seconds": round(seconds_left, 3),
            "candidate_speech_seconds": round(seconds_right, 3),
        }

    with tempfile.TemporaryDirectory(prefix="speaker_guard_worker_") as temporary:
        folder = Path(temporary)
        left_path = folder / "left_clean.wav"
        right_path = folder / "right_clean.wav"
        combined_path = folder / "combined.wav"
        sf.write(left_path, left, SAMPLE_RATE, subtype="PCM_16")
        sf.write(right_path, right, SAMPLE_RATE, subtype="PCM_16")
        pause = np.zeros(int(1.25 * SAMPLE_RATE), dtype=np.float32)
        combined = np.concatenate([left, pause, right])
        sf.write(combined_path, combined, SAMPLE_RATE, subtype="PCM_16")

        result = MODELS.pyannote({
            "waveform": torch.from_numpy(combined).unsqueeze(0),
            "sample_rate": SAMPLE_RATE,
        })
        annotation = result.speaker_diarization
        proposed = len(annotation.labels())

        with torch.inference_mode():
            tita_left = MODELS.titanet.get_embedding(str(left_path))
            tita_right = MODELS.titanet.get_embedding(str(right_path))
            ecapa_left = MODELS.ecapa.encode_batch(
                torch.from_numpy(left).unsqueeze(0)
            )
            ecapa_right = MODELS.ecapa.encode_batch(
                torch.from_numpy(right).unsqueeze(0)
            )

        assessment = conservative_assessment(
            finite_cosine(tita_left, tita_right),
            finite_cosine(ecapa_left, ecapa_right),
            seconds_left,
            seconds_right,
        )
        if (
            assessment["status"] == "possible_speaker_change_needs_human_confirmation"
            and proposed < 2
            and not (
                seconds_left >= 5.0 and 3.0 <= seconds_right < 5.0
                and assessment["titanet_similarity"] < 0.30
                and assessment["ecapa_similarity"] < 0.30
                and assessment["absolute_model_gap"] <= 0.15
            )
        ):
            assessment = {
                **assessment,
                "status": "uncertain_do_not_trigger_change",
                "explanation": (
                    "Verification models suggested a difference, but Pyannote did "
                    "not independently propose two speakers."
                ),
            }
        return {
            "available": True,
            **assessment,
            "pyannote_proposed_speakers": proposed,
            "pyannote_corroborated": proposed >= 2,
            "clarification_basis": (
                "strong_short_turn_verification_consensus" if proposed < 2
                and assessment["status"] == "possible_speaker_change_needs_human_confirmation"
                else "standard_multimodel_policy"
            ),
            "identity_verified": False,
            "reference_speech_seconds": round(seconds_left, 3),
            "candidate_speech_seconds": round(seconds_right, 3),
        }


def respond(payload: dict) -> dict:
    action = payload.get("action")
    if action == "warmup":
        MODELS.load()
        return {"available": True, "status": "ready"}
    if action == "verify":
        return verify(Path(payload["reference"]), Path(payload["candidate"]))
    raise ValueError(f"Unsupported worker action: {action}")


def main() -> int:
    for line in sys.stdin:
        try:
            payload = json.loads(line)
            result = respond(payload)
        except Exception as error:
            result = {
                "available": False,
                "status": "worker_error",
                "error_type": type(error).__name__,
                "reason": str(error),
            }
        print(PREFIX + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

