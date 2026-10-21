# SPEAKER_SHORT_CONSENSUS_CALIBRATION_V2_20261010
# SPEAKER_BOUNDED_CLEAN_COMPARISON_V1_20261010
"""Persistent Pyannote + TitaNet + ECAPA worker for live confirmation.

The worker runs in .venv_speaker_lab, loads models once, and accepts JSON lines
from the phone gateway. It never receives transcripts, names, or database data.
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time
import hashlib
from collections import OrderedDict
from pathlib import Path

# The gateway already uses one PyTorch CPU thread. The isolated worker
# needs its own setting: it does not inherit torch.set_num_threads().
def _bounded_integer(name, default, maximum):
    try:
        return max(1, min(maximum, int(os.getenv(name, str(default)))))
    except (TypeError, ValueError):
        return default

SPEAKER_CPU_THREADS = _bounded_integer(
    "SPEAKER_LAB_TORCH_THREADS", 1, os.cpu_count() or 1,
)
# Explicit local settings take priority; these defaults precede model imports.
os.environ.setdefault("OMP_NUM_THREADS", str(SPEAKER_CPU_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(SPEAKER_CPU_THREADS))
REFERENCE_CACHE_SIZE = 8

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio

torch.set_num_threads(SPEAKER_CPU_THREADS)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    # A library may have started its inter-op pool during import.
    pass

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
# Volatile, bounded embeddings only; no names, identity labels or disk profiles.
_REFERENCE_EMBEDDINGS = OrderedDict()


def report_timing(stage, event, started, **values):
    print("V3_TIMING " + json.dumps({
        "stage": stage, "event": event,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        **values,
    }), flush=True)


def reference_embeddings(left, left_path, timings):
    # Match exact cleaned float32 samples, not a filename or a speaker label.
    key = hashlib.sha256(left.tobytes()).digest()
    cached = _REFERENCE_EMBEDDINGS.get(key)
    timings["reference_cache_hit"] = cached is not None
    if cached is not None:
        _REFERENCE_EMBEDDINGS.move_to_end(key)
        return tuple(value.clone() for value in cached)
    started = time.perf_counter()
    report_timing("reference_embeddings", "started", started)
    with torch.inference_mode():
        tita = MODELS.titanet.get_embedding(str(left_path))
        ecapa = MODELS.ecapa.encode_batch(torch.from_numpy(left).unsqueeze(0))
    timings["reference_embeddings_ms"] = round((time.perf_counter() - started) * 1000, 2)
    report_timing("reference_embeddings", "complete", started)
    _REFERENCE_EMBEDDINGS[key] = tuple(
        value.detach().cpu().clone() for value in (tita, ecapa)
    )
    while len(_REFERENCE_EMBEDDINGS) > REFERENCE_CACHE_SIZE:
        _REFERENCE_EMBEDDINGS.popitem(last=False)
    return tita, ecapa


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
    verification_started = time.perf_counter()
    timings = {"cpu_threads": torch.get_num_threads()}
    stage_started = time.perf_counter()
    report_timing("model_load", "started", stage_started)
    MODELS.load()
    timings["model_load_ms"] = round((time.perf_counter() - stage_started) * 1000, 2)
    report_timing("model_load", "complete", stage_started)
    stage_started = time.perf_counter()
    report_timing("audio_preparation", "started", stage_started)
    left, seconds_left = speech_only(reference)
    right, seconds_right = speech_only(candidate)
    timings["audio_preparation_ms"] = round((time.perf_counter() - stage_started) * 1000, 2)
    report_timing("audio_preparation", "complete", stage_started)
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
            "timings_ms": {**timings, "verification_total_ms": round((time.perf_counter() - verification_started) * 1000, 2)},
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

        tita_left, ecapa_left = reference_embeddings(left, left_path, timings)
        with torch.inference_mode():
            stage_started = time.perf_counter()
            report_timing("titanet_candidate", "started", stage_started)
            tita_right = MODELS.titanet.get_embedding(str(right_path))
            timings["titanet_candidate_ms"] = round((time.perf_counter() - stage_started) * 1000, 2)
            report_timing("titanet_candidate", "complete", stage_started)
            stage_started = time.perf_counter()
            report_timing("ecapa_candidate", "started", stage_started)
            ecapa_right = MODELS.ecapa.encode_batch(
                torch.from_numpy(right).unsqueeze(0)
            )
            timings["ecapa_candidate_ms"] = round((time.perf_counter() - stage_started) * 1000, 2)
            report_timing("ecapa_candidate", "complete", stage_started)

        assessment = conservative_assessment(
            finite_cosine(tita_left, tita_right),
            finite_cosine(ecapa_left, ecapa_right),
            seconds_left,
            seconds_right,
        )
        # Pyannote can veto a proposed change, but it cannot promote a
        # same/uncertain embedding result into a change under the existing policy.
        proposed = None
        if assessment["status"] == "possible_speaker_change_needs_human_confirmation":
            stage_started = time.perf_counter()
            report_timing("pyannote", "started", stage_started)
            result = MODELS.pyannote({
                "waveform": torch.from_numpy(combined).unsqueeze(0),
                "sample_rate": SAMPLE_RATE,
            })
            annotation = result.speaker_diarization
            proposed = len(annotation.labels())
            timings["pyannote_ms"] = round((time.perf_counter() - stage_started) * 1000, 2)
            report_timing("pyannote", "complete", stage_started)
            timings["pyannote_ran"] = True
        else:
            timings["pyannote_ran"] = False
            timings["pyannote_ms"] = 0.0
            report_timing("pyannote", "not_required_for_embedding_decision", verification_started)
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
            "timings_ms": {**timings, "verification_total_ms": round((time.perf_counter() - verification_started) * 1000, 2)},
            **assessment,
            "pyannote_proposed_speakers": proposed,
            "pyannote_corroborated": proposed >= 2 if proposed is not None else None,
            "pyannote_assessment": "performed" if proposed is not None else "not_required_no_change_consensus",
            "clarification_basis": (
                "strong_short_turn_verification_consensus" if proposed is not None and proposed < 2
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
        return {"available": True, "status": "ready", "cpu_threads": torch.get_num_threads()}
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

