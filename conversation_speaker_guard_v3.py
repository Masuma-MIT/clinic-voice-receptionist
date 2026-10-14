# SPEAKER_SHORT_CONSENSUS_CALIBRATION_V2_20261010
# SPEAKER_SHORT_TURN_CLARIFICATION_V1_20261010
"""Offline, conservative conversation-level speaker-change research tool.

Pyannote proposes clusters. TitaNet and ECAPA independently verify whether
those clusters are acoustically different. No live receptionist code is used.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

SAMPLE_RATE = 16000
TITANET_MODEL = "nvidia/speakerverification_en_titanet_large"


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    import torch
    value = float(torch.nn.functional.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item())
    if not math.isfinite(value):
        raise ValueError("A model produced a non-finite similarity score")
    return round(value, 4)


def quality(seconds: float) -> str:
    if seconds < 3.0:
        return "insufficient"
    if seconds < 5.0:
        return "limited"
    return "suitable"


def conservative_assessment(tita: float, ecapa: float, seconds_a: float, seconds_b: float) -> dict:
    """Provisional evidence policy; intentionally refuses borderline decisions."""
    gap = round(abs(tita - ecapa), 4)
    if min(seconds_a, seconds_b) < 3.0:
        status = "insufficient_evidence"
        explanation = "At least one proposed speaker has under 3 seconds of usable speech."
    elif tita >= 0.75 and ecapa >= 0.75 and gap <= 0.15:
        status = "same_speaker_evidence_false_split_suspected"
        explanation = "Both verification models strongly support the same speaker despite separate diarization labels."
    elif min(seconds_a, seconds_b) >= 5.0 and tita < 0.65 and ecapa < 0.65 and gap <= 0.20:
        status = "possible_speaker_change_needs_human_confirmation"
        explanation = "Both verification models support a possible change, but identity is never inferred automatically."
    elif seconds_a >= 5.0 and seconds_b >= 3.0 and tita < 0.30 and ecapa < 0.30 and gap <= 0.15:
        # Short candidates require substantially stronger agreement. The live
        # gateway also requires coherent fast evidence and worker Pyannote >=2.
        # This requests human clarification; it never establishes identity.
        status = "possible_speaker_change_needs_human_confirmation"
        explanation = "A usable reference and strong model agreement support clarification of a short new-speaker turn."
    else:
        status = "uncertain_do_not_trigger_change"
        explanation = "Evidence is mixed or borderline; the safe action is to avoid a speaker-change claim."
    return {
        "status": status,
        "titanet_similarity": tita,
        "ecapa_similarity": ecapa,
        "absolute_model_gap": gap,
        "explanation": explanation,
    }


def load_ecapa(source: Path, runtime: Path):
    import librosa
    import torch
    import huggingface_hub
    import torchaudio
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: []
    original_download = huggingface_hub.hf_hub_download

    def compatible_download(*args, **kwargs):
        if "use_auth_token" in kwargs:
            kwargs["token"] = kwargs.pop("use_auth_token")
        return original_download(*args, **kwargs)

    huggingface_hub.hf_hub_download = compatible_download
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy
    return EncoderClassifier.from_hparams(
        source=str(source), savedir=str(runtime), run_opts={"device": "cpu"},
        local_strategy=LocalStrategy.COPY,
    )


def main() -> int:
    import librosa
    import numpy as np
    import soundfile as sf
    import torch
    parser = argparse.ArgumentParser(description="Offline Pyannote + TitaNet + ECAPA conversation speaker guard")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("speaker_lab_outputs/conversation_guard_v3"))
    parser.add_argument("--ecapa-source", type=Path, default=Path("speaker_models/speechbrain_ecapa_local"))
    parser.add_argument("--ecapa-runtime", type=Path, default=Path("speaker_models/speechbrain_ecapa_runtime_v3"))
    args = parser.parse_args()

    try:
        if not args.audio.is_file():
            raise FileNotFoundError(f"Audio file not found: {args.audio}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        waveform, sr = librosa.load(str(args.audio), sr=SAMPLE_RATE, mono=True)
        waveform = waveform.astype(np.float32)

        import torchaudio
        if not hasattr(torchaudio, "list_audio_backends"):
            torchaudio.list_audio_backends = lambda: []
        from pyannote.audio import Pipeline
        print("Running Pyannote diarization...")
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=True)
        result = pipeline({"waveform": torch.from_numpy(waveform).unsqueeze(0), "sample_rate": sr})
        annotation = result.speaker_diarization

        segments = []
        samples: dict[str, list[np.ndarray]] = defaultdict(list)
        for turn, _, label in annotation.itertracks(yield_label=True):
            start = max(0, int(turn.start * sr))
            end = min(len(waveform), int(turn.end * sr))
            if end <= start:
                continue
            samples[label].append(waveform[start:end])
            segments.append({"start": round(turn.start, 3), "end": round(turn.end, 3), "speaker": label})

        clusters = {}
        for label, parts in sorted(samples.items()):
            audio = np.concatenate(parts).astype(np.float32)
            path = args.output_dir / f"cluster_{label}.wav"
            sf.write(path, audio, sr, subtype="PCM_16")
            seconds = len(audio) / sr
            clusters[label] = {
                "audio_file": str(path.resolve()),
                "speech_seconds": round(seconds, 3),
                "quality": quality(seconds),
                "segment_count": len(parts),
            }

        report = {
            "tool": "conversation_speaker_guard_v3",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_audio": str(args.audio.resolve()),
            "research_warning": "Evidence only. Provisional thresholds require additional labelled same/different-speaker calibration.",
            "pyannote": {"proposed_speaker_count": len(clusters), "segments": segments, "clusters": clusters},
            "pairwise_verification": [],
            "conversation_assessment": {},
            "safety": {"live_v5_changes": False, "database_writes": False, "paid_calls": False},
        }

        labels = list(clusters)
        if len(labels) <= 1:
            report["conversation_assessment"] = {
                "status": "one_speaker_detected",
                "confirmed_change": False,
                "reason": "Pyannote proposed only one speaker cluster.",
            }
        else:
            eligible = [x for x in labels if clusters[x]["speech_seconds"] >= 3.0]
            if len(eligible) < 2:
                report["conversation_assessment"] = {
                    "status": "insufficient_evidence",
                    "confirmed_change": False,
                    "reason": "Fewer than two clusters contain at least 3 seconds of speech.",
                }
            else:
                print("Loading TitaNet once...")
                from nemo.collections.asr.models import EncDecSpeakerLabelModel
                titanet = EncDecSpeakerLabelModel.from_pretrained(TITANET_MODEL, map_location="cpu")
                print("Loading SpeechBrain ECAPA once...")
                ecapa = load_ecapa(args.ecapa_source, args.ecapa_runtime)
                embeddings = {}
                with torch.inference_mode():
                    for label in eligible:
                        path = clusters[label]["audio_file"]
                        tita_e = titanet.get_embedding(path)
                        y = torch.tensor(librosa.load(path, sr=sr, mono=True)[0]).unsqueeze(0)
                        ecapa_e = ecapa.encode_batch(y)
                        embeddings[label] = (tita_e, ecapa_e)
                for i, first in enumerate(eligible):
                    for second in eligible[i + 1:]:
                        tita_score = cosine(embeddings[first][0], embeddings[second][0])
                        ecapa_score = cosine(embeddings[first][1], embeddings[second][1])
                        assessment = conservative_assessment(
                            tita_score, ecapa_score,
                            clusters[first]["speech_seconds"], clusters[second]["speech_seconds"],
                        )
                        report["pairwise_verification"].append({"speaker_a": first, "speaker_b": second, **assessment})

                statuses = [x["status"] for x in report["pairwise_verification"]]
                possible = "possible_speaker_change_needs_human_confirmation" in statuses
                same_only = statuses and all(x == "same_speaker_evidence_false_split_suspected" for x in statuses)
                if possible:
                    status, reason = "possible_change_needs_human_confirmation", "Diarization and both verification models provide change evidence."
                elif same_only:
                    status, reason = "false_split_suspected_no_change", "Verification models indicate that Pyannote split the same voice."
                else:
                    status, reason = "uncertain_do_not_trigger_change", "Evidence is not strong enough for a safe change claim."
                report["conversation_assessment"] = {
                    "status": status,
                    "confirmed_change": False,
                    "reason": reason,
                }

        report_path = args.output_dir / "conversation_speaker_guard_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("CONVERSATION SPEAKER GUARD V3 COMPLETE")
        print("PYANNOTE PROPOSED SPEAKERS:", len(clusters))
        for item in report["pairwise_verification"]:
            print(f"PAIR {item['speaker_a']} vs {item['speaker_b']}: TitaNet={item['titanet_similarity']} ECAPA={item['ecapa_similarity']} => {item['status']}")
        print("FINAL CONSERVATIVE STATUS:", report["conversation_assessment"]["status"])
        print("REPORT:", report_path.resolve())
        print("LIVE V5 CHANGES: NONE")
        print("DATABASE WRITES: NONE")
        print("PAID CALLS: NONE")
        return 0
    except Exception as exc:
        print(f"CONVERSATION SPEAKER GUARD V3 FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
