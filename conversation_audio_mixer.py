import wave
from pathlib import Path

import numpy as np


TARGET_SAMPLE_RATE = 24000


def _read_mono_pcm16(wav_path):
    with wave.open(str(wav_path), "rb") as wav_file:
        channel_count = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        pcm_bytes = wav_file.readframes(wav_file.getnframes())

    if sample_width != 2:
        raise ValueError("Expected 16-bit PCM microphone WAV.")

    samples = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32)

    if channel_count > 1:
        samples = samples.reshape(-1, channel_count).mean(axis=1)

    return samples, sample_rate


def _resample(samples, source_rate, target_rate):
    if samples.size == 0:
        return samples.astype(np.float32)

    if source_rate == target_rate:
        return samples.astype(np.float32, copy=True)

    output_length = max(
        1,
        round(samples.size * target_rate / source_rate),
    )
    source_positions = (
        np.arange(output_length, dtype=np.float64)
        * source_rate
        / target_rate
    )
    return np.interp(
        source_positions,
        np.arange(samples.size, dtype=np.float64),
        samples,
    ).astype(np.float32)


def save_whole_conversation_wav(
    microphone_wav_path,
    output_wav_path,
    microphone_start_perf,
    ai_segments,
    target_rate=TARGET_SAMPLE_RATE,
):
    microphone_wav_path = Path(microphone_wav_path)
    output_wav_path = Path(output_wav_path)
    microphone_samples, microphone_rate = _read_mono_pcm16(
        microphone_wav_path
    )
    microphone_samples = _resample(
        microphone_samples,
        microphone_rate,
        target_rate,
    )

    origin_candidates = []
    if microphone_samples.size and microphone_start_perf is not None:
        origin_candidates.append(microphone_start_perf)

    prepared_ai_segments = []
    for segment in ai_segments:
        start_perf = (
            segment.get("browser_start_perf")
            or segment.get("fallback_start_perf")
        )
        pcm_bytes = bytes(segment.get("pcm", b""))
        usable_bytes = pcm_bytes[:len(pcm_bytes) - len(pcm_bytes) % 2]
        audio_samples = np.frombuffer(
            usable_bytes,
            dtype="<i2",
        ).astype(np.float32)

        if start_perf is None or audio_samples.size == 0:
            continue

        stop_perf = segment.get("stop_perf")
        if stop_perf is not None:
            audible_sample_count = max(
                0,
                round((stop_perf - start_perf) * target_rate),
            )
            audio_samples = audio_samples[:audible_sample_count]

        if audio_samples.size == 0:
            continue

        origin_candidates.append(start_perf)
        prepared_ai_segments.append((start_perf, audio_samples))

    origin_perf = min(origin_candidates) if origin_candidates else 0.0
    microphone_offset = 0
    track_length = 0

    if microphone_samples.size and microphone_start_perf is not None:
        microphone_offset = max(
            0,
            round((microphone_start_perf - origin_perf) * target_rate),
        )
        track_length = max(
            track_length,
            microphone_offset + microphone_samples.size,
        )

    ai_with_offsets = []
    for start_perf, audio_samples in prepared_ai_segments:
        sample_offset = max(
            0,
            round((start_perf - origin_perf) * target_rate),
        )
        ai_with_offsets.append((sample_offset, audio_samples))
        track_length = max(
            track_length,
            sample_offset + audio_samples.size,
        )

    mixed = np.zeros(track_length, dtype=np.float32)
    if microphone_samples.size:
        mixed[
            microphone_offset:microphone_offset + microphone_samples.size
        ] += microphone_samples

    for sample_offset, audio_samples in ai_with_offsets:
        mixed[sample_offset:sample_offset + audio_samples.size] += audio_samples

    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    normalization_scale = 32767.0 / peak if peak > 32767.0 else 1.0
    output_samples = np.clip(
        np.rint(mixed * normalization_scale),
        -32768,
        32767,
    ).astype("<i2")

    output_wav_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_wav_path), "wb") as output_wav:
        output_wav.setnchannels(1)
        output_wav.setsampwidth(2)
        output_wav.setframerate(target_rate)
        output_wav.writeframes(output_samples.tobytes())

    return {
        "path": str(output_wav_path),
        "sample_rate": target_rate,
        "duration_seconds": round(output_samples.size / target_rate, 3),
        "ai_segments_included": len(ai_with_offsets),
        "normalization_scale": round(normalization_scale, 6),
    }
