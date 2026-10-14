"""Prepare speaker audio without changing STT or call recordings."""
import audioop

def trim_speaker_edges(pcm, threshold=180, padding_ms=300):
    pcm = bytes(pcm)
    if len(pcm) % 2:
        raise ValueError("PCM16 must contain complete samples")
    frame_bytes = 640
    active = []
    for start in range(0, len(pcm), frame_bytes):
        frame = pcm[start:start + frame_bytes]
        if frame and audioop.rms(frame, 2) >= threshold:
            active.append(start)
    if not active:
        return b""
    padding_bytes = int(16000 * padding_ms / 1000) * 2
    first = max(0, active[0] - padding_bytes)
    last = min(len(pcm), active[-1] + frame_bytes + padding_bytes)
    return pcm[first:last]
