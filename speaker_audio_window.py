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


class TranscriptSpeakerAudio:
    """Select caller samples by final STT word timestamps on the sent stream.

    Missing or invalid alignment yields no voice evidence, never another turn's
    audio. The complete input recording and transcription stream stay intact.
    """
    def __init__(self, sample_rate=16000, padding_seconds=0.08):
        self.sample_rate = sample_rate
        self.padding_seconds = padding_seconds
        self.ranges = []
        self.invalid = False

    @staticmethod
    def value(obj, name, default=None):
        return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)

    def add_final(self, alternative):
        import math
        words = self.value(alternative, "words", []) or []
        intervals = []
        for word in words:
            try:
                start = float(self.value(word, "start"))
                end = float(self.value(word, "end"))
            except (TypeError, ValueError):
                self.invalid = True
                return
            if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
                self.invalid = True
                return
            intervals.append((start, end))
        if not intervals:
            self.invalid = True
            return
        self.ranges.extend(intervals)

    def take(self, stream_pcm16):
        ranges, invalid = self.ranges, self.invalid
        self.ranges, self.invalid = [], False
        if invalid or not ranges:
            return b"", {"status": "unavailable_word_alignment", "audio_seconds": 0.0}
        total_seconds = len(stream_pcm16) / (self.sample_rate * 2)
        # Do not silently borrow a different segment when timing is ahead of
        # received audio or when a provider result is malformed.
        if any(end > total_seconds + 0.10 for start, end in ranges):
            return b"", {"status": "word_timing_outside_received_audio", "audio_seconds": 0.0}
        windows = []
        for start, end in sorted(ranges):
            a = max(0, int((start - self.padding_seconds) * self.sample_rate))
            b = min(len(stream_pcm16)//2, int((end + self.padding_seconds) * self.sample_rate))
            if b <= a:
                return b"", {"status": "empty_word_audio", "audio_seconds": 0.0}
            if windows and a <= windows[-1][1]:
                windows[-1] = (windows[-1][0], max(b, windows[-1][1]))
            else:
                windows.append((a,b))
        audio = b"".join(bytes(stream_pcm16[a*2:b*2]) for a,b in windows)
        return audio, {
            "status": "final_word_timestamps", "word_intervals": len(ranges),
            "first_word_start_seconds": min(start for start,end in ranges),
            "last_word_end_seconds": max(end for start,end in ranges),
            "audio_seconds": round(len(audio)/(self.sample_rate*2), 3),
            "stream_start_seconds": round(min(a for a,b in windows)/self.sample_rate,3),
            "stream_end_seconds": round(max(b for a,b in windows)/self.sample_rate,3),
        }
