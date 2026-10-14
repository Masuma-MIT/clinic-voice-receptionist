# SPEAKER_VERIFICATION_DEADLINE_V1_20261004
"""Safe bridge from the live gateway to the isolated speaker-lab environment."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import wave
from speaker_worker_io import start_worker_reader, request_worker, stop_worker
from pathlib import Path


class LiveMultiModelVerifier:
    """One shared worker, serialised to protect CPU/RAM during concurrent calls."""

    RESULT_PREFIX = "V3_RESULT "

    def __init__(self):
        root = Path(__file__).resolve().parent
        self.enabled = os.getenv("SPEAKER_GUARD_V3_ENABLED", "false").lower() == "true"
        self.python = Path(os.getenv(
            "SPEAKER_LAB_PYTHON",
            str(root / ".venv_speaker_lab" / "Scripts" / "python.exe"),
        ))
        self.worker = Path(os.getenv(
            "SPEAKER_GUARD_V3_WORKER",
            str(root / "speaker_guard_v3_worker.py"),
        ))
        self.timeout = float(os.getenv("SPEAKER_GUARD_V3_TIMEOUT_SECONDS", "35"))
        self._process: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._output = None
        self.warmup_timeout = float(os.getenv("SPEAKER_GUARD_V3_WARMUP_TIMEOUT_SECONDS", "120"))

    def available(self) -> bool:
        return self.enabled and self.python.is_file() and self.worker.is_file()

    def _start_locked(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        self._process = subprocess.Popen(
            [str(self.python), "-u", str(self.worker)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            bufsize=1,
            cwd=str(self.worker.parent),
        )
        self._output = start_worker_reader(self._process)

    def _request_locked(self, request: dict) -> dict:
        self._start_locked()
        assert self._process is not None and self._output is not None
        timeout = self.warmup_timeout if request.get("action") == "warmup" else self.timeout
        try:
            return request_worker(self._process, self._output, request, timeout, self.RESULT_PREFIX)
        except Exception:
            self._process = None
            self._output = None
            raise

    def warmup(self) -> dict:
        if not self.available():
            return {"available": False, "status": "disabled_or_missing"}
        if not self._lock.acquire(blocking=False):
            return {"available": False, "status": "worker_busy"}
        try:
            return self._request_locked({"action": "warmup"})
        finally:
            self._lock.release()

    @staticmethod
    def _write_pcm16(path: Path, audio: bytes) -> None:
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(audio)

    def verify_pcm16(self, reference: bytes, candidate: bytes) -> dict:
        if not self.available():
            return {"available": False, "status": "disabled_or_missing"}
        if not self._lock.acquire(blocking=False):
            return {"available": False, "status": "worker_busy"}
        try:
            with tempfile.TemporaryDirectory(prefix="speaker_guard_v3_") as temporary:
                folder = Path(temporary)
                left = folder / "reference.wav"
                right = folder / "candidate.wav"
                self._write_pcm16(left, reference)
                self._write_pcm16(right, candidate)
                return self._request_locked({"action": "verify", "reference": str(left), "candidate": str(right)})
        finally:
            self._lock.release()

    def close(self) -> None:
        with self._lock:
            process = self._process
            self._process = None
            self._output = None
            if process is not None:
                stop_worker(process)

