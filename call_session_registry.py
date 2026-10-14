"""Concurrency accounting for isolated Twilio call sessions."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone


class CallCapacityError(RuntimeError):
    pass


class CallSessionRegistry:
    """Track active calls without storing patient text, audio, or identity data."""

    def __init__(self, maximum: int | None = None):
        configured = maximum if maximum is not None else int(
            os.getenv("MAX_CONCURRENT_PHONE_CALLS", "4")
        )
        self.maximum = max(1, configured)
        self._active: dict[str, str] = {}
        self._lock = asyncio.Lock()

    async def acquire(self, session_id: str) -> dict:
        async with self._lock:
            if session_id in self._active:
                raise ValueError(f"Duplicate session id: {session_id}")
            if len(self._active) >= self.maximum:
                raise CallCapacityError("Telephone gateway is at configured capacity.")
            self._active[session_id] = datetime.now(timezone.utc).isoformat()
            return self.snapshot_unlocked()

    async def release(self, session_id: str) -> dict:
        async with self._lock:
            self._active.pop(session_id, None)
            return self.snapshot_unlocked()

    async def snapshot(self) -> dict:
        async with self._lock:
            return self.snapshot_unlocked()

    def snapshot_unlocked(self) -> dict:
        return {
            "active_calls": len(self._active),
            "maximum_calls": self.maximum,
            "available_capacity": max(0, self.maximum - len(self._active)),
            # Anonymous session ids only; no caller or patient information.
            "session_ids": sorted(self._active),
        }

