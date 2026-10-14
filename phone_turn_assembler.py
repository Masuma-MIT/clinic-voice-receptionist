"""Debounce Deepgram phone endpoints without losing short speech fragments.

This module has no network or database dependencies.  It only joins final
transcript fragments from the same caller utterance; it never guesses letters.
"""

import asyncio
from collections.abc import Callable


class PhoneTurnAssembler:
    def __init__(self, *, normal_pause: float = 0.45, name_pause: float = 2.4):
        self.normal_pause = normal_pause
        self.name_pause = name_pause
        self.name_capture = False
        self.fragments: list[str] = []
        self._dispatch_task: asyncio.Task | None = None

    def add_final(self, transcript: str) -> None:
        if transcript.strip():
            self.fragments.append(transcript.strip())

    def caller_is_speaking(self) -> None:
        self.cancel_dispatch()

    def schedule(self, on_ready: Callable[[], None]) -> None:
        self.cancel_dispatch()
        if not self.fragments:
            return

        async def dispatch_after_pause() -> None:
            await asyncio.sleep(self.name_pause if self.name_capture else self.normal_pause)
            if self.fragments:
                on_ready()

        self._dispatch_task = asyncio.create_task(dispatch_after_pause())

    def take(self) -> str:
        transcript = " ".join(self.fragments).strip()
        self.fragments.clear()
        self.cancel_dispatch()
        return transcript

    def cancel_dispatch(self) -> None:
        task = self._dispatch_task
        self._dispatch_task = None
        if task is not None and not task.done() and task is not asyncio.current_task():
            task.cancel()
