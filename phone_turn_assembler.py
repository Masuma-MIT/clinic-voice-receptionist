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

    def schedule(self, on_ready: Callable[[], None], *, minimum_pause: float = 0.0) -> None:
        self.cancel_dispatch()
        if not self.fragments:
            return

        async def dispatch_after_pause() -> None:
            await asyncio.sleep(max(self.name_pause if self.name_capture else self.normal_pause, minimum_pause))
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


class PhonePromptContext:
    """Bind short assent to a confirmation actually sent before caller speech.

    Positions use samples on the same 16 kHz input timeline as STT words.
    Tokens are in-call control markers, never patient authentication.
    """
    def __init__(self):
        self.events = [(0, None)]

    @staticmethod
    def token(state):
        import json
        pending = state.get("pending_action") or {}
        if pending.get("action") and pending.get("arguments"):
            return "transaction:" + json.dumps({"action": pending["action"], "arguments": pending["arguments"]}, sort_keys=True, default=str)
        flow = state.get("reference_flow_v2") or {}
        if flow.get("simple_v6") and flow.get("stage") in {"confirm_cancel", "confirm_reschedule"}:
            record = flow.get("record") or {}
            if record.get("appointment_id"):
                action = "cancel" if flow["stage"] == "confirm_cancel" else "reschedule"
                arguments = {"appointment_id": record["appointment_id"]}
                if action == "reschedule":
                    selected = flow.get("selected") or {}
                    if not selected.get("slot_id"):
                        return None
                    arguments["new_slot_id"] = selected["slot_id"]
                return "transaction:" + json.dumps({"action": action, "arguments": arguments}, sort_keys=True, default=str)
        for field in ("nameflow_v1", "lookup_nameflow"):
            flow = state.get(field) or {}
            if flow.get("stage") == "readback" and flow.get("candidate"):
                return "name:" + json.dumps({"candidate":flow["candidate"],"parts":flow.get("parts")},sort_keys=True)
        return None

    @classmethod
    def spoken_token(cls, state, text):
        import re
        token = cls.token(state)
        value = str(text or "").lower()
        if token and token.startswith("transaction:"):
            if not re.search(r"\b(?:confirm|proceed)\b", value):
                return None
        if token and token.startswith("name:"):
            if "name" not in value or not any(word in value for word in ("correct", "spelling", "letters", "spelled")):
                return None
        return token

    def publish(self, input_sample, token):
        self.events.append((int(input_sample), token))

    def capture(self, speech_start_seconds):
        if speech_start_seconds is None:
            return None
        sample = int(float(speech_start_seconds) * 16000)
        for position, token in reversed(self.events):
            if position <= sample:
                return token
        return None
