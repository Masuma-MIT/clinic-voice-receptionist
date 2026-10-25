# CALL_FLOW_STABILITY_V4_20261012
# NONBLOCKING_FAST_SPEAKER_WARMUP_V1_20261007
# WHOLE_CALL_SPEAKER_EVIDENCE_V1_20261003
"""Separate Twilio Media Streams gateway for the Clinic Voice Receptionist.

This service deliberately runs on port 8001 and does not modify app_final.py.
It accepts Twilio's 8 kHz mu-law call audio, converts it to 16 kHz PCM for
Deepgram Nova-3 and the existing speaker-change guard, then returns Aura-2
audio to Twilio as 8 kHz mu-law.
"""

import asyncio
from contextlib import aclosing
import audioop
import base64
import html
import json
import os
import time
import wave
from datetime import datetime
from pathlib import Path

import torch
torch.set_num_threads(1)

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType

from aura2_tts_streaming import stream_aura2
from call_session_registry import CallCapacityError, CallSessionRegistry
from live_multimodel_verifier import LiveMultiModelVerifier
from phone_turn_assembler import PhoneTurnAssembler, PhonePromptContext
from phone_stt_monitor import SpeechListenerStopped, receive_phone_message
from receptionist_controller_qwen37_experiment import create_receptionist_session
from speaker_change_guard import SpeakerChangeGuard
from speaker_audio_window import trim_speaker_edges, TranscriptSpeakerAudio
from voice_identity_store import VoiceIdentityStore


async def assess_phone_speaker_when_ready(guard, warmup_task, pcm16):
    """Keep enquiries responsive while the fast speaker model is starting."""
    if not warmup_task.done():
        return {
            "available": False,
            "decision": "unavailable",
            "possible_change": False,
            "status": "fast_model_warming_up",
            "reason": "Speaker model is loading; conversation continues without a voice claim.",
        }
    try:
        warmup_task.result()
    except (asyncio.CancelledError, Exception) as error:
        return {
            "available": False,
            "decision": "unavailable",
            "possible_change": False,
            "status": "fast_model_warmup_failed",
            "error_type": type(error).__name__,
            "reason": "Speaker model is unavailable; conversation continues without a voice claim.",
        }
    return await asyncio.to_thread(guard.assess_pcm16, pcm16)


load_dotenv()

app = FastAPI(title="Clinic Voice Receptionist - Twilio Phone Gateway")

CALL_REGISTRY = CallSessionRegistry()
MULTIMODEL_VERIFIER = LiveMultiModelVerifier()

PHONE_AUDIO_FOLDER = Path("phone_audio_logs")
PHONE_CONVERSATION_FOLDER = Path("phone_conversation_logs")
PHONE_AUDIO_FOLDER.mkdir(exist_ok=True)
PHONE_CONVERSATION_FOLDER.mkdir(exist_ok=True)

TWILIO_FRAME_BYTES = 160  # 20 ms of 8 kHz mu-law audio


def public_base_url() -> str:
    """Return the temporary public HTTPS tunnel URL configured locally."""
    value = os.getenv("TWILIO_PUBLIC_BASE_URL", "").strip().rstrip("/")
    if not value.startswith("https://"):
        raise RuntimeError(
            "TWILIO_PUBLIC_BASE_URL must be an https:// public tunnel URL."
        )
    return value


def to_twiml_stream_response() -> str:
    base_url = public_base_url()
    stream_url = "wss://" + base_url.removeprefix("https://") + "/twilio/media"
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
        "<Response><Connect><Stream url=\""
        + html.escape(stream_url, quote=True)
        + "\" /></Connect></Response>"
    )


def mulaw_8k_to_pcm16_16k(mulaw_audio: bytes, rate_state=None):
    """Convert Twilio inbound audio to the existing 16 kHz PCM pipeline format."""
    pcm16_8k = audioop.ulaw2lin(mulaw_audio, 2)
    pcm16_16k, next_state = audioop.ratecv(
        pcm16_8k,
        2,
        1,
        8000,
        16000,
        rate_state,
    )
    return pcm16_16k, next_state


async def send_aura2_to_twilio(
    websocket: WebSocket,
    stream_sid: str,
    text: str,
    stream_provider=stream_aura2,
    on_first_audio=None,
):
    """Send paced phone audio, retrying only before any audio was delivered."""
    phone_frame_bytes = TWILIO_FRAME_BYTES * 5  # 100 ms
    last_error = None
    delivery_started = time.perf_counter()
    failed_attempts = []

    for attempt in range(1, 4):
        rate_state = None
        pending = bytearray()
        frames_sent = 0
        attempt_started = time.perf_counter()
        first_audio_wait_ms = None

        try:
            async with aclosing(stream_provider(text)) as audio_stream:
                async for pcm24k in audio_stream:
                    pcm8k, rate_state = audioop.ratecv(
                        pcm24k,
                        2,
                        1,
                        24000,
                        8000,
                        rate_state,
                    )
                    pending.extend(audioop.lin2ulaw(pcm8k, 2))
    
                    while len(pending) >= phone_frame_bytes:
                        frame = bytes(pending[:phone_frame_bytes])
                        del pending[:phone_frame_bytes]
                        await websocket.send_text(
                            json.dumps(
                                {
                                    "event": "media",
                                    "streamSid": stream_sid,
                                    "media": {
                                        "payload": base64.b64encode(frame).decode("ascii"),
                                    },
                                }
                            )
                        )
                        if first_audio_wait_ms is None:
                            first_audio_wait_ms = round(
                                (time.perf_counter() - attempt_started) * 1000,
                                2,
                            )
                        if frames_sent == 0 and on_first_audio is not None:
                            on_first_audio()
                        frames_sent += 1
                        await asyncio.sleep(0.10)

            if pending:
                frame = bytes(pending).ljust(phone_frame_bytes, b"\xff")
                await websocket.send_text(
                    json.dumps(
                        {
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {
                                "payload": base64.b64encode(frame).decode("ascii"),
                            },
                        }
                    )
                )
                if first_audio_wait_ms is None:
                    first_audio_wait_ms = round(
                        (time.perf_counter() - attempt_started) * 1000,
                        2,
                    )
                if frames_sent == 0 and on_first_audio is not None:
                    on_first_audio()
                frames_sent += 1

            return {
                "attempt": attempt,
                "frames_sent": frames_sent,
                "first_audio_wait_ms": first_audio_wait_ms,
                "tts_total_first_audio_wait_ms": round((attempt_started-delivery_started)*1000+first_audio_wait_ms,2) if first_audio_wait_ms is not None else None,
                "tts_failed_attempts": failed_attempts,
            }
        except asyncio.CancelledError:
            raise
        except Exception as error:
            last_error = error
            failed_attempts.append(dict(attempt=attempt, error_type=type(error).__name__, elapsed_ms=round((time.perf_counter()-attempt_started)*1000,2), frames_sent=frames_sent))
            if frames_sent or attempt == 3:
                raise
            await asyncio.sleep(float(attempt))

    raise RuntimeError(f"Aura-2 did not produce phone audio: {last_error}")


def write_call_evidence(path: Path, title: str, values: list[tuple[str, object]]):
    with path.open("a", encoding="utf-8") as evidence:
        evidence.write("=" * 72 + "\n")
        evidence.write(title + "\n")
        evidence.write("Recorded at: " + datetime.now().astimezone().isoformat() + "\n")
        for label, value in values:
            evidence.write(f"{label}: {value}\n")
        evidence.write("\n")


def save_patient_wav(path: Path, pcm16_audio: bytes):
    if not pcm16_audio:
        return
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(pcm16_audio)


class RetryingDeepgramListen:
    """Open Deepgram STT with bounded retries and evidence for every attempt."""

    def __init__(self, client, evidence_path: Path, **options):
        self.client = client
        self.evidence_path = evidence_path
        self.options = options
        self.manager = None
        self.connection = None

    async def __aenter__(self):
        last_error = None
        for attempt in range(1, 4):
            self.manager = self.client.listen.v1.connect(**self.options)
            try:
                self.connection = await self.manager.__aenter__()
                write_call_evidence(
                    self.evidence_path,
                    "DEEPGRAM STT CONNECTED",
                    [("Attempt", attempt)],
                )
                return self.connection
            except Exception as error:
                last_error = error
                self.manager = None
                write_call_evidence(
                    self.evidence_path,
                    "DEEPGRAM STT CONNECTION RETRY",
                    [
                        ("Attempt", attempt),
                        ("Error type", type(error).__name__),
                        ("Error", str(error)),
                    ],
                )
                if attempt < 3:
                    await asyncio.sleep(float(attempt))

        write_call_evidence(
            self.evidence_path,
            "DEEPGRAM STT CONNECTION FAILED",
            [
                ("Attempts", 3),
                ("Error type", type(last_error).__name__),
                ("Error", str(last_error)),
            ],
        )
        return None

    async def __aexit__(self, exc_type, exc_value, traceback):
        if self.manager is not None:
            return await self.manager.__aexit__(exc_type, exc_value, traceback)
        return False


@app.get("/twilio/health")
async def health_check():
    return {"status": "ok", "service": "clinic-twilio-phone-gateway"}


@app.get("/twilio/concurrency")
async def concurrency_status():
    """Free, privacy-safe capacity evidence; contains no caller information."""
    return await CALL_REGISTRY.snapshot()


@app.get("/twilio/preflight")
async def phone_preflight():
    """Free readiness check for configuration and Deepgram TCP connectivity."""
    checks = {
        "public_url_configured": False,
        "deepgram_key_configured": bool(os.getenv("DEEPGRAM_API_KEY")),
        "deepgram_tcp_443": False,
    }
    try:
        public_base_url()
        checks["public_url_configured"] = True
    except RuntimeError:
        pass

    writer = None
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection("api.deepgram.com", 443),
            timeout=5.0,
        )
        checks["deepgram_tcp_443"] = True
    except (TimeoutError, OSError):
        pass
    finally:
        if writer is not None:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    ready = all(checks.values())
    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "status": "ready" if ready else "not_ready",
            "checks": checks,
            "paid_phone_call_required": False,
        },
    )


@app.post("/twilio/voice")
async def incoming_twilio_call(_: Request):
    """Return TwiML that starts one bidirectional Twilio Media Stream."""
    try:
        return Response(content=to_twiml_stream_response(), media_type="application/xml")
    except RuntimeError as error:
        return Response(
            content=(
                "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
                "<Response><Say>The clinic phone service is not configured yet.</Say></Response>"
            ),
            media_type="application/xml",
            status_code=503,
            headers={"X-Clinic-Gateway-Error": str(error)},
        )


@app.websocket("/twilio/media")
async def twilio_media_stream(websocket: WebSocket):
    await websocket.accept()

    deepgram_key = os.getenv("DEEPGRAM_API_KEY")
    if not deepgram_key:
        await websocket.close(code=1011)
        return

    call_started = datetime.now().astimezone()
    session_id = call_started.strftime("%Y%m%d_%H%M%S_%f")
    evidence_path = PHONE_CONVERSATION_FOLDER / f"twilio_call_{session_id}.txt"
    patient_wav_path = PHONE_AUDIO_FOLDER / f"twilio_patient_{session_id}.wav"

    session_registered = False
    try:
        capacity = await CALL_REGISTRY.acquire(session_id)
        session_registered = True
    except CallCapacityError:
        await websocket.close(code=1013, reason="Telephone gateway at capacity")
        return

    async def release_session():
        nonlocal session_registered
        if session_registered:
            session_registered = False
            await CALL_REGISTRY.release(session_id)

    receptionist = create_receptionist_session()
    receptionist.messages.append(
        {
            "role": "system",
            "content": (
                "PHONE NAME-CAPTURE POLICY: Remember names already given in this "
                "call. Accept a single name if the caller says that is their "
                "complete name. Ask for spelling once when preparing a booking; "
                "the caller may spell first and last names in separate phrases. "
                "Keep correctly heard parts and ask only about an uncertain part. "
                "Read back the exact candidate once and accept a clear 'yes, "
                "correct' as confirmation of that read-back, not as evidence of "
                "letters that were never heard. Never bypass the deterministic "
                "name-spelling or appointment confirmation gates. If the spelling "
                "cannot be verified, offer human assistance instead of looping."
            ),
        }
    )
    speaker_guard = SpeakerChangeGuard()
    voice_identity_store = None
    if os.getenv("VOICE_IDENTITY_ENABLED", "false").strip().lower() == "true":
        voice_identity_key = os.getenv("VOICE_IDENTITY_KEY", "").strip()
        if voice_identity_key:
            voice_identity_store = VoiceIdentityStore(
                os.getenv(
                    "VOICE_IDENTITY_PROFILE_PATH",
                    "voice_identity_profiles.enc",
                ),
                voice_identity_key,
            )
    # Load the speaker model while the greeting and STT connection are being
    # established. This removes model-startup time from the first patient turn.
    speaker_warmup_task = asyncio.create_task(
        asyncio.to_thread(SpeakerChangeGuard._get_model)
    )
    multimodel_warmup_task = None
    if MULTIMODEL_VERIFIER.available():
        multimodel_warmup_task = asyncio.create_task(
            asyncio.to_thread(MULTIMODEL_VERIFIER.warmup)
        )
    deepgram = AsyncDeepgramClient(api_key=deepgram_key)
    stream_sid = None
    inbound_pcm16 = bytearray()
    speaker_audio_alignment = TranscriptSpeakerAudio()
    rate_state = None
    turn_assembler = PhoneTurnAssembler()
    prompt_context = PhonePromptContext()
    turn_lock = asyncio.Lock()
    tts_lock = asyncio.Lock()
    silence_finalize_task = None
    active_turn_task = None
    speaking_task = None
    greeting_task = None
    turn_pending = False
    queued_final_turns: list[tuple[str, bytes, str | None]] = []
    tts_playing = False
    tts_generation = 0
    audio_delivery_task = None
    pending_playback_mark = None
    tts_started_at = 0.0
    barge_in_in_progress = False
    last_transcript_at = 0.0
    closed = False

    write_call_evidence(
        evidence_path,
        "TWILIO PHONE CALL STARTED",
        [
            ("Session ID", session_id),
            ("Patient WAV", patient_wav_path),
            ("Active calls", capacity["active_calls"]),
            ("Maximum calls", capacity["maximum_calls"]),
            ("Speaker Guard V3 enabled", MULTIMODEL_VERIFIER.available()),
        ],
    )

    async def speak(text: str):
        nonlocal tts_playing, tts_started_at, tts_generation, audio_delivery_task, pending_playback_mark
        if closed or not stream_sid or not text.strip():
            return
        async with tts_lock:
            if closed:
                return
            lower_text = text.lower()
            turn_assembler.name_capture = (
                ("spell" in lower_text and "name" in lower_text)
                or ("full name" in lower_text and "?" in lower_text)
                or ("spell" in lower_text and "surname" in lower_text)
            )
            tts_generation += 1
            generation = tts_generation
            tts_playing = True
            pending_playback_mark = None
            sent_complete = False
            tts_started_at = time.monotonic()
            write_call_evidence(evidence_path, "AI RESPONSE", [("Text", text)])
            try:
                prompt_token = prompt_context.spoken_token(receptionist.session_state, text)
                audio_delivery_task = asyncio.create_task(send_aura2_to_twilio(
                    websocket, stream_sid, text,
                    on_first_audio=lambda: prompt_context.publish(len(inbound_pcm16)//2, prompt_token),
                ))
                metrics = await audio_delivery_task
                if generation == tts_generation and metrics.get("frames_sent", 0):
                    pending_playback_mark = "clinic-playback-" + str(generation)
                    await websocket.send_text(json.dumps({"event":"mark", "streamSid":stream_sid, "mark":{"name":pending_playback_mark}}))
                    sent_complete = True
                write_call_evidence(
                    evidence_path,
                    "TTS DELIVERY COMPLETE",
                    list(metrics.items()),
                )
            except asyncio.CancelledError:
                write_call_evidence(
                    evidence_path,
                    "TTS DELIVERY INTERRUPTED",
                    [("Reason", "Confirmed caller speech during playback")],
                )
                if asyncio.current_task().cancelling():
                    raise
                # Only the child audio task was cancelled by barge-in.
                # The caller-turn worker and its queue must stay alive.
                return
            except Exception as error:
                write_call_evidence(
                    evidence_path,
                    "TTS DELIVERY FAILED",
                    [
                        ("Error type", type(error).__name__),
                        ("Error", str(error)),
                    ],
                )
                return
            finally:
                if generation == tts_generation:
                    audio_delivery_task = None
                    if not sent_complete:
                        tts_playing = False
                        pending_playback_mark = None

    async def process_completed_turn(transcript: str, speaker_pcm16: bytes, reply_to_prompt=None):
        nonlocal speaking_task
        transcript = transcript.strip()
        if not transcript:
            return
        async with turn_lock:
            import re
            from copy import deepcopy
            active_prompt = prompt_context.token(receptionist.session_state)
            normalized = receptionist._normalize_confirmation_text(transcript)
            assent_only = (bool(normalized) and
                set(normalized.split()) <= {'yes','yeah','yep','yup','please','book','it','confirm','correct','okay','ok','sure','that','this','is','s','right','go','ahead','i','do','perfect','fine','great','thank','thanks','you','all','good','sounds','proceed','confirmed','the','booking','appointment','certainly','absolutely','agreed','for','me'} and
                any(word in normalized.split() for word in {'yes','yeah','yep','yup','correct','okay','ok','sure','confirm','book','perfect','fine','great','proceed','confirmed','agreed'}))
            if assent_only and reply_to_prompt is not None and reply_to_prompt != active_prompt:
                write_call_evidence(evidence_path, "STALE CONFIRMATION NOT REAPPLIED", [("Transcript", transcript)])
                return
            if assent_only and active_prompt is not None and reply_to_prompt != active_prompt:
                # The caller spoke before this readback was sent. Do not turn an
                # answer to an earlier question into permission for a new action.
                write_call_evidence(evidence_path, "CONFIRMATION PRECEDES READBACK", [("Transcript", transcript)])
                pending = receptionist.session_state.get("pending_action") or {}
                flow = receptionist.session_state.get("nameflow_v1") or {}
                if pending.get("action") == "book":
                    await speak(receptionist._booking_confirmation_text_v1(pending.get("details") or pending.get("confirmation_details") or {}))
                elif flow.get("stage") == "readback":
                    await speak(receptionist._nameflow_readback(flow))
                elif (receptionist.session_state.get('reference_flow_v2') or {}).get('simple_v6'):
                    await speak(receptionist._management_prompt_v6() or "Please tell me the change you would like to make.")
                else:
                    await speak("Please confirm the exact details I read back, or tell me what you would like to change.")
                return
            management_before = deepcopy(receptionist.session_state.get('management_commit_v6'))
            booking_before = deepcopy(receptionist.session_state.get('same_call_booking_v1'))
            pending_before = deepcopy(receptionist.session_state.get('pending_action'))
            write_call_evidence(evidence_path, "PATIENT TURN RECEIVED", [("Transcript", transcript)])
            started = time.perf_counter()
            speaker_started = time.perf_counter()
            guard_state_before_turn = speaker_guard.snapshot_state()
            speaker_assessment = await assess_phone_speaker_when_ready(
                speaker_guard, speaker_warmup_task, speaker_pcm16,
            )
            if speaker_assessment.get("possible_change"):
                pair = speaker_guard.multimodel_pair(
                    speaker_assessment,
                    speaker_pcm16,
                )
                verification = None
                try:
                    if not MULTIMODEL_VERIFIER.available():
                        verification = {"available": False, "status": "disabled_or_missing"}
                    elif multimodel_warmup_task is not None and not multimodel_warmup_task.done():
                        verification = {"available": False, "status": "worker_warming_up"}
                    elif pair is not None:
                        verification = await asyncio.wait_for(
                            asyncio.to_thread(
                                MULTIMODEL_VERIFIER.verify_pcm16,
                                pair[0],
                                pair[1],
                            ),
                            timeout=MULTIMODEL_VERIFIER.timeout + 2.5,
                        )
                except (asyncio.TimeoutError, Exception) as error:
                    verification = {
                        "available": False,
                        "status": "timeout_or_error",
                        "error_type": type(error).__name__,
                    }

                speaker_assessment = speaker_guard.resolve_multimodel(
                    guard_state_before_turn, speaker_assessment, verification,
                )
            if (
                voice_identity_store is not None
                and speaker_assessment.get("decision") == "enrolled_session_baseline"
                and speaker_guard.last_embedding is not None
            ):
                try:
                    returning_match = await asyncio.to_thread(
                        voice_identity_store.match,
                        speaker_guard.last_embedding,
                    )
                    if returning_match.get("matched"):
                        speaker_assessment["returning_voice_candidate"] = (
                            returning_match["candidate_patient_name"]
                        )
                        speaker_assessment["returning_voice_similarity"] = (
                            returning_match["similarity"]
                        )
                        speaker_assessment["identity_verified"] = False
                except Exception as error:
                    write_call_evidence(
                        evidence_path,
                        "RETURNING VOICE MATCH WARNING",
                        [
                            ("Error type", type(error).__name__),
                            ("Error", str(error)),
                        ],
                    )
            speaker_finished = time.perf_counter()
            controller_started = time.perf_counter()
            receptionist.session_state['language_delivery_v9'] = {
                'channel': 'phone', 'matches_readback': bool(active_prompt is not None and reply_to_prompt == active_prompt),
            }
            controller_task = asyncio.create_task(asyncio.to_thread(
                receptionist.process_turn, transcript, speaker_assessment,
            ))
            while not controller_task.done():
                done, _ = await asyncio.wait({controller_task}, timeout=15.0)
                if not done:
                    write_call_evidence(evidence_path, "CONTROLLER STILL PROCESSING", [("Elapsed seconds", round(time.perf_counter()-controller_started, 1))])
                    try:
                        await speak("I am still checking your request. Please give me a moment.")
                    except Exception as notice_error:
                        write_call_evidence(evidence_path, "PROCESSING NOTICE FAILED", [("Error type", type(notice_error).__name__)])
            response = await controller_task
            booking_state = receptionist.session_state
            write_call_evidence(evidence_path, "BOOKING STATE AFTER CONTROLLER", [
                ("Selected slot", booking_state.get("selected_booking_v4")),
                ("Verified name retained", bool(booking_state.get("verified_patient_name_key"))),
                ("Name stage", (booking_state.get("nameflow_v1") or {}).get("stage")),
                ("Pending transaction", booking_state.get("pending_action")),
                ("Completed in this call", booking_state.get("same_call_booking_v1")),
            ])
            controller_finished = time.perf_counter()
            logged_assessment = dict(speaker_assessment)
            if "returning_voice_candidate" in logged_assessment:
                logged_assessment["returning_voice_candidate"] = "[REDACTED]"
            write_call_evidence(
                evidence_path,
                "PATIENT TURN",
                [
                    ("Transcript", transcript),
                    ("Speaker assessment", json.dumps(logged_assessment)),
                    ("Speaker assessment latency ms", round((speaker_finished - speaker_started) * 1000, 2)),
                    ("Controller latency ms", round((controller_finished - controller_started) * 1000, 2)),
                    ("Response latency ms", round((time.perf_counter() - started) * 1000, 2)),
                ],
            )
            booking_created = (receptionist.session_state.get('same_call_booking_v1') != booking_before and
                               bool(receptionist.session_state.get('same_call_booking_v1')))
            management_completed = receptionist.session_state.get('management_commit_v6') != management_before
            if queued_final_turns and not booking_created and not management_completed:
                # Keep the prepared proposal; an undelivered prompt cannot authorise a write.
                if receptionist.messages and receptionist.messages[-1].get('role') == 'assistant' and receptionist.messages[-1].get('content') == response:
                    receptionist.messages.pop()
                write_call_evidence(evidence_path, "RESPONSE SUPERSEDED BEFORE PLAYBACK", [("Reason", "A newer completed caller turn is waiting")])
                return
            speaking_task = asyncio.create_task(speak(response))
            try:
                await speaking_task
            except asyncio.CancelledError:
                if not closed and speaking_task.cancelled():
                    # Barge-in interrupted the audio, not the caller-turn worker.
                    pass
                else:
                    raise
            finally:
                speaking_task = None

    def dispatch_final_turn():
        nonlocal active_turn_task, turn_pending
        transcript = turn_assembler.take()
        if not transcript or closed:
            return
        selected_audio, alignment = speaker_audio_alignment.take(inbound_pcm16)
        speaker_snapshot = trim_speaker_edges(selected_audio)
        reply_to_prompt = prompt_context.capture(alignment.get("first_word_start_seconds"))
        write_call_evidence(evidence_path, "SPEAKER AUDIO ALIGNMENT", list(alignment.items()))
        if turn_pending:
            import re
            presence = re.sub(r"[^a-z ]", "", transcript.lower()).strip()
            if presence in {"hello", "hello hello", "are you there", "hello are you there", "can you hear me"}:
                write_call_evidence(evidence_path, "PRESENCE CHECK DURING PROCESSING", [("Transcript", transcript)])
                return
            queued_final_turns.append((transcript, speaker_snapshot, reply_to_prompt))
            write_call_evidence(
                evidence_path,
                "PHONE TURN QUEUED",
                [("Transcript", transcript)],
            )
            return
        turn_pending = True

        async def run_one_turn():
            nonlocal turn_pending
            async def guarded_turn(turn_text, turn_audio, reply_prompt):
                try:
                    await process_completed_turn(turn_text, turn_audio, reply_prompt)
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    write_call_evidence(evidence_path, "PHONE TURN PROCESSING FAILED", [("Transcript", turn_text), ("Error type", type(error).__name__)])
                    # A DB result may be ambiguous. Never replay a failed turn
                    # or allow a subsequent yes to repeat an uncertain write.
                    receptionist.session_state['pending_action'] = None
                    receptionist.session_state['same_call_name_correction_v1'] = None
                    receptionist.session_state['reference_flow_v2'] = None
                    receptionist.session_state['phone_result_uncertain_v4'] = True
                    if not closed:
                        try:
                            await speak("I could not verify the result of that request. If you were confirming a booking or change, please check the appointment before repeating it. I can still help with general questions.")
                        except Exception as notice_error:
                            write_call_evidence(evidence_path, "TURN FAILURE NOTICE FAILED", [("Error type", type(notice_error).__name__)])
            try:
                await guarded_turn(transcript, speaker_snapshot, reply_to_prompt)
                while queued_final_turns and not closed:
                    queued_transcript, queued_audio, queued_prompt = queued_final_turns.pop(0)
                    await guarded_turn(queued_transcript, queued_audio, queued_prompt)
            finally:
                turn_pending = False

        active_turn_task = asyncio.create_task(run_one_turn())

    async def interrupt_tts_for_caller_speech(transcript: str):
        nonlocal barge_in_in_progress, tts_playing, tts_generation, pending_playback_mark
        if (
            barge_in_in_progress
            or not tts_playing
        ):
            return
        barge_in_in_progress = True
        try:
            write_call_evidence(
                evidence_path,
                "CONFIRMED PHONE BARGE-IN",
                [("Recognized caller speech", transcript)],
            )
            tts_generation += 1
            pending_playback_mark = None
            if audio_delivery_task is not None and not audio_delivery_task.done():
                audio_delivery_task.cancel()
            tts_playing = False
            try:
                await websocket.send_text(
                    json.dumps({"event": "clear", "streamSid": stream_sid})
                )
            except (WebSocketDisconnect, RuntimeError):
                pass
        finally:
            barge_in_in_progress = False

    # Twilio's start event is handled before opening STT. This guarantees that
    # the caller receives a greeting even while a transient STT connection is
    # being retried, and queued media remains available on the WebSocket.
    try:
        while not stream_sid:
            first_raw_message = await websocket.receive_text()
            first_message = json.loads(first_raw_message)
            first_event = first_message.get("event")
            if first_event == "start":
                stream_sid = (
                    first_message.get("start", {}).get("streamSid")
                    or first_message.get("streamSid")
                )
                if stream_sid:
                    greeting_task = asyncio.create_task(
                        speak("Hello, welcome to the clinic. How can I help you today?")
                    )
            elif first_event == "stop":
                await release_session()
                return
    except WebSocketDisconnect:
        await release_session()
        return

    async with RetryingDeepgramListen(
        deepgram,
        evidence_path,
        model="nova-3",
        language="en",
        keyterm=["appointment", "availability", "reschedule", "clinic", "consultation"],
        encoding="linear16",
        sample_rate=16000,
        channels=1,
        interim_results=True,
        smart_format=True,
        punctuate=True,
        endpointing=700,
        utterance_end_ms=1200,
        vad_events=True,
    ) as dg_connection:

        if dg_connection is None:
            await speak(
                "I'm sorry, the clinic's speech service is temporarily unavailable. "
                "Please try again shortly."
            )
            await websocket.close(code=1011)
            await release_session()
            return

        def on_deepgram_message(message):
            nonlocal silence_finalize_task, last_transcript_at
            message_type = str(getattr(message, "type", ""))
            if message_type == "SpeechStarted":
                turn_assembler.caller_is_speaking()
                if tts_playing:
                    asyncio.create_task(interrupt_tts_for_caller_speech("SpeechStarted event"))
                return
            if message_type == "UtteranceEnd":
                turn_assembler.schedule(dispatch_final_turn)
                return
            channel = getattr(message, "channel", None)
            alternatives = getattr(channel, "alternatives", None) if channel else None
            if not alternatives:
                return
            transcript = str(getattr(alternatives[0], "transcript", "") or "").strip()
            if transcript:
                last_transcript_at = time.monotonic()
                if silence_finalize_task is not None:
                    silence_finalize_task.cancel()
                    silence_finalize_task = None
                if not bool(getattr(message, "is_final", False)):
                    turn_assembler.caller_is_speaking()

            if transcript and tts_playing:
                asyncio.create_task(
                    interrupt_tts_for_caller_speech(transcript)
                )
            if transcript and bool(getattr(message, "is_final", False)):
                write_call_evidence(evidence_path, "STT FINAL RECEIVED", [("Transcript", transcript)])
                speaker_audio_alignment.add_final(alternatives[0])
                turn_assembler.add_final(transcript)
                # is_final describes stable words, not a finished utterance.
                # A bounded fallback covers absent endpoint events without
                # forcing the recogniser to finish an in-progress sentence.
                turn_assembler.schedule(dispatch_final_turn, minimum_pause=3.0)
            if bool(getattr(message, "speech_final", False)):
                if silence_finalize_task is not None:
                    silence_finalize_task.cancel()
                    silence_finalize_task = None
                turn_assembler.schedule(dispatch_final_turn)

        def on_deepgram_error(error):
            if not closed:
                write_call_evidence(evidence_path, "DEEPGRAM STT LISTENER ERROR", [("Error type", type(error).__name__)])

        def on_deepgram_close(message):
            if not closed:
                write_call_evidence(evidence_path, "DEEPGRAM STT LISTENER CLOSED", [])

        dg_connection.on(EventType.ERROR, on_deepgram_error)
        dg_connection.on(EventType.CLOSE, on_deepgram_close)
        dg_connection.on(EventType.MESSAGE, on_deepgram_message)
        listener_task = asyncio.create_task(dg_connection.start_listening())

        try:
            while True:
                raw_message = await receive_phone_message(websocket, listener_task)
                message = json.loads(raw_message)
                event = message.get("event")

                if event == "start":
                    stream_sid = message.get("start", {}).get("streamSid") or message.get("streamSid")
                    if stream_sid:
                        greeting_task = asyncio.create_task(
                            speak("Hello, welcome to the clinic. How can I help you today?")
                        )
                    continue

                if event == "mark":
                    mark_name = message.get("mark", {}).get("name")
                    if pending_playback_mark is not None and mark_name == pending_playback_mark:
                        pending_playback_mark = None
                        tts_playing = False
                        write_call_evidence(evidence_path, "PHONE PLAYBACK ENDED", [("Generation", tts_generation)])
                    continue

                if event == "media":
                    payload = message.get("media", {}).get("payload", "")
                    if not payload:
                        continue
                    mulaw_audio = base64.b64decode(payload)
                    pcm16_audio, rate_state = mulaw_8k_to_pcm16_16k(mulaw_audio, rate_state)
                    inbound_pcm16.extend(pcm16_audio)
                    await dg_connection.send_media(pcm16_audio)
                    continue

                if event == "stop":
                    dispatch_final_turn()
                    break

        except SpeechListenerStopped as error:
            write_call_evidence(evidence_path, "DEEPGRAM STT SESSION FAILED", [("Error type", type(error).__name__), ("Reason", str(error))])
            turn_assembler.cancel_dispatch()
            queued_final_turns.clear()
            pending_tasks = [task for task in (greeting_task, speaking_task, active_turn_task) if task is not None and not task.done()]
            for task in pending_tasks:
                task.cancel()
            if pending_tasks:
                await asyncio.gather(*pending_tasks, return_exceptions=True)
            try:
                await websocket.send_text(json.dumps({"event": "clear", "streamSid": stream_sid}))
                await asyncio.wait_for(speak("The speech connection has stopped, so I cannot hear you. Please hang up and call again shortly."), timeout=8.0)
            except Exception as fallback_error:
                write_call_evidence(evidence_path, "STT FAILURE NOTICE FAILED", [("Error type", type(fallback_error).__name__)])
            finally:
                try:
                    await websocket.close(code=1011)
                except Exception:
                    pass
        except WebSocketDisconnect:
            pass
        finally:
            turn_assembler.cancel_dispatch()
            if not closed:
                closed = True
                save_patient_wav(patient_wav_path, bytes(inbound_pcm16))
                write_call_evidence(
                    evidence_path,
                    "TWILIO PHONE CALL ENDED",
                    [("Patient WAV saved", patient_wav_path)],
                )
            listener_task.cancel()
            if silence_finalize_task is not None:
                silence_finalize_task.cancel()
            if speaking_task is not None and not speaking_task.done():
                speaking_task.cancel()
            if active_turn_task is not None and not active_turn_task.done():
                active_turn_task.cancel()
            if not speaker_warmup_task.done():
                speaker_warmup_task.cancel()
            if multimodel_warmup_task is not None and not multimodel_warmup_task.done():
                multimodel_warmup_task.cancel()
            for task in (
                listener_task,
                greeting_task,
                speaking_task,
                active_turn_task,
                speaker_warmup_task,
                multimodel_warmup_task,
            ):
                if task is not None:
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass
            capacity_after = await CALL_REGISTRY.release(session_id)
            session_registered = False
            write_call_evidence(
                evidence_path,
                "TWILIO CALL CAPACITY RELEASED",
                [("Active calls", capacity_after["active_calls"])],
            )


