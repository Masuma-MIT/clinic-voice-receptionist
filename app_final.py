import asyncio
import json
import os
import time
import wave
from array import array
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

import torch
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from silero_vad import VADIterator, load_silero_vad
from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType

from receptionist_controller_qwen37_experiment import create_receptionist_session
from aura2_tts_streaming import stream_aura2
from conversation_audio_mixer import save_whole_conversation_wav
from speaker_change_guard import SpeakerChangeGuard


# ============================================================
# BASIC SETUP
# ============================================================

load_dotenv()

app = FastAPI(
    title="Clinic Voice Receptionist"
)

torch.set_num_threads(1)

AUDIO_FOLDER = Path("audio_logs")
AUDIO_FOLDER.mkdir(exist_ok=True)

LATENCY_FOLDER = Path("latency_logs")
LATENCY_FOLDER.mkdir(exist_ok=True)

CONVERSATION_FOLDER = Path("conversation_logs")
CONVERSATION_FOLDER.mkdir(exist_ok=True)


# ============================================================
# CONFIDENCE-AWARE STT SETTINGS
# ============================================================

HIGH_CONFIDENCE_THRESHOLD = 0.90
LOW_CONFIDENCE_THRESHOLD = 0.75

CRITICAL_WORD_THRESHOLD = 0.85
MAX_REPEAT_ATTEMPTS = 2


# ============================================================
# CONFIDENCE DECISION
# ============================================================

def evaluate_stt_confidence(
    confidence: float
) -> dict:

    if confidence >= HIGH_CONFIDENCE_THRESHOLD:

        return {
            "decision": "ACCEPT",
            "reason": "High STT confidence",
            "send_to_llm": True,
        }

    if confidence >= LOW_CONFIDENCE_THRESHOLD:

        return {
            "decision": "CLARIFY",
            "reason": "Moderate STT confidence",
            "send_to_llm": False,
        }

    return {
        "decision": "REPEAT",
        "reason": "Low STT confidence",
        "send_to_llm": False,
    }


# ============================================================
# LOW-CONFIDENCE WORD CHECK
# ============================================================

def get_low_confidence_words(
    words,
    threshold=CRITICAL_WORD_THRESHOLD
):

    low_confidence_words = []

    for word_info in words:

        word = getattr(
            word_info,
            "word",
            ""
        )

        confidence = float(
            getattr(
                word_info,
                "confidence",
                0.0
            ) or 0.0
        )

        if confidence < threshold:

            low_confidence_words.append(
                {
                    "word": word,
                    "confidence": confidence
                }
            )

    return low_confidence_words


# ============================================================
# WORD ERROR RATE (REFERENCE REQUIRED)
# ============================================================

def normalize_wer_words(
    text
):

    normalized = "".join(
        character.lower()
        if character.isalnum()
        else " "
        for character in str(
            text or ""
        )
    )

    return normalized.split()


def calculate_word_error_rate(
    reference_text,
    recognized_text
):

    reference_words = normalize_wer_words(
        reference_text
    )

    recognized_words = normalize_wer_words(
        recognized_text
    )

    if not reference_words:

        return {
            "calculated": False,
            "reason": "Reference transcript unavailable.",
            "reference_word_count": 0,
            "substitutions": None,
            "deletions": None,
            "insertions": None,
            "word_errors": None,
            "wer_percent": None,
        }


    rows = len(reference_words) + 1
    columns = len(recognized_words) + 1

    # Cell tuple:
    # (total errors, substitutions, deletions, insertions)
    matrix = [
        [None] * columns
        for _ in range(rows)
    ]

    matrix[0][0] = (0, 0, 0, 0)

    for row in range(1, rows):
        matrix[row][0] = (
            row,
            0,
            row,
            0,
        )

    for column in range(1, columns):
        matrix[0][column] = (
            column,
            0,
            0,
            column,
        )


    for row in range(1, rows):

        for column in range(1, columns):

            previous_diagonal = matrix[
                row - 1
            ][
                column - 1
            ]

            if (
                reference_words[row - 1]
                == recognized_words[column - 1]
            ):

                matrix[row][column] = (
                    previous_diagonal
                )

                continue


            substitution = (
                previous_diagonal[0] + 1,
                previous_diagonal[1] + 1,
                previous_diagonal[2],
                previous_diagonal[3],
            )

            previous_deletion = matrix[
                row - 1
            ][
                column
            ]

            deletion = (
                previous_deletion[0] + 1,
                previous_deletion[1],
                previous_deletion[2] + 1,
                previous_deletion[3],
            )

            previous_insertion = matrix[
                row
            ][
                column - 1
            ]

            insertion = (
                previous_insertion[0] + 1,
                previous_insertion[1],
                previous_insertion[2],
                previous_insertion[3] + 1,
            )

            matrix[row][column] = min(
                substitution,
                deletion,
                insertion,
                key=lambda item: (
                    item[0],
                    item[1],
                    item[2],
                    item[3],
                ),
            )


    errors, substitutions, deletions, insertions = (
        matrix[-1][-1]
    )

    return {
        "calculated": True,
        "reason": "Compared with supplied reference transcript.",
        "reference_word_count": len(reference_words),
        "recognized_word_count": len(recognized_words),
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "word_errors": errors,
        "wer_percent": round(
            errors
            / len(reference_words)
            * 100,
            2,
        ),
    }


# ============================================================
# HOME PAGE
# ============================================================

@app.get("/")
async def home():

    return {
        "status": "running",
        "system": "Clinic Voice Receptionist",
    }


# ============================================================
# BROWSER TEST PAGE
# ============================================================

@app.get(
    "/test",
    response_class=HTMLResponse
)
async def test_page():

    return r"""
<!DOCTYPE html>

<html>

<head>

    <meta charset="UTF-8">

    <title>
        Clinic Voice Receptionist
    </title>

</head>

<body>

    <h2>
        Clinic Voice Receptionist
    </h2>

    <button
        id="startButton"
        onclick="startConversation()"
    >
        Start Conversation
    </button>

    <button
        id="stopButton"
        onclick="stopConversation()"
        disabled
    >
        End Conversation
    </button>


    <p>
        <b>WER reference (optional):</b><br>

        <input
            id="werReference"
            type="text"
            size="80"
            placeholder="Type the exact test sentence before speaking"
        >

        <button
            id="werReferenceButton"
            onclick="setWerReference()"
        >
            Set for next turn
        </button>

        <span id="werReferenceStatus">
            Not set
        </span>
    </p>

    <p>
        <b>Status:</b>
        <span id="status">
            Microphone not started
        </span>
    </p>

    <p>
        <b>Silero VAD:</b>
        <span id="vadStatus">
            Waiting
        </span>
    </p>

    <p>
        <b>Nova-3 Transcript:</b>
        <span id="transcript">
            Waiting...
        </span>
    </p>

    <p>
        <b>Confidence:</b>
        <span id="confidence">
            --
        </span>
    </p>

    <p>
        <b>Result type:</b>
        <span id="resultType">
            --
        </span>
    </p>

    <p>
        <b>Confidence Decision:</b>
        <span id="confidenceDecision">
            --
        </span>
    </p>

    <p>
        <b>AI Response:</b>
        <span id="aiResponse">
            Waiting...
        </span>
    </p>

    <p>
        <b>AI Voice:</b>
        <span id="aiVoiceStatus">
            Waiting
        </span>
    </p>


<script>

let ws = null;

let audioStream = null;
let audioContext = null;
let source = null;
let processor = null;
let silentGain = null;

let isAISpeaking = false;
let completeTurnDisplayed = false;

let ttsAudioContext = null;
let ttsNextStartTime = 0;

const aura2Sources = new Set();
let ttsPlaybackGeneration = 0;

let currentTtsServerGeneration = null;
let currentLatencyTurnId = null;
let ttsPlaybackAckSent = false;

const TTS_SAMPLE_RATE = 24000;


// ============================================================
// DEEPGRAM AURA-2 PCM AUDIO PLAYER
// ============================================================

function stopAura2Playback() {

    ttsPlaybackGeneration += 1;

    currentTtsServerGeneration = null;
    currentLatencyTurnId = null;
    ttsPlaybackAckSent = false;


    for (
        const audioSource
        of aura2Sources
    ) {

        try {

            audioSource.stop();

        } catch (error) {

        }


        try {

            audioSource.disconnect();

        } catch (error) {

        }
    }


    aura2Sources.clear();


    if (ttsAudioContext) {

        ttsNextStartTime =
            ttsAudioContext.currentTime;
    }


    isAISpeaking =
        false;


    document.getElementById(
        "aiVoiceStatus"
    ).innerText =
        "Ready for patient";
}


async function playAura2PCM(arrayBuffer) {

    if (
        !arrayBuffer
        || arrayBuffer.byteLength < 2
    ) {

        return;
    }


    const playbackGeneration =
        ttsPlaybackGeneration;


    if (!ttsAudioContext) {

        ttsAudioContext =
            new AudioContext();
    }


    if (
        ttsAudioContext.state
        === "suspended"
    ) {

        await ttsAudioContext.resume();
    }


    if (
        playbackGeneration
        !== ttsPlaybackGeneration
    ) {

        return;
    }


    const pcm16 =
        new Int16Array(
            arrayBuffer
        );


    const audioBuffer =
        ttsAudioContext.createBuffer(
            1,
            pcm16.length,
            TTS_SAMPLE_RATE
        );


    const channel =
        audioBuffer.getChannelData(
            0
        );


    for (
        let index = 0;
        index < pcm16.length;
        index++
    ) {

        channel[index] =
            pcm16[index] / 32768;
    }


    const audioSource =
        ttsAudioContext.createBufferSource();


    audioSource.buffer =
        audioBuffer;


    audioSource.connect(
        ttsAudioContext.destination
    );


    const startTime =
        Math.max(
            ttsAudioContext.currentTime + 0.03,
            ttsNextStartTime
        );


    ttsNextStartTime =
        startTime
        + audioBuffer.duration;


    aura2Sources.add(
        audioSource
    );


    isAISpeaking =
        true;


    document.getElementById(
        "aiVoiceStatus"
    ).innerText =
        "AI speaking - Aura-2";


    audioSource.onended =
        function() {

            aura2Sources.delete(
                audioSource
            );


            try {

                audioSource.disconnect();

            } catch (error) {

            }


            if (
                playbackGeneration
                !== ttsPlaybackGeneration
            ) {

                return;
            }


            if (
                aura2Sources.size === 0
            ) {

                isAISpeaking =
                    false;


                ttsNextStartTime =
                    ttsAudioContext.currentTime;


                document.getElementById(
                    "aiVoiceStatus"
                ).innerText =
                    "Ready for patient";
            }
        };


    audioSource.start(
        startTime
    );


    if (
        !ttsPlaybackAckSent
        && currentLatencyTurnId !== null
        && currentTtsServerGeneration !== null
        && ws
        && ws.readyState === WebSocket.OPEN
    ) {

        ttsPlaybackAckSent = true;

        const acknowledgementGeneration =
            currentTtsServerGeneration;

        const acknowledgementTurnId =
            currentLatencyTurnId;

        const acknowledgementPlaybackGeneration =
            playbackGeneration;

        const scheduledDelayMilliseconds =
            Math.max(
                0,
                (
                    startTime
                    - ttsAudioContext.currentTime
                )
                * 1000
            );

        window.setTimeout(
            function() {

                if (
                    acknowledgementPlaybackGeneration
                    !== ttsPlaybackGeneration
                ) {

                    return;
                }

                if (
                    !ws
                    || ws.readyState
                    !== WebSocket.OPEN
                ) {

                    return;
                }

                ws.send(
                    JSON.stringify(
                        {
                            type:
                                "tts_playback_started",

                            generation:
                                acknowledgementGeneration,

                            turn_id:
                                acknowledgementTurnId,

                            scheduled_delay_ms:
                                scheduledDelayMilliseconds,

                            browser_epoch_ms:
                                Date.now()
                        }
                    )
                );
            },
            scheduledDelayMilliseconds
        );
    }
}

/*
Remove simple Markdown before speaking.
*/
function cleanTextForSpeech(text) {

    if (!text) {
        return "";
    }

    return text
        .replace(/\*\*/g, "")
        .replace(/\*/g, "")
        .replace(/#/g, "")
        .replace(/`/g, "")
        .trim();
}


/*
Speak the AI receptionist response.
*/
function speakAIResponse(text) {

    if (!("speechSynthesis" in window)) {

        document.getElementById(
            "aiVoiceStatus"
        ).innerText =
            "Browser speech synthesis unavailable";

        return;
    }


    window.speechSynthesis.cancel();


    const cleanText =
        cleanTextForSpeech(
            text
        );


    const utterance =
        new SpeechSynthesisUtterance(
            cleanText
        );


    utterance.lang =
        "en-AU";

    utterance.rate =
        1.0;

    utterance.pitch =
        1.0;

    utterance.volume =
        1.0;


    utterance.onstart =
        function() {

            isAISpeaking =
                true;

            document.getElementById(
                "aiVoiceStatus"
            ).innerText =
                "ðŸ”Š AI speaking";
        };


    utterance.onend =
        function() {

            isAISpeaking =
                false;

            document.getElementById(
                "aiVoiceStatus"
            ).innerText =
                "ðŸŽ¤ Ready for patient";
        };


    utterance.onerror =
        function(event) {

            isAISpeaking =
                false;

            document.getElementById(
                "aiVoiceStatus"
            ).innerText =
                "Voice error";

            console.error(
                "Speech synthesis error:",
                event
            );
        };


    window.speechSynthesis.speak(
        utterance
    );
}


// ============================================================
// OPTIONAL WER REFERENCE FOR NEXT PATIENT TURN
// ============================================================

function setWerReference() {

    const referenceInput =
        document.getElementById(
            "werReference"
        );

    const statusElement =
        document.getElementById(
            "werReferenceStatus"
        );

    const referenceText =
        referenceInput.value.trim();

    if (!referenceText) {

        statusElement.innerText =
            "Enter a reference sentence first";

        return;
    }


    if (
        !ws
        || ws.readyState !== WebSocket.OPEN
    ) {

        statusElement.innerText =
            "Start the conversation first";

        return;
    }


    ws.send(
        JSON.stringify(
            {
                type: "wer_reference",
                text: referenceText
            }
        )
    );

    statusElement.innerText =
        "Sending reference...";
}


// ============================================================
// START CONVERSATION
// ============================================================

async function startConversation() {

    document.getElementById(
        "startButton"
    ).disabled =
        true;

    document.getElementById(
        "stopButton"
    ).disabled =
        true;

    document.getElementById(
        "status"
    ).innerText =
        "Connecting...";

    document.getElementById(
        "vadStatus"
    ).innerText =
        "Waiting";

    document.getElementById(
        "transcript"
    ).innerText =
        "Waiting...";

    document.getElementById(
        "confidence"
    ).innerText =
        "--";

    document.getElementById(
        "resultType"
    ).innerText =
        "--";

    document.getElementById(
        "confidenceDecision"
    ).innerText =
        "--";

    document.getElementById(
        "aiResponse"
    ).innerText =
        "Waiting...";

    document.getElementById(
        "aiVoiceStatus"
    ).innerText =
        "Waiting";

    document.getElementById(
        "werReferenceStatus"
    ).innerText =
        "Not set";

    isAISpeaking =
        false;


    try {

        // ----------------------------------------------------
        // CONNECT TO FASTAPI WEBSOCKET
        // ----------------------------------------------------

        ws =
            new WebSocket(
                "ws://127.0.0.1:8000/ws"
            );


        ws.binaryType = "arraybuffer";

        await new Promise(
            (
                resolve,
                reject
            ) => {

                ws.onopen =
                    resolve;

                ws.onerror =
                    reject;
            }
        );


        // ----------------------------------------------------
        // RECEIVE FASTAPI RESULTS
        // ----------------------------------------------------

        ws.onmessage =
            async function(event) {

                if (
                    event.data
                    instanceof ArrayBuffer
                ) {

                    await playAura2PCM(
                        event.data
                    );

                    return;
                }


                const data =
                    JSON.parse(
                        event.data
                    );


                if (
                    data.type
                    === "wer_reference_status"
                ) {

                    document.getElementById(
                        "werReferenceStatus"
                    ).innerText =
                        data.accepted
                        ? "Ready for next patient turn"
                        : "Reference was not accepted";

                    if (data.accepted) {

                        document.getElementById(
                            "werReference"
                        ).value = "";
                    }
                }


                if (
                    data.type === "tts_start"
                ) {

                    stopAura2Playback();

                    currentTtsServerGeneration =
                        data.generation;

                    currentLatencyTurnId =
                        (
                            data.turn_id
                            === undefined
                        )
                        ? null
                        : data.turn_id;

                    ttsPlaybackAckSent =
                        false;

                    isAISpeaking =
                        true;

                    document.getElementById(
                        "aiVoiceStatus"
                    ).innerText =
                        "AI voice starting - Aura-2";
                }


                if (
                    data.type === "tts_stop"
                ) {

                    stopAura2Playback();

                    document.getElementById(
                        "aiVoiceStatus"
                    ).innerText =
                        "Interrupted - ready for patient";
                }


                if (
                    data.type === "tts_error"
                ) {

                    stopAura2Playback();

                    document.getElementById(
                        "aiVoiceStatus"
                    ).innerText =
                        "Aura-2 voice error";
                }


                if (
                    data.type === "tts_end"
                    && aura2Sources.size === 0
                ) {

                    isAISpeaking =
                        false;

                    document.getElementById(
                        "aiVoiceStatus"
                    ).innerText =
                        "Ready for patient";
                }


                // ---------------------------------------------
                // SILERO VAD
                // ---------------------------------------------

                if (
                    data.type === "vad"
                ) {

                    if (
                        data.event.start !== undefined
                    ) {

                        completeTurnDisplayed =
                            false;


                        document.getElementById(
                            "vadStatus"
                        ).innerText =
                            "ðŸŸ¢ SPEECH DETECTED";
                    }


                    if (
                        data.event.end !== undefined
                    ) {

                        document.getElementById(
                            "vadStatus"
                        ).innerText =
                            "SPEECH ENDED";
                    }
                }


                // ---------------------------------------------
                // DEEPGRAM TRANSCRIPT
                // ---------------------------------------------

                if (
                    data.type === "transcript"
                ) {

                    const isCompleteTurn =
                        data.complete_turn === true;


                    if (
                        isCompleteTurn
                        || !completeTurnDisplayed
                    ) {

                        document.getElementById(
                            "transcript"
                        ).innerText =
                            data.text;


                        document.getElementById(
                            "confidence"
                        ).innerText =
                            (
                                data.confidence
                                * 100
                            ).toFixed(1)
                            + "%";


                        document.getElementById(
                            "resultType"
                        ).innerText =
                            isCompleteTurn
                            ? "COMPLETE TURN"
                            : (
                                data.is_final
                                ? "FINAL"
                                : "INTERIM"
                            );
                    }


                    if (isCompleteTurn) {

                        completeTurnDisplayed =
                            true;
                    }
                }


                // ---------------------------------------------
                // CONFIDENCE DECISION
                // ---------------------------------------------

                if (
                    data.type
                    === "confidence_decision"
                ) {

                    document.getElementById(
                        "confidenceDecision"
                    ).innerText =
                        data.decision
                        + " - "
                        + data.reason;
                }


                // ---------------------------------------------
                // AI RESPONSE
                // ---------------------------------------------

                if (
                    data.type
                    === "ai_response"
                ) {

                    document.getElementById(
                        "aiResponse"
                    ).innerText =
                        data.text;

                }
            };


        ws.onclose =
            function() {

                console.log(
                    "FastAPI WebSocket closed"
                );
            };


        // ----------------------------------------------------
        // MICROPHONE
        // ----------------------------------------------------

        audioStream =
            await navigator.mediaDevices.getUserMedia(
                {

                    audio: {

                        echoCancellation:
                            true,

                        noiseSuppression:
                            true,

                        autoGainControl:
                            true,

                        channelCount:
                            1
                    }
                }
            );


        audioContext =
            new AudioContext(
                {
                    sampleRate:
                        16000
                }
            );


        await audioContext.resume();


        // ----------------------------------------------------
        // AUDIO WORKLET
        // ----------------------------------------------------

        const processorCode = `

            class PCMProcessor
                extends AudioWorkletProcessor {

                constructor() {

                    super();

                    this.samples = [];
                }


                process(inputs) {

                    const input =
                        inputs[0];


                    if (
                        input &&
                        input.length > 0 &&
                        input[0]
                    ) {


                        const channel =
                            input[0];


                        for (
                            let i = 0;
                            i < channel.length;
                            i++
                        ) {

                            this.samples.push(
                                channel[i]
                            );
                        }


                        while (
                            this.samples.length
                            >= 512
                        ) {


                            const pcm =
                                new Int16Array(
                                    512
                                );


                            for (
                                let i = 0;
                                i < 512;
                                i++
                            ) {


                                const sample =
                                    Math.max(
                                        -1,
                                        Math.min(
                                            1,
                                            this.samples[i]
                                        )
                                    );


                                pcm[i] =
                                    sample < 0
                                    ? sample * 32768
                                    : sample * 32767;
                            }


                            this.samples =
                                this.samples.slice(
                                    512
                                );


                            this.port.postMessage(
                                pcm.buffer,
                                [
                                    pcm.buffer
                                ]
                            );
                        }
                    }


                    return true;
                }
            }


            registerProcessor(
                "pcm-processor",
                PCMProcessor
            );
        `;


        const blob =
            new Blob(
                [
                    processorCode
                ],
                {
                    type:
                        "application/javascript"
                }
            );


        const workletURL =
            URL.createObjectURL(
                blob
            );


        await audioContext.audioWorklet.addModule(
            workletURL
        );


        URL.revokeObjectURL(
            workletURL
        );


        source =
            audioContext.createMediaStreamSource(
                audioStream
            );


        processor =
            new AudioWorkletNode(
                audioContext,
                "pcm-processor"
            );


        // FULL-DUPLEX SPOKEN INTERRUPTION + FAST FINALIZE
        // Keep the echo-cancelled microphone stream active while Aura-2
        // speaks so the server can recognize a real spoken interruption.
        processor.port.onmessage =
            function(event) {


                if (
                    ws &&
                    ws.readyState
                        === WebSocket.OPEN
                ) {

                    ws.send(
                        event.data
                    );
                }
            };


        silentGain =
            audioContext.createGain();


        silentGain.gain.value =
            0;


        source.connect(
            processor
        );


        processor.connect(
            silentGain
        );


        silentGain.connect(
            audioContext.destination
        );


        document.getElementById(
            "status"
        ).innerText =
            "ðŸŽ¤ Live microphone active";


        document.getElementById(
            "aiVoiceStatus"
        ).innerText =
            "ðŸŽ¤ Ready for patient";


        document.getElementById(
            "stopButton"
        ).disabled =
            false;
    }


    catch (error) {

        document.getElementById(
            "status"
        ).innerText =
            "Microphone error: "
            + error.message;


        document.getElementById(
            "startButton"
        ).disabled =
            false;


        if (ws) {

            ws.close();

            ws =
                null;
        }
    }
}


// ============================================================
// STOP CONVERSATION
// ============================================================

async function stopConversation() {

    document.getElementById(
        "stopButton"
    ).disabled =
        true;


    window.speechSynthesis.cancel();


    stopAura2Playback();

    if (ttsAudioContext) {

        ttsNextStartTime =
            ttsAudioContext.currentTime;
    }


    isAISpeaking =
        false;


    if (processor) {

        processor.disconnect();

        processor =
            null;
    }


    if (source) {

        source.disconnect();

        source =
            null;
    }


    if (silentGain) {

        silentGain.disconnect();

        silentGain =
            null;
    }


    if (audioStream) {

        audioStream
            .getTracks()
            .forEach(
                track =>
                    track.stop()
            );

        audioStream =
            null;
    }


    if (audioContext) {

        await audioContext.close();

        audioContext =
            null;
    }


    await new Promise(

        resolve =>
            setTimeout(
                resolve,
                700
            )
    );


    if (ws) {

        ws.close();

        ws =
            null;
    }


    document.getElementById(
        "status"
    ).innerText =
        "Conversation ended";


    document.getElementById(
        "aiVoiceStatus"
    ).innerText =
        "Stopped";


    document.getElementById(
        "startButton"
    ).disabled =
        false;
}


</script>

</body>

</html>
"""


# ============================================================
# LIVE AUDIO WEBSOCKET
# ============================================================

SPOKEN_INTERRUPTION_COMMANDS = {
    "stop",
    "stop please",
    "please stop",
    "stop talking",
    "wait",
    "wait please",
    "please wait",
    "hold on",
    "please hold on",
    "hang on",
    "one moment",
    "just a moment",
    "pause",
    "pause please",
    "please pause",
    "let me speak",
    "can i speak",
}

SPOKEN_INTERRUPTION_PREFIXES = (
    "stop ",
    "please stop ",
    "wait ",
    "please wait ",
    "hold on ",
    "please hold on ",
    "hang on ",
    "one moment ",
    "just a moment ",
    "pause ",
    "please pause ",
    "let me speak ",
    "can i speak ",
    "excuse me ",
    "sorry ",
)


def normalize_spoken_phrase(
    text
):

    cleaned = "".join(
        character.lower()
        if character.isalnum()
        else " "
        for character in str(
            text or ""
        )
    )

    return " ".join(
        cleaned.split()
    )


def looks_like_tts_echo(
    transcript,
    current_ai_text
):

    normalized_transcript = normalize_spoken_phrase(
        transcript
    )

    normalized_ai_text = normalize_spoken_phrase(
        current_ai_text
    )

    if (
        not normalized_transcript
        or not normalized_ai_text
    ):

        return False

    if normalized_transcript in normalized_ai_text:

        return True

    transcript_words = normalized_transcript.split()
    ai_words = normalized_ai_text.split()

    if len(transcript_words) < 3:

        return False

    minimum_window = max(
        1,
        len(transcript_words) - 2,
    )

    maximum_window = min(
        len(ai_words),
        len(transcript_words) + 2,
    )

    best_similarity = 0.0

    for window_size in range(
        minimum_window,
        maximum_window + 1,
    ):

        for start_index in range(
            0,
            len(ai_words) - window_size + 1,
        ):

            candidate = " ".join(
                ai_words[
                    start_index:
                    start_index + window_size
                ]
            )

            similarity = SequenceMatcher(
                None,
                normalized_transcript,
                candidate,
            ).ratio()

            best_similarity = max(
                best_similarity,
                similarity,
            )

    return best_similarity >= 0.84


def classify_spoken_interruption(
    transcript,
    current_ai_text="",
    is_final=False
):

    normalized = (
        normalize_spoken_phrase(
            transcript
        )
    )

    if not normalized:

        return {
            "interrupt": False,
            "command_only": False,
            "normalized": "",
            "reason": "empty",
        }

    command_only = (
        normalized
        in SPOKEN_INTERRUPTION_COMMANDS
    )

    recognized_phrase = (
        command_only
        or normalized.startswith(
            SPOKEN_INTERRUPTION_PREFIXES
        )
    )

    normalized_ai_text = (
        normalize_spoken_phrase(
            current_ai_text
        )
    )

    if (
        not command_only
        and normalized_ai_text
        and looks_like_tts_echo(
            normalized,
            normalized_ai_text,
        )
    ):

        return {
            "interrupt": False,
            "command_only": False,
            "normalized": normalized,
            "reason": "possible_tts_echo",
        }

    if recognized_phrase:

        return {
            "interrupt": True,
            "command_only": command_only,
            "normalized": normalized,
            "reason": "recognized_spoken_phrase",
        }

    if (
        bool(is_final)
        and normalized_ai_text
    ):

        return {
            "interrupt": True,
            "command_only": False,
            "normalized": normalized,
            "reason":
                "final_patient_turn_during_tts",
        }

    return {
        "interrupt": False,
        "command_only": False,
        "normalized": normalized,
        "reason":
            "awaiting_final_or_no_active_ai_text",
    }


@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket
):

    await websocket.accept()


    # One receptionist memory session
    # for this WebSocket conversation.
    receptionist = (
        create_receptionist_session()
    )

    # Per-call temporary voice comparison; no permanent voice ID is stored.
    speaker_guard = SpeakerChangeGuard()


    print(
        "Browser audio connected"
    )

    print(
        "Unified receptionist session created"
    )


    # ========================================================
    # SILERO VAD
    # ========================================================

    silero_model = (
        load_silero_vad()
    )


    vad_iterator = VADIterator(
        silero_model,
        sampling_rate=16000,
        threshold=0.5,
        min_silence_duration_ms=700,
        speech_pad_ms=100
    )


    # ========================================================
    # AUDIO LOG
    # ========================================================

    timestamp = (
        datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )
    )


    audio_path = (
        AUDIO_FOLDER
        /
        f"live_session_{timestamp}.wav"
    )

    # WHOLE-CONVERSATION AUDIO EVIDENCE
    whole_conversation_audio_path = (
        AUDIO_FOLDER
        /
        f"conversation_session_{timestamp}.wav"
    )

    whole_audio_state = {
        "microphone_start_perf": None,
        "ai_segments": {},
        "finalized": False,
    }


    print(
        "Saving WAV:",
        audio_path
    )

    print(
        "Saving whole-conversation WAV:",
        whole_conversation_audio_path
    )


    # ========================================================
    # DEEPGRAM
    # ========================================================

    deepgram = AsyncDeepgramClient(
        api_key=os.getenv(
            "DEEPGRAM_API_KEY"
        )
    )


    # Server-to-browser message queue.
    result_queue = (
        asyncio.Queue()
    )


    # Accepted STT transcript-to-receptionist queue.
    receptionist_queue = (
        asyncio.Queue()
    )


    websocket_send_lock = (
        asyncio.Lock()
    )


    tts_state = {
        "task": None,
        "generation": 0,
        "active": False,
        "text": "",
    }


    interruption_state = {
        "suppress_command_until_final": False,
    }


    conversation_state = {
        "greeted": False,
    }


    # ========================================================
    # FINAL LATENCY INSTRUMENTATION
    # ========================================================

    latency_path = (
        LATENCY_FOLDER
        /
        f"latency_session_{timestamp}.jsonl"
    )

    latency_path.write_text(
        "",
        encoding="utf-8",
    )

    latency_state = {
        "next_turn_id": 0,
        "turns": {},
        "generation_to_turn": {},
    }

    print(
        "Saving latency evidence:",
        latency_path
    )


    # ========================================================
    # CONVERSATION + WER EVIDENCE
    # ========================================================

    conversation_path = (
        CONVERSATION_FOLDER
        /
        f"conversation_session_{timestamp}.txt"
    )

    conversation_evidence_state = {
        "pending_wer_reference": None,
        "wer_results": [],
        "finalized": False,
    }


    def append_conversation_block(
        title,
        fields
    ):

        recorded_at = (
            datetime.now()
            .astimezone()
            .isoformat()
        )

        with conversation_path.open(
            "a",
            encoding="utf-8",
        ) as conversation_file:

            conversation_file.write(
                "=" * 72
                + "\n"
            )

            conversation_file.write(
                str(title)
                + "\n"
            )

            conversation_file.write(
                "Recorded at: "
                + recorded_at
                + "\n"
            )

            for label, value in fields:

                if isinstance(
                    value,
                    (
                        dict,
                        list,
                    )
                ):

                    rendered_value = json.dumps(
                        value,
                        ensure_ascii=False,
                        sort_keys=True,
                    )

                elif value is None:

                    rendered_value = (
                        "Not available"
                    )

                else:

                    rendered_value = str(
                        value
                    )

                conversation_file.write(
                    str(label)
                    + ": "
                    + rendered_value
                    + "\n"
                )

            conversation_file.write(
                "\n"
            )


    conversation_path.write_text(
        "CLINIC VOICE RECEPTIONIST CONVERSATION EVIDENCE\n"
        + "Session ID: "
        + timestamp
        + "\n"
        + "Session started: "
        + datetime.now().astimezone().isoformat()
        + "\n"
        + "Audio evidence: "
        + str(audio_path)
        + "\n"
        + "Whole-conversation audio evidence: "
        + str(whole_conversation_audio_path)
        + "\n"
        + "Latency evidence: "
        + str(latency_path)
        + "\n\n",
        encoding="utf-8",
    )

    print(
        "Saving conversation evidence:",
        conversation_path
    )


    def finalize_conversation_evidence():

        if conversation_evidence_state[
            "finalized"
        ]:

            return

        conversation_evidence_state[
            "finalized"
        ] = True


        metric_names = (
            "speech_end_to_complete_transcript_ms",
            "complete_transcript_to_llm_request_ms",
            "llm_request_to_response_ready_ms",
            "tts_request_to_first_audio_chunk_ms",
            "speech_end_to_first_audible_audio_ms",
        )

        latency_averages = {}
        latency_maximums = {}

        for metric_name in metric_names:

            values = [
                build_latency_metrics(
                    turn
                ).get(
                    metric_name
                )
                for turn in latency_state[
                    "turns"
                ].values()
            ]

            values = [
                value
                for value in values
                if value is not None
            ]

            latency_averages[
                metric_name
            ] = (
                round(
                    sum(values)
                    / len(values),
                    3,
                )
                if values
                else None
            )

            latency_maximums[
                metric_name
            ] = (
                round(
                    max(values),
                    3,
                )
                if values
                else None
            )


        wer_values = [
            result["wer_percent"]
            for result in conversation_evidence_state[
                "wer_results"
            ]
            if result.get(
                "wer_percent"
            ) is not None
        ]

        average_wer = (
            round(
                sum(wer_values)
                / len(wer_values),
                2,
            )
            if wer_values
            else None
        )

        append_conversation_block(
            "SESSION SUMMARY",
            [
                (
                    "Patient turns",
                    len(
                        latency_state[
                            "turns"
                        ]
                    ),
                ),
                (
                    "WER-tested turns",
                    len(wer_values),
                ),
                (
                    "Average WER percent",
                    average_wer,
                ),
                (
                    "Average latency milliseconds",
                    latency_averages,
                ),
                (
                    "Maximum latency milliseconds",
                    latency_maximums,
                ),
                (
                    "Session ended",
                    datetime.now()
                    .astimezone()
                    .isoformat(),
                ),
            ],
        )


    def finalize_whole_conversation_audio():

        if whole_audio_state["finalized"]:
            return

        whole_audio_state["finalized"] = True

        try:
            audio_summary = save_whole_conversation_wav(
                audio_path,
                whole_conversation_audio_path,
                whole_audio_state[
                    "microphone_start_perf"
                ],
                list(
                    whole_audio_state[
                        "ai_segments"
                    ].values()
                ),
            )

            append_conversation_block(
                "WHOLE-CONVERSATION AUDIO",
                [
                    (
                        "Path",
                        whole_conversation_audio_path,
                    ),
                    (
                        "Summary",
                        audio_summary,
                    ),
                ],
            )

        except Exception as audio_mix_error:
            print(
                "WHOLE-CONVERSATION AUDIO MIX FAILED:",
                repr(audio_mix_error)
            )

            append_conversation_block(
                "WHOLE-CONVERSATION AUDIO ERROR",
                [
                    (
                        "Error",
                        repr(audio_mix_error),
                    ),
                ],
            )


    def latency_difference_ms(
        turn,
        start_key,
        end_key
    ):

        start_value = turn.get(
            start_key
        )

        end_value = turn.get(
            end_key
        )

        if (
            start_value is None
            or end_value is None
        ):

            return None

        return round(
            (
                end_value
                - start_value
            )
            * 1000,
            3
        )


    def build_latency_metrics(turn):

        return {
            "speech_end_to_complete_transcript_ms":
                latency_difference_ms(
                    turn,
                    "speech_end_perf",
                    "complete_turn_perf",
                ),

            "last_final_segment_to_complete_transcript_ms":
                latency_difference_ms(
                    turn,
                    "last_final_segment_perf",
                    "complete_turn_perf",
                ),

            "complete_transcript_to_llm_request_ms":
                latency_difference_ms(
                    turn,
                    "complete_turn_perf",
                    "llm_request_perf",
                ),

            "llm_request_to_response_ready_ms":
                latency_difference_ms(
                    turn,
                    "llm_request_perf",
                    "response_ready_perf",
                ),

            "response_ready_to_tts_request_ms":
                latency_difference_ms(
                    turn,
                    "response_ready_perf",
                    "tts_request_perf",
                ),

            "tts_request_to_first_audio_chunk_ms":
                latency_difference_ms(
                    turn,
                    "tts_request_perf",
                    "first_audio_chunk_perf",
                ),

            "speech_end_to_first_audio_chunk_ms":
                latency_difference_ms(
                    turn,
                    "speech_end_perf",
                    "first_audio_chunk_perf",
                ),

            "first_audio_chunk_to_browser_playback_ms":
                latency_difference_ms(
                    turn,
                    "first_audio_chunk_perf",
                    "browser_playback_perf",
                ),

            "speech_end_to_first_audible_audio_ms":
                latency_difference_ms(
                    turn,
                    "speech_end_perf",
                    "browser_playback_perf",
                ),
        }


    def record_latency_event(
        turn,
        event_name
    ):

        record = {
            "event": event_name,
            "turn_id": turn["turn_id"],
            "recorded_at":
                datetime.now()
                .astimezone()
                .isoformat(),
            "transcript_characters":
                turn.get(
                    "transcript_characters",
                    0
                ),
            **build_latency_metrics(
                turn
            ),
        }

        if (
            turn.get(
                "browser_scheduled_delay_ms"
            )
            is not None
        ):

            record[
                "browser_scheduled_delay_ms"
            ] = turn[
                "browser_scheduled_delay_ms"
            ]

        with latency_path.open(
            "a",
            encoding="utf-8",
        ) as latency_file:

            latency_file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

        print(
            "LATENCY METRICS:",
            json.dumps(
                record,
                sort_keys=True,
            )
        )

        if event_name == "browser_playback_started":

            append_conversation_block(
                "TURN "
                + str(
                    turn["turn_id"]
                )
                + " LATENCY",
                [
                    (
                        "Metrics milliseconds",
                        build_latency_metrics(
                            turn
                        ),
                    ),
                    (
                        "Browser scheduled delay milliseconds",
                        turn.get(
                            "browser_scheduled_delay_ms"
                        ),
                    ),
                ],
            )

        return record


    def record_browser_playback_started(
        payload
    ):

        try:

            generation = int(
                payload.get(
                    "generation"
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            print(
                "IGNORED INVALID PLAYBACK ACK:",
                payload
            )

            return


        ai_segment = whole_audio_state[
            "ai_segments"
        ].get(generation)

        if (
            ai_segment is not None
            and ai_segment.get(
                "browser_start_perf"
            )
            is None
        ):
            ai_segment[
                "browser_start_perf"
            ] = time.perf_counter()


        try:

            turn_id = int(
                payload.get(
                    "turn_id"
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            print(
                "IGNORED INVALID PLAYBACK ACK:",
                payload
            )

            return


        expected_turn_id = (
            latency_state[
                "generation_to_turn"
            ].get(
                generation
            )
        )

        if expected_turn_id != turn_id:

            print(
                "IGNORED STALE PLAYBACK ACK:",
                payload
            )

            return


        turn = (
            latency_state[
                "turns"
            ].get(
                turn_id
            )
        )

        if (
            turn is None
            or turn.get(
                "browser_playback_perf"
            )
            is not None
        ):

            return


        turn[
            "browser_playback_perf"
        ] = time.perf_counter()

        try:

            turn[
                "browser_scheduled_delay_ms"
            ] = round(
                float(
                    payload.get(
                        "scheduled_delay_ms",
                        0.0,
                    )
                ),
                3,
            )

        except (
            TypeError,
            ValueError,
        ):

            turn[
                "browser_scheduled_delay_ms"
            ] = None


        record = record_latency_event(
            turn,
            "browser_playback_started",
        )

        result_queue.put_nowait(
            {
                "type":
                    "latency_metrics",

                "turn_id":
                    turn_id,

                "metrics":
                    record,
            }
        )


    async def send_json_to_browser(payload):

        async with websocket_send_lock:

            await websocket.send_text(
                json.dumps(payload)
            )


    async def send_bytes_to_browser(payload):

        async with websocket_send_lock:

            await websocket.send_bytes(
                payload
            )


    def ensure_ai_audio_segment(
        generation
    ):

        return whole_audio_state[
            "ai_segments"
        ].setdefault(
            generation,
            {
                "generation": generation,
                "pcm": bytearray(),
                "browser_start_perf": None,
                "fallback_start_perf": None,
                "stop_perf": None,
            },
        )


    def mark_ai_audio_stopped(
        generation
    ):

        ai_segment = whole_audio_state[
            "ai_segments"
        ].get(generation)

        if (
            ai_segment is not None
            and ai_segment.get("stop_perf") is None
        ):
            ai_segment["stop_perf"] = time.perf_counter()


    async def stream_voice_response(
        voice_text,
        generation
    ):

        completed = False

        turn_id = (
            latency_state[
                "generation_to_turn"
            ].get(
                generation
            )
        )

        latency_turn = (
            latency_state[
                "turns"
            ].get(
                turn_id
            )
            if turn_id is not None
            else None
        )

        first_audio_chunk_recorded = False

        ai_audio_segment = ensure_ai_audio_segment(
            generation
        )

        try:

            tts_state["active"] = True
            tts_state["text"] = voice_text

            await send_json_to_browser(
                {
                    "type": "tts_start",
                    "generation": generation,
                    "turn_id": turn_id,
                }
            )

            if latency_turn is not None:

                latency_turn[
                    "tts_request_perf"
                ] = time.perf_counter()


            async for audio_chunk in stream_aura2(
                voice_text
            ):

                if (
                    generation
                    != tts_state["generation"]
                ):

                    break


                if ai_audio_segment[
                    "fallback_start_perf"
                ] is None:
                    ai_audio_segment[
                        "fallback_start_perf"
                    ] = time.perf_counter() + 0.03

                ai_audio_segment["pcm"].extend(
                    audio_chunk
                )


                if (
                    latency_turn is not None
                    and not first_audio_chunk_recorded
                ):

                    first_audio_chunk_recorded = True

                    latency_turn[
                        "first_audio_chunk_perf"
                    ] = time.perf_counter()

                    record_latency_event(
                        latency_turn,
                        "server_first_audio_chunk",
                    )


                await send_bytes_to_browser(
                    audio_chunk
                )

            completed = (
                generation
                == tts_state["generation"]
            )

        except asyncio.CancelledError:

            if (
                generation
                == tts_state["generation"]
            ):

                try:

                    await send_json_to_browser(
                        {
                            "type": "tts_stop",
                            "reason": "cancelled",
                        }
                    )

                except Exception:

                    pass

            raise

        except Exception as tts_error:

            print(
                "Aura-2 streaming error:",
                repr(tts_error)
            )

            if (
                generation
                == tts_state["generation"]
            ):

                try:

                    await send_json_to_browser(
                        {
                            "type": "tts_error",
                            "message": str(
                                tts_error
                            ),
                        }
                    )

                except Exception:

                    pass

        finally:

            if (
                generation
                == tts_state["generation"]
            ):

                tts_state["active"] = False
                tts_state["text"] = ""

                if (
                    tts_state["task"]
                    is asyncio.current_task()
                ):

                    tts_state["task"] = None

                if completed:

                    try:

                        await send_json_to_browser(
                            {
                                "type": "tts_end",
                                "generation": generation,
                            }
                        )

                    except Exception:

                        pass


    def start_voice_response(
        voice_text,
        latency_turn_id=None
    ):

        previous_task = (
            tts_state["task"]
        )

        previous_generation = tts_state[
            "generation"
        ]

        mark_ai_audio_stopped(
            previous_generation
        )

        tts_state["generation"] += 1

        if (
            previous_task is not None
            and not previous_task.done()
        ):

            previous_task.cancel()

        generation = (
            tts_state["generation"]
        )

        if latency_turn_id is not None:

            latency_state[
                "generation_to_turn"
            ][
                generation
            ] = latency_turn_id

        tts_state["task"] = (
            asyncio.create_task(
                stream_voice_response(
                    voice_text,
                    generation
                )
            )
        )


    def stop_voice_for_spoken_interruption(
        reason
    ):

        current_task = (
            tts_state.get(
                "task"
            )
        )

        has_live_task = (
            current_task is not None
            and not current_task.done()
        )

        if (
            not tts_state.get(
                "active",
                False
            )
            and not has_live_task
        ):

            return False

        mark_ai_audio_stopped(
            tts_state["generation"]
        )

        tts_state["generation"] += 1
        tts_state["active"] = False
        tts_state["text"] = ""
        tts_state["task"] = None

        if has_live_task:

            current_task.cancel()

        result_queue.put_nowait(
            {
                "type": "tts_stop",
                "reason": reason,
            }
        )

        print(
            "SPOKEN INTERRUPTION:",
            reason
        )

        append_conversation_block(
            "SPOKEN INTERRUPTION",
            [
                (
                    "Reason",
                    reason,
                ),
                (
                    "AI voice stopped",
                    True,
                ),
            ],
        )

        return True


    # ========================================================
    # SEND RESULTS TO BROWSER
    # ========================================================

    async def send_results_to_browser():

        try:

            while True:

                result = (
                    await result_queue.get()
                )

                if isinstance(result, bytes):

                    await send_bytes_to_browser(
                        result
                    )

                    continue

                if (
                    isinstance(
                        result,
                        dict
                    )
                    and result.get(
                        "type"
                    )
                    == "ai_response"
                    and not conversation_state[
                        "greeted"
                    ]
                ):

                    response_text = (
                        result.get(
                            "text",
                            ""
                        )
                    ).strip()

                    if response_text:

                        response_lower = (
                            response_text.lower()
                        )

                        greeting_starts = (
                            "hello",
                            "hi ",
                            "hi,",
                            "hi.",
                            "hi!",
                            "good morning",
                            "good afternoon",
                            "good evening",
                            "welcome",
                        )

                        if not response_lower.startswith(
                            greeting_starts
                        ):

                            response_text = (
                                "Hello. "
                                + response_text
                            )

                        result = {
                            **result,
                            "text": response_text,
                        }

                        conversation_state[
                            "greeted"
                        ] = True


                if (
                    isinstance(
                        result,
                        dict
                    )
                    and result.get(
                        "type"
                    )
                    == "ai_response"
                    and not result.get(
                        "startup_greeting",
                        False
                    )
                ):

                    response_text = (
                        result.get(
                            "text",
                            ""
                        )
                    ).strip()

                    response_lower = (
                        response_text.lower()
                    )

                    existing_offer_markers = (
                        "anything else",
                        "how else can i help",
                        "what else can i help",
                        "further assistance",
                    )

                    follow_up_markers = (
                        "please provide",
                        "please tell me",
                        "please choose",
                        "please select",
                        "please confirm",
                        "let me know",
                        "would you like",
                        "could you",
                        "can you",
                        "may i",
                    )

                    farewell_markers = (
                        "goodbye",
                        "bye",
                        "have a wonderful day",
                        "have a lovely day",
                        "take care",
                    )

                    already_has_offer = any(
                        marker in response_lower
                        for marker
                        in existing_offer_markers
                    )

                    already_has_follow_up = (
                        "?" in response_text
                        or any(
                            marker in response_lower
                            for marker
                            in follow_up_markers
                        )
                    )

                    is_farewell = any(
                        marker in response_lower
                        for marker
                        in farewell_markers
                    )

                    if (
                        response_text
                        and not already_has_offer
                        and not already_has_follow_up
                        and not is_farewell
                    ):

                        response_text = (
                            response_text
                            + " Is there anything else "
                            + "I can help you with?"
                        )

                        result = {
                            **result,
                            "text": response_text,
                        }


                if (
                    isinstance(
                        result,
                        dict
                    )
                    and result.get(
                        "type"
                    )
                    == "ai_response"
                ):

                    ai_turn_id = result.get(
                        "latency_turn_id"
                    )

                    append_conversation_block(
                        (
                            "AI STARTUP GREETING"
                            if ai_turn_id is None
                            else "AI TURN "
                            + str(ai_turn_id)
                        ),
                        [
                            (
                                "Text",
                                result.get(
                                    "text",
                                    ""
                                ),
                            ),
                        ],
                    )


                await send_json_to_browser(
                    result
                )

                if (
                    isinstance(
                        result,
                        dict
                    )
                    and result.get(
                        "type"
                    )
                    == "ai_response"
                ):

                    ai_voice_text = (
                        result.get(
                            "text",
                            ""
                        )
                    ).strip()

                    if ai_voice_text:

                        start_voice_response(
                            ai_voice_text,
                            result.get(
                                "latency_turn_id"
                            )
                        )

        except asyncio.CancelledError:

            raise

        except Exception as sender_error:

            print(
                "Browser sender stopped:",
                repr(sender_error)
            )


    # ========================================================
    # PROCESS ACCEPTED TRANSCRIPTS
    # ========================================================

    async def process_receptionist_turns():

        while True:

            queued_turn = (
                await receptionist_queue.get()
            )

            if isinstance(
                queued_turn,
                dict
            ):

                transcript = (
                    queued_turn.get(
                        "transcript",
                        ""
                    )
                )

                latency_turn_id = (
                    queued_turn.get(
                        "turn_id"
                    )
                )

                speaker_audio = queued_turn.get(
                    "speaker_audio"
                )

            else:

                transcript = queued_turn
                latency_turn_id = None
                speaker_audio = None


            latency_turn = (
                latency_state[
                    "turns"
                ].get(
                    latency_turn_id
                )
                if latency_turn_id is not None
                else None
            )


            try:

                print()

                print(
                    "===================================="
                )

                print(
                    "PATIENT FINAL TRANSCRIPT"
                )

                print(
                    "===================================="
                )

                print(
                    transcript
                )


                # Run synchronous Qwen/Supabase work
                # outside the main asyncio event loop.
                if latency_turn is not None:

                    latency_turn[
                        "llm_request_perf"
                    ] = time.perf_counter()


                speaker_assessment = (
                    await asyncio.to_thread(
                        speaker_guard.assess_pcm16,
                        speaker_audio,
                    )
                )

                if speaker_assessment.get("possible_change"):
                    append_conversation_block(
                        "SPEAKER-CHANGE SAFETY CHECK",
                        [
                            ("Result", "Possible different speaker"),
                            ("Similarity score", speaker_assessment.get("similarity")),
                            ("Threshold", speaker_assessment.get("threshold")),
                            ("Action", "Re-confirm caller context before appointment action"),
                        ],
                    )

                ai_text = (
                    await asyncio.to_thread(
                        receptionist.process_turn,
                        transcript,
                        speaker_assessment,
                    )
                )


                if latency_turn is not None:

                    latency_turn[
                        "response_ready_perf"
                    ] = time.perf_counter()

                    record_latency_event(
                        latency_turn,
                        "response_ready",
                    )


                print()

                print(
                    "===================================="
                )

                print(
                    "AI RECEPTIONIST RESPONSE"
                )

                print(
                    "===================================="
                )

                print(
                    ai_text
                )


                result_queue.put_nowait(
                    {

                        "type":
                            "ai_response",

                        "text":
                            ai_text,

                        "latency_turn_id":
                            latency_turn_id
                    }
                )


            except Exception as error:

                print(
                    "RECEPTIONIST ERROR:",
                    repr(
                        error
                    )
                )


                with open(
                    "qwen37_last_error.txt",
                    "w",
                    encoding="utf-8",
                ) as error_log:
                    error_log.write(repr(error))

                result_queue.put_nowait(
                    {

                        "type":
                            "ai_response",

                        "text":
                            "I'm sorry, I had trouble processing that. "
                            "Could you please try again?"
                    }
                )


    # ========================================================
    # BACKGROUND TASKS
    # ========================================================

    browser_sender_task = (
        asyncio.create_task(
            send_results_to_browser()
        )
    )


    receptionist_task = (
        asyncio.create_task(
            process_receptionist_turns()
        )
    )


    result_queue.put_nowait(
        {
            "type": "ai_response",
            "text":
                "Hello, welcome to the clinic. "
                "How can I help you today?",
            "startup_greeting": True,
        }
    )


    deepgram_listener_task = (
        None
    )


    deepgram_keepalive_task = (
        None
    )


    try:

        # ====================================================
        # DEEPGRAM STREAMING CONNECTION
        # ====================================================

        async with deepgram.listen.v1.connect(
            model="nova-3",
            language="en",
            keyterm=[
                "consultation",
                "consultation fee",
                "consultation fees",
                "General Practice",
                "general practitioner",
                "GP",
                "appointment",
                "availability",
                "reschedule",
                "rescheduling",
                "cancellation",
                "medical certificate",
                "Cardiology",
                "Psychiatry",
                "Dietetics",
                "Nutrition",
                "Exercise Physiology",
                "Dermatology",
                "Paediatrics",
                "ENT",
                "Neurology",
                "Orthopaedics",
                "Gynaecology",
                "Endocrinology",
                "referral",
                "prescription",
                "billing",
                "Dr Olivia Wilson",
                "Dr Michael Chen",
                "Dr Emily Brown",
            ],
            encoding="linear16",
            sample_rate=16000,
            channels=1,
            interim_results=True,
            smart_format=True,
            punctuate=True,
            endpointing=300
        ) as dg_connection:


            # =================================================
            # DEEPGRAM CALLBACK
            # =================================================


            # FINAL TRANSCRIPT DEBOUNCE
            # Deepgram can emit multiple is_final segments for one utterance.
            # Collect nearby segments and submit one complete patient turn.
            final_turn_state = {
                "parts": [],
                "task": None,
                "speaking": False,
                "last_confidence": 0.0,
                "last_final_received_perf": None,
                "speech_end_perf": None,
                "speaker_audio": bytearray(),
                "completed_speaker_audio": b"",
            }

            async def flush_final_patient_turn():

                try:
                    await asyncio.sleep(0.25)
                except asyncio.CancelledError:
                    return

                if final_turn_state["speaking"]:
                    final_turn_state["task"] = None
                    return

                combined_transcript = " ".join(
                    final_turn_state["parts"]
                ).strip()

                combined_confidence = final_turn_state["last_confidence"]
                completed_speaker_audio = final_turn_state["completed_speaker_audio"]
                final_turn_state["completed_speaker_audio"] = b""
                final_turn_state["parts"].clear()
                final_turn_state["task"] = None

                if combined_transcript:

                    latency_state[
                        "next_turn_id"
                    ] += 1

                    latency_turn_id = (
                        latency_state[
                            "next_turn_id"
                        ]
                    )

                    latency_turn = {
                        "turn_id":
                            latency_turn_id,

                        "transcript_characters":
                            len(
                                combined_transcript
                            ),

                        "speech_end_perf":
                            final_turn_state.get(
                                "speech_end_perf"
                            ),

                        "last_final_segment_perf":
                            final_turn_state.get(
                                "last_final_received_perf"
                            ),

                        "complete_turn_perf":
                            time.perf_counter(),
                    }

                    latency_state[
                        "turns"
                    ][
                        latency_turn_id
                    ] = latency_turn

                    record_latency_event(
                        latency_turn,
                        "complete_turn_ready",
                    )

                    wer_reference = (
                        conversation_evidence_state[
                            "pending_wer_reference"
                        ]
                    )

                    conversation_evidence_state[
                        "pending_wer_reference"
                    ] = None

                    wer_result = (
                        calculate_word_error_rate(
                            wer_reference,
                            combined_transcript,
                        )
                    )

                    if wer_result[
                        "calculated"
                    ]:

                        conversation_evidence_state[
                            "wer_results"
                        ].append(
                            wer_result
                        )

                    append_conversation_block(
                        "PATIENT TURN "
                        + str(
                            latency_turn_id
                        ),
                        [
                            (
                                "Recognized transcript",
                                combined_transcript,
                            ),
                            (
                                "STT confidence percent",
                                round(
                                    combined_confidence
                                    * 100,
                                    2,
                                ),
                            ),
                            (
                                "WER reference transcript",
                                wer_reference,
                            ),
                            (
                                "WER calculated",
                                wer_result[
                                    "calculated"
                                ],
                            ),
                            (
                                "WER percent",
                                wer_result[
                                    "wer_percent"
                                ],
                            ),
                            (
                                "WER substitutions",
                                wer_result[
                                    "substitutions"
                                ],
                            ),
                            (
                                "WER deletions",
                                wer_result[
                                    "deletions"
                                ],
                            ),
                            (
                                "WER insertions",
                                wer_result[
                                    "insertions"
                                ],
                            ),
                        ],
                    )


                    final_turn_state[
                        "speech_end_perf"
                    ] = None

                    final_turn_state[
                        "last_final_received_perf"
                    ] = None


                    print(
                        "DEBOUNCED PATIENT TURN:",
                        combined_transcript,
                    )
                    result_queue.put_nowait(
                        {
                            "type": "transcript",
                            "text": combined_transcript,
                            "confidence": combined_confidence,
                            "is_final": True,
                            "complete_turn": True,
                            "latency_turn_id":
                                latency_turn_id,
                        }
                    )
                    receptionist_queue.put_nowait(
                        {
                            "transcript":
                                combined_transcript,

                            "turn_id":
                                latency_turn_id,

                            "speaker_audio":
                                completed_speaker_audio,
                        }
                    )

            def on_deepgram_message(
                message
            ):

                channel = getattr(
                    message,
                    "channel",
                    None
                )


                if not channel:

                    return


                alternatives = getattr(
                    channel,
                    "alternatives",
                    None
                )


                if not alternatives:

                    return


                alternative = (
                    alternatives[0]
                )


                transcript = (
                    getattr(
                        alternative,
                        "transcript",
                        ""
                    )
                    or ""
                )


                if not transcript.strip():

                    return


                confidence = float(
                    getattr(
                        alternative,
                        "confidence",
                        0.0
                    )
                    or 0.0
                )


                is_final = bool(
                    getattr(
                        message,
                        "is_final",
                        False
                    )
                )


                interruption_result = (
                    classify_spoken_interruption(
                        transcript,
                        tts_state.get(
                            "text",
                            ""
                        ),
                        is_final=is_final
                    )
                )

                normalized_interruption = (
                    interruption_result[
                        "normalized"
                    ]
                )

                if (
                    tts_state.get(
                        "active",
                        False
                    )
                    and interruption_result[
                        "reason"
                    ]
                    == "possible_tts_echo"
                ):

                    print(
                        "TTS ECHO SUPPRESSED:",
                        transcript
                    )

                    return

                if (
                    interruption_state[
                        "suppress_command_until_final"
                    ]
                    and is_final
                ):

                    interruption_state[
                        "suppress_command_until_final"
                    ] = False

                    if (
                        normalized_interruption
                        in SPOKEN_INTERRUPTION_COMMANDS
                    ):

                        result_queue.put_nowait(
                            {
                                "type": "transcript",
                                "text": transcript,
                                "confidence": confidence,
                                "is_final": True,
                                "interruption_command": True,
                            }
                        )

                        print(
                            "SPOKEN INTERRUPTION COMMAND "
                            "COMPLETED:",
                            transcript
                        )

                        return

                if (
                    confidence >= 0.70
                    and interruption_result[
                        "interrupt"
                    ]
                    and tts_state.get(
                        "active",
                        False
                    )
                ):

                    voice_stopped = (
                        stop_voice_for_spoken_interruption(
                            "spoken barge-in: "
                            + interruption_result[
                                "reason"
                            ]
                            + ": "
                            + normalized_interruption
                        )
                    )

                    if (
                        voice_stopped
                        and interruption_result[
                            "command_only"
                        ]
                    ):

                        interruption_state[
                            "suppress_command_until_final"
                        ] = not is_final

                        result_queue.put_nowait(
                            {
                                "type": "transcript",
                                "text": transcript,
                                "confidence": confidence,
                                "is_final": is_final,
                                "interruption_command": True,
                            }
                        )

                        return


                print(
                    "NOVA-3:",
                    transcript,
                    "| Confidence:",
                    confidence,
                    "| Final:",
                    is_final
                )


                result_queue.put_nowait(
                    {

                        "type":
                            "transcript",

                        "text":
                            transcript,

                        "confidence":
                            confidence,

                        "is_final":
                            is_final
                    }
                )


                # =============================================
                # FINAL TRANSCRIPT
                # =============================================

                if is_final:

                    final_turn_state[
                        "last_final_received_perf"
                    ] = time.perf_counter()

                    words = getattr(
                        alternative,
                        "words",
                        []
                    ) or []


                    print(
                        "WORD CONFIDENCE:"
                    )


                    for word_info in words:

                        print(

                            getattr(
                                word_info,
                                "word",
                                ""
                            ),

                            "|",

                            round(

                                float(

                                    getattr(
                                        word_info,
                                        "confidence",
                                        0.0
                                    )
                                    or 0.0
                                ),

                                3
                            )
                        )


                    low_confidence_words = (
                        get_low_confidence_words(
                            words
                        )
                    )


                    if low_confidence_words:

                        print(
                            "LOW-CONFIDENCE WORDS:"
                        )


                        for item in low_confidence_words:

                            print(
                                item["word"],
                                "|",
                                round(
                                    item["confidence"],
                                    3
                                )
                            )

                    else:

                        print(
                            "LOW-CONFIDENCE WORDS: NONE"
                        )


                    confidence_decision = (
                        evaluate_stt_confidence(
                            confidence
                        )
                    )


                    print(
                        "CONFIDENCE DECISION:",
                        confidence_decision[
                            "decision"
                        ],
                        "| Reason:",
                        confidence_decision[
                            "reason"
                        ],
                        "| Send to LLM:",
                        confidence_decision[
                            "send_to_llm"
                        ]
                    )


                    result_queue.put_nowait(
                        {

                            "type":
                                "confidence_decision",

                            "decision":
                                confidence_decision[
                                    "decision"
                                ],

                            "reason":
                                confidence_decision[
                                    "reason"
                                ],

                            "send_to_llm":
                                confidence_decision[
                                    "send_to_llm"
                                ]
                        }
                    )


                    # =========================================
                    # ACCEPTED TRANSCRIPT TO RECEPTIONIST
                    # =========================================

                    if (
                        confidence_decision[
                            "send_to_llm"
                        ]
                    ):

                        final_segment = transcript.strip()
                        final_parts = final_turn_state["parts"]

                        if final_segment:

                            if (
                                final_parts
                                and final_segment.startswith(
                                    final_parts[-1]
                                )
                            ):
                                final_parts[-1] = final_segment

                            elif (
                                not final_parts
                                or final_parts[-1] != final_segment
                            ):
                                final_parts.append(final_segment)

                        final_turn_state["last_confidence"] = confidence

                        if final_turn_state["speaking"]:
                            return

                        previous_task = final_turn_state["task"]

                        if (
                            previous_task is not None
                            and not previous_task.done()
                        ):
                            previous_task.cancel()

                        final_turn_state["task"] = (
                            asyncio.create_task(
                                flush_final_patient_turn()
                            )
                        )


                    # =========================================
                    # MODERATE CONFIDENCE
                    # =========================================

                    elif (
                        confidence_decision[
                            "decision"
                        ]
                        == "CLARIFY"
                    ):

                        result_queue.put_nowait(
                            {

                                "type":
                                    "ai_response",

                                "text":
                                    "I didn't hear that clearly enough. "
                                    "Could you please say that again?"
                            }
                        )


                    # =========================================
                    # LOW CONFIDENCE
                    # =========================================

                    elif (
                        confidence_decision[
                            "decision"
                        ]
                        == "REPEAT"
                    ):

                        result_queue.put_nowait(
                            {

                                "type":
                                    "ai_response",

                                "text":
                                    "Sorry, I couldn't hear that clearly. "
                                    "Could you please repeat it?"
                            }
                        )


            # =================================================
            # DEEPGRAM ERROR
            # =================================================

            def on_deepgram_error(
                error
            ):

                print(
                    "Deepgram error:",
                    error
                )


            dg_connection.on(
                EventType.MESSAGE,
                on_deepgram_message
            )


            dg_connection.on(
                EventType.ERROR,
                on_deepgram_error
            )


            deepgram_listener_task = (
                asyncio.create_task(
                    dg_connection.start_listening()
                )
            )


            async def deepgram_keepalive_loop():

                try:

                    while True:

                        await asyncio.sleep(
                            3
                        )


                        await dg_connection.send_keep_alive()


                except asyncio.CancelledError:

                    raise


                except Exception as error:

                    print(
                        "Deepgram KeepAlive stopped:",
                        repr(
                            error
                        )
                    )


            deepgram_keepalive_task = (
                asyncio.create_task(
                    deepgram_keepalive_loop()
                )
            )


            print(
                "Deepgram Nova-3 connected"
            )


            print(
                "Silero VAD ready"
            )


            # =================================================
            # SAVE + PROCESS LIVE AUDIO
            # =================================================

            with wave.open(
                str(
                    audio_path
                ),
                "wb"
            ) as wav_file:


                wav_file.setnchannels(
                    1
                )


                wav_file.setsampwidth(
                    2
                )


                wav_file.setframerate(
                    16000
                )


                while True:

                    incoming_message = (
                        await websocket.receive()
                    )

                    if (
                        incoming_message.get(
                            "type"
                        )
                        == "websocket.disconnect"
                    ):

                        raise WebSocketDisconnect(
                            code=incoming_message.get(
                                "code",
                                1000
                            )
                        )


                    incoming_text = (
                        incoming_message.get(
                            "text"
                        )
                    )

                    if incoming_text is not None:

                        try:

                            browser_message = (
                                json.loads(
                                    incoming_text
                                )
                            )

                        except Exception:

                            browser_message = None


                        if isinstance(
                            browser_message,
                            dict
                        ):

                            browser_message_type = (
                                browser_message.get(
                                    "type"
                                )
                            )

                            if (
                                browser_message_type
                                == "tts_playback_started"
                            ):

                                record_browser_playback_started(
                                    browser_message
                                )

                            elif (
                                browser_message_type
                                == "wer_reference"
                            ):

                                reference_text = str(
                                    browser_message.get(
                                        "text",
                                        ""
                                    )
                                ).strip()

                                accepted = bool(
                                    reference_text
                                )

                                conversation_evidence_state[
                                    "pending_wer_reference"
                                ] = (
                                    reference_text
                                    if accepted
                                    else None
                                )

                                result_queue.put_nowait(
                                    {
                                        "type":
                                            "wer_reference_status",
                                        "accepted":
                                            accepted,
                                    }
                                )

                                print(
                                    "WER REFERENCE SET FOR NEXT TURN:",
                                    accepted
                                )

                        continue


                    audio_bytes = (
                        incoming_message.get(
                            "bytes"
                        )
                    )

                    if not audio_bytes:

                        continue


                    # -----------------------------------------
                    # SAVE AUDIO
                    # -----------------------------------------

                    if whole_audio_state[
                        "microphone_start_perf"
                    ] is None:
                        whole_audio_state[
                            "microphone_start_perf"
                        ] = time.perf_counter()

                    wav_file.writeframesraw(
                        audio_bytes
                    )


                    # -----------------------------------------
                    # SILERO VAD
                    # -----------------------------------------

                    pcm_values = array(
                        "h",
                        audio_bytes
                    )


                    if len(
                        pcm_values
                    ) == 512:

                        audio_tensor = (
                            torch.tensor(
                                pcm_values,
                                dtype=torch.float32
                            )
                            / 32768.0
                        )


                        vad_event = vad_iterator(
                            audio_tensor,
                            return_seconds=True
                        )


                        if vad_event:

                            print(
                                "SILERO VAD:",
                                vad_event
                            )

                            if "start" in vad_event:
                                final_turn_state["speaker_audio"].clear()
                                final_turn_state["speaking"] = True
                                final_turn_state["speech_end_perf"] = None
                                final_turn_state["last_final_received_perf"] = None
                                pending_task = final_turn_state["task"]
                                if (
                                    pending_task is not None
                                    and not pending_task.done()
                                ):
                                    pending_task.cancel()
                                final_turn_state["task"] = None

                            if "end" in vad_event:
                                final_turn_state["speaker_audio"].extend(audio_bytes)
                                final_turn_state["completed_speaker_audio"] = bytes(
                                    final_turn_state["speaker_audio"]
                                )
                                final_turn_state["speaking"] = False
                                final_turn_state["speech_end_perf"] = time.perf_counter()

                                try:
                                    await dg_connection.send_finalize()
                                    print(
                                        "NOVA-3 FINALIZE REQUESTED AT SPEECH END"
                                    )
                                except Exception as finalize_error:
                                    print(
                                        "NOVA-3 FINALIZE REQUEST FAILED:",
                                        repr(finalize_error)
                                    )

                                pending_task = final_turn_state["task"]
                                if (
                                    pending_task is not None
                                    and not pending_task.done()
                                ):
                                    pending_task.cancel()

                                if final_turn_state["parts"]:
                                    final_turn_state["task"] = (
                                        asyncio.create_task(
                                            flush_final_patient_turn()
                                        )
                                    )


                            result_queue.put_nowait(
                                {

                                    "type":
                                        "vad",

                                    "event":
                                        vad_event
                                }
                            )


                    if final_turn_state["speaking"]:
                        final_turn_state["speaker_audio"].extend(audio_bytes)

                    # -----------------------------------------
                    # SEND AUDIO TO DEEPGRAM
                    # -----------------------------------------

                    await dg_connection.send_media(
                        audio_bytes
                    )


    # ========================================================
    # NORMAL DISCONNECT
    # ========================================================

    except WebSocketDisconnect:

        print(
            "Conversation ended normally"
        )


    # ========================================================
    # SERVER ERROR
    # ========================================================

    except Exception as error:

        print(
            "SERVER ERROR:",
            repr(
                error
            )
        )


    # ========================================================
    # CLEANUP
    # ========================================================

    finally:

        vad_iterator.reset_states()


        if (
            deepgram_keepalive_task
            is not None
        ):

            deepgram_keepalive_task.cancel()


            try:

                await deepgram_keepalive_task

            except asyncio.CancelledError:

                pass

            except Exception:

                pass


        if (
            deepgram_listener_task
            is not None
        ):

            deepgram_listener_task.cancel()


            try:

                await deepgram_listener_task

            except asyncio.CancelledError:

                pass

            except Exception:

                pass


        current_tts_task = (
            tts_state.get(
                "task"
            )
        )

        mark_ai_audio_stopped(
            tts_state["generation"]
        )

        tts_state["generation"] += 1
        tts_state["active"] = False
        tts_state["text"] = ""

        if (
            current_tts_task is not None
            and not current_tts_task.done()
        ):

            current_tts_task.cancel()

            try:

                await current_tts_task

            except asyncio.CancelledError:

                pass

            except Exception:

                pass


        receptionist_task.cancel()


        try:

            await receptionist_task

        except asyncio.CancelledError:

            pass


        browser_sender_task.cancel()


        try:

            await browser_sender_task

        except asyncio.CancelledError:

            pass


        finalize_whole_conversation_audio()


        finalize_conversation_evidence()


        print(
            "Conversation evidence saved:",
            conversation_path
        )


        print(
            "Audio saved:",
            audio_path
        )


        print(
            "Whole conversation audio saved:",
            whole_conversation_audio_path
        )

