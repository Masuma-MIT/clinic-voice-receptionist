# Project Completion Notes

## Title
Generative AI Voice Assistant for Automated Clinic Scheduling and Patient Inquiry Management

## Aim
To design and evaluate an integrated real-time voice receptionist that can understand patient speech, answer clinic enquiries, manage appointment workflows through controlled database tools, and return natural spoken responses.

## Implemented pipeline
Patient microphone â†’ browser audio processing (noise suppression, echo cancellation and gain control) â†’ AudioWorklet (16 kHz mono PCM) â†’ WebSocket/FastAPI â†’ Silero neural VAD â†’ Deepgram Nova-3 streaming STT â†’ confidence-aware handling â†’ Qwen LLM receptionist controller/RAG â†’ controlled Supabase scheduling functions â†’ Deepgram Aura-2 TTS â†’ browser playback.

## Technologies
- FastAPI and Uvicorn for the real-time Python backend
- WebSocket, Chrome microphone capture and AudioWorklet for streaming audio
- Silero VAD for speech activity detection
- Deepgram Nova-3 streaming speech-to-text and Aura-2 text-to-speech
- Qwen hosted through Model Studio for generative conversation and tool selection
- Supabase PostgreSQL for services, doctors, slots and appointments
- Controlled controller functions for database actions and booking confirmation gating

## Successful implementation evidence
- Live multi-turn conversations were completed and stored as per-session transcript logs and whole-conversation WAV files.
- The system answered clinic-service and next-day GP availability questions, then offered specific doctor/date/time slots.
- Spoken interruption was recorded: the AI voice stopped and processed the patient's new question.
- Database safety was demonstrated: no appointment write occurred before explicit booking confirmation.
- A confirmed booking test was completed and the final booking was retained as requested.
- A whole-conversation recording was successfully generated at 24 kHz; the evidence log reports included AI segments and total session duration.

## Measured evidence example
In the saved 03 September 2026 session, speech-end to completed transcript was approximately 231â€“245 ms. TTS first audio chunk was approximately 1.96â€“1.99 seconds after the TTS request. Overall speech-end to first audible audio varied by turn (approximately 10.20â€“17.06 seconds), mainly reflecting LLM response time. One controlled WER reference comparison recorded 16.67% WER (one substitution); other turns without a supplied reference were correctly marked as not calculated.

## Evidence package contents
The accompanying evidence folder contains the latest curated conversation logs, whole-conversation WAV files, latency JSONL files, database/slot audit material when available, and this implementation documentation. The evidence is preserved without re-running paid services.

## Limitations
- Evaluation evidence is from a proof-of-concept test set rather than a large clinical field trial.
- WER requires a manually supplied reference transcript; it is not available for every free-form turn.
- End-to-end latency can increase when the external LLM service is slow.
- The system is English-focused in the demonstrated live implementation.
- It is not integrated with a production EHR/FHIR workflow and is not a substitute for clinical judgement or emergency services.

## Current completion status
The implementation is complete for demonstration and research evidence purposes. No further feature work is required before supervisor presentation; the priority is secure handover and clear reporting.
