# Clinic Voice Receptionist â€” Supervisor Run Guide

## Project purpose
This proof-of-concept is a real-time Generative AI voice receptionist for automated clinic scheduling and patient enquiry management.

## What is included
- FastAPI/WebSocket voice application source
- Browser microphone client and AudioWorklet code
- Receptionist controller, controlled scheduling tools and RAG/clinic-knowledge integration
- Curated implementation evidence (conversation transcripts, session WAV recordings, latency logs and database audit material)
- Project completion notes

## What is deliberately not included
- `.env` and all API keys, service credentials and private configuration
- Python virtual environment, caches, temporary logs, backups and patch-history files

## Prerequisites
- Windows 10/11 and PowerShell
- Python version used by the project (recommended: create a fresh virtual environment)
- The packages listed in the supplied requirements file, where present
- A Supabase project with the project schema/data, plus active credentials for the speech, LLM and TTS services

## Secure configuration
1. Create a local `.env` file in the project root. Do not commit, email or upload it.
2. Use the existing developer's `.env` as the authoritative list of variable names; copy only the names, never the values.
3. Supply your own authorised values for the Deepgram STT/TTS service, Qwen/Model Studio service, and Supabase connection.

## Start
1. Open PowerShell in the extracted project folder.
2. Create and activate a Python virtual environment, then install the project's dependencies.
3. Add your local `.env` values.
4. Run `start_final_receptionist.ps1` if included. If not, use the FastAPI/Uvicorn command documented in the application source.
5. Open `http://127.0.0.1:8000/test` in Chrome, allow microphone access, and begin a conversation.

## Demonstrated capabilities
- Natural clinic enquiries (services, availability, doctor/service questions)
- Multi-turn appointment booking with confirmation before database write
- Appointment lookup/change/cancellation workflows where configured
- Streaming speech recognition, AI reasoning/tool selection and natural TTS reply
- Spoken interruption/barge-in, with the AI voice stopped and the new patient request processed
- Turn-level conversation transcript, WER fields, latency metrics and session WAV evidence

## Important evaluation note
This is an integration proof-of-concept, not a clinical diagnosis system. It does not provide emergency medical advice. Production deployment would require security review, privacy governance, operational monitoring and clinical approval.
