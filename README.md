# Generative AI Voice Assistant for Automated Appointment Booking and General Enquiries Management

## Purpose
This research prototype integrates pretrained technologies for spoken clinic administrative enquiries and confirmation-controlled appointment management. Development involved repeated testing and application refinement, without training or fine-tuning the underlying models.

## Architecture
- Browser: microphone / AudioWorklet -> FastAPI WebSocket -> Silero VAD.
- Telephone: Twilio Voice / Media Streams -> separate FastAPI gateway.
- Transcription: Deepgram Nova-3.
- Dialogue and tool orchestration: Qwen 3.7 Plus.
- Structured administrative knowledge retrieval and appointment tools: Supabase PostgreSQL.
- Speech generation: Deepgram Aura-2, Asteria voice.
- Speaker assessment: SpeechBrain ECAPA; telephone verification additionally uses NVIDIA TitaNet and Pyannote.
- Controls: name verification, transaction confirmation, speaker clarification, interruption handling and queued-turn recovery.

## Source entry points
- app_final.py: browser application, port 8000.
- twilio_phone_gateway.py: telephone application, port 8001.
- receptionist_controller_qwen37_experiment.py: shared dialogue and appointment controller.
- speaker_guard_v3_worker.py: additional telephone speaker-verification worker.
The controller filename is retained to preserve the submitted implementation.

## Environment
The implementation used Python 3.12.6 on Windows 11.
requirements_main.txt records the main environment.
requirements_speaker_lab.txt records the separate speaker environment.
These files record installed versions; installation and compatibility must be checked on the reviewing computer.

## Configuration
Copy .env.example to .env locally and provide authorised Deepgram, Qwen and Supabase settings.
Credentials, recordings, private profiles, model caches and virtual environments are excluded from Git.
A matching Supabase schema, appointment functions and demonstration data are required; source code alone does not provision the database.

## Browser demonstration
Create a .venv environment and install requirements_main.txt.
Configure the database and local .env.
Run .\start_final_receptionist.ps1.
Open http://127.0.0.1:8000/test in Chrome and allow microphone access.

## Telephone demonstration
Run .\start_twilio_phone_gateway.ps1 in a separate PowerShell window.
Provide a public HTTPS tunnel to port 8001.
Set TWILIO_PUBLIC_BASE_URL to that tunnel URL.
Configure the Twilio incoming voice webhook to <public-url>/twilio/voice using POST.
Telephone testing requires an authorised Twilio account and telephone number.

## Speaker-model setup
Additional telephone verification requires .venv_speaker_lab, its dependency list, and access to:
- speechbrain/spkrec-ecapa-voxceleb
- nvidia/speakerverification_en_titanet_large
- pyannote/speaker-diarization-community-1
The worker expects local ECAPA assets in speaker_models/speechbrain_ecapa_local.
Pyannote requires authorised Hugging Face access and acceptance of its model conditions.
Model weights and account tokens are not included in this repository.

## Scope and evidence
The system supports administrative enquiries and appointment workflows, not personalised diagnosis or treatment advice.
Speaker clarification is not patient authentication.
The thesis reports browser, telephone, automated, acoustic and regression evidence with their respective evaluation boundaries.
This source package does not currently include participant recordings or test reports.

## Review status
The 15 submitted Python modules passed static syntax checks during packaging.
This does not establish successful installation or a live demonstration on a different computer.

## Access and execution

This repository contains the browser and Twilio implementations, demonstration database scripts, and setup instructions. Credentials are excluded from the public repository. Running the system requires authorised API credentials and the configuration described in the setup guides. Assistance with execution can be arranged if further verification is required.

- [Professor access guide — Word document](docs/Professor_AI_Receptionist_Access_Guide.docx)
- [Demonstration database setup](README_DATABASE_SETUP.md)
- [Speaker-model setup](README_SPEAKER_SETUP.md)