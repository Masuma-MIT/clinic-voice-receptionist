Clinic Voice Receptionist - Twilio Phone Gateway

Purpose
-------
This is a separate development gateway for real inbound Twilio calls. It does
not replace or edit app_final.py, which remains the browser receptionist.

Audio path
----------
Twilio Australian number -> Twilio Media Streams -> this gateway (port 8001)
-> 8 kHz mu-law to 16 kHz PCM -> Deepgram Nova-3 -> existing Qwen controller
and speaker-change guard -> Aura-2 -> 8 kHz mu-law -> caller.

Before starting
---------------
1. Keep the existing .env file with DEEPGRAM_API_KEY and controller settings.
2. Add one non-secret line to .env after creating a public HTTPS tunnel:
   TWILIO_PUBLIC_BASE_URL=https://your-temporary-public-url
3. Configure the purchased Twilio number's incoming Voice webhook to:
   https://your-temporary-public-url/twilio/voice
   using HTTP POST.
4. Do not expose or share Twilio credentials or Deepgram keys.

Run
---
powershell -ExecutionPolicy Bypass -File .\start_twilio_phone_gateway.ps1

Evidence
--------
The gateway writes patient-side 16 kHz WAV files to phone_audio_logs and call
transcript/speaker-assessment evidence to phone_conversation_logs.

Scope
-----
This is a controlled thesis-development gateway. It must be tested with the
Twilio number before any claim that real phone-call integration is complete.
