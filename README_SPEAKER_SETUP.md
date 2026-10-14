# Speaker-model setup for supervisor review

These instructions are for a reviewing computer running Windows and Python 3.12. Run commands from the repository folder.

## Environment
Install the main application environment first. Then:
.\.venv\Scripts\python.exe -m venv .venv_speaker_lab
.\.venv_speaker_lab\Scripts\python.exe -m pip install -r requirements_speaker_lab.txt --extra-index-url https://download.pytorch.org/whl/cpu
.\.venv_speaker_lab\Scripts\python.exe -m pip check

## Hugging Face access
Use your own Hugging Face account. Accept the access conditions for pyannote/speaker-diarization-community-1.
Authenticate locally:
.\.venv_speaker_lab\Scripts\python.exe -c "from huggingface_hub import login; login()"

Never publish tokens or populated .env files.

## Local ECAPA files
Repository: speechbrain/spkrec-ecapa-voxceleb
Revision: 0f99f2d0ebe89ac095bcc5903c4dd8f72b367286

Download hyperparams.yaml, embedding_model.ckpt, mean_var_norm_emb.ckpt, classifier.ckpt and label_encoder.txt into:
speaker_models/speechbrain_ecapa_local/

In hyperparams.yaml, set pretrained_path to the absolute path of this folder using forward slashes.

Create custom.py in the same folder containing:
# ECAPA uses standard SpeechBrain classes; no custom definitions are required.

## Worker readiness
Run:
'{"action":"warmup"}' | .\.venv_speaker_lab\Scripts\python.exe -u .\speaker_guard_v3_worker.py

The worker also loads NVIDIA TitaNet and Pyannote. Initial loading requires internet access and authorised model access.

Expected output:
V3_RESULT {"available": true, "status": "ready"}

This confirms loading, not speaker-change accuracy.

## Telephone gateway
Follow README_DATABASE_SETUP.md and configure authorised credentials in the local .env file.

Run:
.\start_twilio_phone_gateway.ps1

Provide a public HTTPS tunnel to port 8001. Set TWILIO_PUBLIC_BASE_URL to its URL and restart the gateway after changing this setting.

Configure the Twilio incoming voice webhook to:
<public-url>/twilio/voice
Method: POST

Keep the gateway and tunnel running during telephone interaction. Model files, credentials and virtual environments are excluded from Git.
