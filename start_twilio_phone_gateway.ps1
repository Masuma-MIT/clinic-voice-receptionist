# SPEAKER_VERIFICATION_DEADLINE_V1_20261004
Set-Location $PSScriptRoot

Write-Host "Starting separate Twilio Phone Gateway on port 8001"
Write-Host "This does not replace the browser receptionist on port 8000."
Write-Host "Press Ctrl+C to stop only the phone gateway."

# Final research configuration. The heavy multi-model guard runs in the
# isolated speaker-lab environment and fails safely without stopping calls.
if (-not $env:SPEAKER_GUARD_V3_ENABLED) { $env:SPEAKER_GUARD_V3_ENABLED = "true" }
# Replace the previous standard deadline even when this shell retained it.
if (-not $env:SPEAKER_GUARD_V3_TIMEOUT_SECONDS -or $env:SPEAKER_GUARD_V3_TIMEOUT_SECONDS -eq "15") { $env:SPEAKER_GUARD_V3_TIMEOUT_SECONDS = "35" }
Write-Host "Speaker Guard V3 verification allowance: $env:SPEAKER_GUARD_V3_TIMEOUT_SECONDS seconds"
if (-not $env:MAX_CONCURRENT_PHONE_CALLS) { $env:MAX_CONCURRENT_PHONE_CALLS = "4" }
Write-Host "Speaker Guard V3 feature flag: $env:SPEAKER_GUARD_V3_ENABLED"
Write-Host "Configured simultaneous-call capacity: $env:MAX_CONCURRENT_PHONE_CALLS"

& .\.venv\Scripts\python.exe -m uvicorn twilio_phone_gateway:app --host 0.0.0.0 --port 8001
