$ErrorActionPreference = "Stop"

$ProjectRoot = $PSScriptRoot
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $PythonExe)) {
    throw "Virtual-environment Python was not found: $PythonExe"
}

$Listeners = @(
    Get-NetTCPConnection `
        -LocalPort 8000 `
        -State Listen `
        -ErrorAction SilentlyContinue
)

if ($Listeners.Count -gt 0) {
    $ProcessIds = (
        $Listeners.OwningProcess |
        Sort-Object -Unique
    ) -join ", "

    throw "Port 8000 is already in use by process: $ProcessIds"
}

Set-Location $ProjectRoot

Write-Host "Starting canonical Clinic Voice Receptionist"
Write-Host "Test page: http://127.0.0.1:8000/test"
Write-Host "Press Ctrl+C to stop the server"

& $PythonExe `
    -m uvicorn `
    app_final:app `
    --app-dir $ProjectRoot `
    --host 127.0.0.1 `
    --port 8000
