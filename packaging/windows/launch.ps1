$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Python = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path $Python)) { throw "Private local runtime is not installed. Run the offline installer first." }

$env:AAOS_OFFLINE_MODE = "1"
$env:AI_ENABLED = "false"
$env:AI_PROVIDER = "none"
$env:TELEMETRY_ENABLED = "false"
$env:FEEDBACK_UPLOAD_ENABLED = "false"
$env:STORAGE_PROVIDER = "local"
$env:PIP_NO_INDEX = "1"
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"

& $Python (Join-Path $Root 'scripts\bootstrap_local.py')
if ($LASTEXITCODE -ne 0) { throw "Local workspace initialization failed." }
# Node/npm are not invoked or required in offline desktop mode;
# the prebuilt static frontend is served directly by the local API process over loopback.
Write-Host "Backend launched on http://127.0.0.1:8000 in offline mode."
