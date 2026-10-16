$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

# Release locks are intended to be installed in a clean supported interpreter,
# not the developer's globally contaminated Python environment.
$Python = Get-Command py -ErrorAction SilentlyContinue
if ($Python) {
    & py -3.12 --version
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 is required. Install Python 3.12 x64 and rerun this script.' }
    & py -3.12 -m venv .venv-release
} else {
    $PythonExe = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'
    if (-not (Test-Path $PythonExe)) { throw 'Python 3.12 was not found. Install Python 3.12 x64 or make the py launcher available.' }
    & $PythonExe -m venv .venv-release
}
if ($LASTEXITCODE -ne 0) { throw 'Could not create .venv-release.' }
$VenvPython = Join-Path $Root '.venv-release\Scripts\python.exe'
& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'pip bootstrap failed.' }
& $VenvPython -m pip install --require-hashes -r (Join-Path $Root 'requirements-lock.txt')
if ($LASTEXITCODE -ne 0) { throw 'Locked dependency installation failed; do not remove hashes or relax pins to force a pass.' }
& $VenvPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'pip check failed.' }
& $VenvPython scripts\audit_dependency_lock.py --lock requirements-lock.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency lock audit failed.' }
$env:MINDED_CLEAN_ROOM = '1'
& $VenvPython scripts\release_gate.py
if ($LASTEXITCODE -ne 0) { throw 'Release gate failed. Inspect the generated report; this is not a release pass.' }
