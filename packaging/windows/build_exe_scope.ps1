param(
  [string]$PythonRuntimePath = "",
  [string]$OfflineBundleRoot = (Join-Path $PSScriptRoot 'bundle')
)
$ErrorActionPreference = "Stop"

Write-Host "Minded offline desktop bundle build"
Write-Host "Before running this script, install packaging\windows\requirements-desktop.txt (pywebview, pyinstaller) into the build Python/venv from your certified wheelhouse."
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

# Select authoritative build Python: explicit $PythonRuntimePath if specified, else repo .venv, else active Python
$BuildPython = if ($PythonRuntimePath -and (Test-Path $PythonRuntimePath -PathType Leaf)) {
  $PythonRuntimePath
} elseif (Test-Path (Join-Path $Root '.venv\Scripts\python.exe') -PathType Leaf) {
  Join-Path $Root '.venv\Scripts\python.exe'
} else {
  (Get-Command python.exe -ErrorAction SilentlyContinue).Source
}

if (-not $BuildPython -or -not (Test-Path $BuildPython -PathType Leaf)) {
  throw "Build Python interpreter not found. Specify -PythonRuntimePath or create .venv at repo root."
}
Write-Host "Using build Python: $BuildPython"

# Validate the frontend dependency contract before doing any build work.
& $BuildPython (Join-Path $Root 'scripts\audit_frontend_lock.py')
if ($LASTEXITCODE -ne 0) { throw "Frontend dependency contract failed. Resolve package.json/package-lock.json before release packaging." }

if (-not (Test-Path $BuildPython -PathType Leaf)) { throw "Isolated release build Python is missing." }
& $BuildPython -m pip show pyinstaller *> $null
if ($LASTEXITCODE -ne 0) { throw "PyInstaller is required in the isolated release-build environment." }

# Node/npm are build-time dependencies only. The end-user bundle contains no
# Node requirement because the Next.js application is exported to static files.
Push-Location (Join-Path $Root 'apps\web')
try {
  & npm ci --offline --no-audit --ignore-scripts
  if ($LASTEXITCODE -ne 0) { throw "Offline npm dependency installation failed; use the certified npm cache and reconciled lockfile." }
  & npm run build
  if ($LASTEXITCODE -ne 0) { throw "Next.js production static export failed." }
} finally { Pop-Location }

$StaticOut = Join-Path $Root 'apps\web\out'
if (-not (Test-Path $StaticOut -PathType Container)) { throw "Static frontend output was not produced at apps\web\out." }

if (Test-Path $OfflineBundleRoot) { Remove-Item -Recurse -Force $OfflineBundleRoot }
New-Item -ItemType Directory -Force -Path $OfflineBundleRoot | Out-Null

& $BuildPython -m pip show pywebview *> $null
if ($LASTEXITCODE -ne 0) { throw "pywebview is required in the isolated release-build environment (native desktop window)." }

# --windowed / --noconsole: no terminal window is shown, matching a normal
# installed application. --hidden-import covers pywebview's Windows backend,
# which PyInstaller's static analysis cannot always discover on its own.
$IconArgs = @()
$AppIcon = Join-Path $PSScriptRoot 'app.ico'
if (Test-Path $AppIcon -PathType Leaf) { $IconArgs = @('--icon', $AppIcon) }

& $BuildPython -m pyinstaller --noconfirm --clean --name MindedAAOS --onedir --windowed `
  --paths $Root `
  --hidden-import webview.platforms.edgechromium `
  --hidden-import webview.platforms.winforms `
  --collect-all webview `
  @IconArgs `
  --distpath $OfflineBundleRoot (Join-Path $Root 'packaging\windows\minded_entry.py')
if ($LASTEXITCODE -ne 0) { throw "PyInstaller bundle failed." }

$Bundle = Join-Path $OfflineBundleRoot 'MindedAAOS'
Copy-Item $StaticOut (Join-Path $Bundle 'frontend') -Recurse -Force
$IconSrc = Join-Path $PSScriptRoot 'app.ico'
if (Test-Path $IconSrc -PathType Leaf) { Copy-Item $IconSrc (Join-Path $Bundle 'app.ico') -Force }
Set-Content -Path (Join-Path $Bundle 'OFFLINE_MODE') -Value '1' -Encoding ascii
@{
  offline = $true
  network_policy = 'loopback-only'
  ai_enabled = $false
  telemetry_enabled = $false
  storage_provider = 'local'
  executable = 'MindedAAOS.exe'
  frontend = 'frontend'
  runtime_dependency = 'self-contained'
  ui = 'native-window'
  console = 'hidden'
} | ConvertTo-Json -Depth 4 | Set-Content -Path (Join-Path $Bundle 'MindedAAOS.exe.manifest.json') -Encoding utf8

Write-Host "Offline desktop bundle created at: $Bundle"
Write-Host "Run scripts\offline_desktop_certification.py --bundle $Bundle before release."
