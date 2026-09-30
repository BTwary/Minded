param(
  [string]$PythonRuntimePath = "",
  [string]$OfflineBundleRoot = (Join-Path $PSScriptRoot 'bundle')
)
$ErrorActionPreference = "Stop"

Write-Host "MindEd AA-OS offline desktop bundle build"
Write-Host "Before running this script, install packaging\windows\requirements-desktop.txt (pywebview, pyinstaller) into the build Python/venv."
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
$StaticOut = Join-Path $Root 'apps\web\out'
if (-not (Test-Path $StaticOut -PathType Container)) {
  Push-Location (Join-Path $Root 'apps\web')
  try {
    if (Test-Path 'package-lock.json' -PathType Leaf) {
      & npm ci --no-audit --ignore-scripts
    }
    & npm run build
    if ($LASTEXITCODE -ne 0) { throw "Next.js production static export failed." }
  } finally { Pop-Location }
}

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

$OnefilePath = Join-Path $OfflineBundleRoot 'standalone'
if (Test-Path $OnefilePath) { Remove-Item -Recurse -Force $OnefilePath }

& $BuildPython -m PyInstaller --noconfirm --clean --name MindEd_AAOS_v1.0.0_Windows_x64 --onefile --windowed `
  --paths $Root `
  --add-data "$StaticOut;frontend" `
  --add-data "$AppIcon;." `
  --hidden-import webview.platforms.edgechromium `
  --hidden-import webview.platforms.winforms `
  --hidden-import uvicorn.logging `
  --hidden-import uvicorn.loops `
  --hidden-import uvicorn.loops.auto `
  --hidden-import uvicorn.protocols `
  --hidden-import uvicorn.protocols.http `
  --hidden-import uvicorn.protocols.http.auto `
  --hidden-import uvicorn.protocols.websockets `
  --hidden-import uvicorn.protocols.websockets.auto `
  --hidden-import uvicorn.lifespan `
  --hidden-import uvicorn.lifespan.on `
  --collect-all webview `
  --collect-all apps `
  --collect-all packages `
  --exclude-module pytest `
  --exclude-module _pytest `
  --exclude-module IPython `
  --exclude-module notebook `
  --exclude-module jupyter `
  @IconArgs `
  --distpath $OnefilePath (Join-Path $Root 'packaging\windows\minded_entry.py')
if ($LASTEXITCODE -ne 0) { throw "PyInstaller bundle failed." }

$BuiltExe = Join-Path $OnefilePath 'MindEd_AAOS_v1.0.0_Windows_x64.exe'
$Bundle = Join-Path $OfflineBundleRoot 'MindedAAOS'
New-Item -ItemType Directory -Force -Path $Bundle | Out-Null
Copy-Item $BuiltExe (Join-Path $Bundle 'MindEd_AAOS_v1.0.0_Windows_x64.exe') -Force
Copy-Item $BuiltExe (Join-Path $Bundle 'MindedAAOS.exe') -Force
New-Item -ItemType Directory -Force -Path (Join-Path $Bundle '_internal') | Out-Null
Copy-Item $StaticOut (Join-Path $Bundle 'frontend') -Recurse -Force
$IconSrc = Join-Path $PSScriptRoot 'app.ico'
if (Test-Path $IconSrc -PathType Leaf) { Copy-Item $IconSrc (Join-Path $Bundle 'app.ico') -Force }
Set-Content -Path (Join-Path $Bundle 'OFFLINE_MODE') -Value '1' -Encoding ascii

$TargetExe = Join-Path $Bundle 'MindedAAOS.exe'
$ProductExe = Join-Path $Bundle 'MindEd_AAOS_v1.0.0_Windows_x64.exe'

# Produce SHA256SUMS.txt
$SumsPath = Join-Path $Bundle 'SHA256SUMS.txt'
$Hashes = @()
if (Test-Path $ProductExe) {
  $h = Get-FileHash -Algorithm SHA256 $ProductExe
  $Hashes += "$($h.Hash)  MindEd_AAOS_v1.0.0_Windows_x64.exe"
}
if (Test-Path $TargetExe) {
  $h = Get-FileHash -Algorithm SHA256 $TargetExe
  $Hashes += "$($h.Hash)  MindedAAOS.exe"
}
$Hashes | Out-File -FilePath $SumsPath -Encoding ascii

@{
  product = 'MindEd AA-OS'
  version = '1.0.0'
  offline = $true
  network_policy = 'loopback-only'
  ai_enabled = $false
  telemetry_enabled = $false
  storage_provider = 'local'
  executable = 'MindEd_AAOS_v1.0.0_Windows_x64.exe'
  fallback_executable = 'MindedAAOS.exe'
  frontend = 'frontend'
  runtime_dependency = 'self-contained'
  ui = 'native-window'
  console = 'hidden'
} | ConvertTo-Json -Depth 4 | Set-Content -Path (Join-Path $Bundle 'MindedAAOS.exe.manifest.json') -Encoding utf8

Write-Host "Offline desktop bundle created at: $Bundle"
Write-Host "Bundle artifacts:"
Get-ChildItem -Path $Bundle | Select-Object Name, Length

