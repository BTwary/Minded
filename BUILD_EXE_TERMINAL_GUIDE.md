# MindEd AA-OS — Complete Windows EXE Build Guide (Terminal Commands)

This guide provides the exact terminal commands required to build the standalone Windows offline desktop executable (`MindedAAOS.exe` / `MindEd_AAOS_v1.0.0_Windows_x64.exe`) from scratch.

---

## Prerequisites (Build Machine)

Ensure the following tools are installed on your Windows 10/11 x64 system:
- **Python 3.12 64-bit** with Python Launcher (`py.exe`)
- **Node.js LTS** (v18+ or v20+) with `npm`
- **Microsoft Edge WebView2 Runtime** (pre-installed on Windows 10/11)

---

## Quick Method: Automated 1-Script Build

Open PowerShell in `C:\Users\bhask\Downloads\MindEd` and run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\BUILD_AND_TEST_EXE.ps1 -SkipLaunch
```

> **Note**: To also launch the GUI immediately after building for a smoke test, run `.\BUILD_AND_TEST_EXE.ps1` without the `-SkipLaunch` flag.

---

## Step-by-Step Manual Terminal Build Process

If you prefer to execute each step manually or inspect intermediate steps, follow these instructions:

### Step 1 — Check Environment & Prerequisites

Open PowerShell and navigate to the project directory:

```powershell
cd C:\Users\bhask\Downloads\MindEd

Write-Host "`n=== Checking Python ===" -ForegroundColor Cyan
py -3.12 --version

Write-Host "`n=== Checking Node & NPM ===" -ForegroundColor Cyan
node --version
npm --version
```

---

### Step 2 — Create the Isolated Build Virtual Environment

Create and activate a dedicated virtual environment using Python 3.12:

```powershell
# Create .venv-exe-test environment
py -3.12 -m venv .venv-exe-test

# Activate virtual environment
.\.venv-exe-test\Scripts\Activate.ps1

# Verify the active Python interpreter
python -c "import sys; print('Active Python:', sys.executable, sys.version)"
```

---

### Step 3 — Install Hash-Locked Python Dependencies

Install all pinned and verified analytical & desktop dependencies:

```powershell
# 1. Upgrade pip, setuptools, wheel
python -m pip install --upgrade pip setuptools wheel

# 2. Install hash-locked analytical dependencies
python -m pip install --require-hashes -r requirements-lock.txt

# 3. Install desktop WebView and PyInstaller toolchain
python -m pip install --require-hashes -r packaging\windows\requirements-desktop.txt

# 4. Verify PyInstaller installation
python -m pip show pyinstaller
```

---

### Step 4 — Build the Next.js Frontend Static Export

MindEd embeds a local Next.js frontend into the desktop bundle. Compile it to static files:

```powershell
cd apps\web

# Clean previous build artifacts
if (Test-Path .next) { Remove-Item -Recurse -Force .next }
if (Test-Path out) { Remove-Item -Recurse -Force out }

# Install frontend dependencies from lockfile
npm ci --no-audit --ignore-scripts

# Build production static export (outputs to apps\web\out)
npm run build

# Return to repo root
cd ..\..

# Verify static frontend assets exist
Test-Path apps\web\out\index.html
```

---

### Step 5 — Run PyInstaller to Package the Desktop Bundle

Package the application into a standalone folder using PyInstaller:

```powershell
$Root = (Get-Location).Path
$BundleRoot = Join-Path $Root "packaging\windows\bundle"
$WorkRoot = Join-Path $env:TEMP ("MindedAAOS-PyInstaller-" + [guid]::NewGuid().ToString('N'))
$StaticOut = Join-Path $Root "apps\web\out"
$AppIcon = Join-Path $Root "packaging\windows\app.ico"
$OnefilePath = Join-Path $BundleRoot "standalone"

# Run PyInstaller
python -m PyInstaller --noconfirm --clean --name MindEd_AAOS_v1.0.0_Windows_x64 --onedir --noupx --windowed `
  --workpath $WorkRoot `
  --specpath $WorkRoot `
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
  --icon $AppIcon `
  --distpath $OnefilePath (Join-Path $Root "packaging\windows\minded_entry.py")

# Clean temp PyInstaller files
Remove-Item -Recurse -Force $WorkRoot -ErrorAction SilentlyContinue
```

---

### Step 6 — Finalize the Offline Bundle Structure

Organize the final bundle directory:

```powershell
$BuiltDir = Join-Path $OnefilePath "MindEd_AAOS_v1.0.0_Windows_x64"
$Bundle = Join-Path $BundleRoot "MindedAAOS"

if (Test-Path $Bundle) { Remove-Item -Recurse -Force $Bundle }
New-Item -ItemType Directory -Force -Path $Bundle | Out-Null

# Copy built binaries & payload
Copy-Item (Join-Path $BuiltDir "*") $Bundle -Recurse -Force

# Create standard launcher alias
Copy-Item (Join-Path $Bundle "MindEd_AAOS_v1.0.0_Windows_x64.exe") (Join-Path $Bundle "MindedAAOS.exe") -Force

# Copy static frontend and icon
Copy-Item $StaticOut (Join-Path $Bundle "frontend") -Recurse -Force
Copy-Item $AppIcon (Join-Path $Bundle "app.ico") -Force

# Set offline marker
Set-Content -Path (Join-Path $Bundle "OFFLINE_MODE") -Value "1" -Encoding ascii
```

---

### Step 7 — Run Offline Desktop Certification

Certify the bundle against offline verification requirements:

```powershell
python scripts\offline_desktop_certification.py --bundle packaging\windows\bundle
```

---

### Step 8 — Verify Built Artifacts & Launch

Check the generated executable and run it:

```powershell
# 1. Verify files exist
Get-ChildItem -Path packaging\windows\bundle\MindedAAOS

# 2. Check SHA-256 hash
Get-FileHash -Algorithm SHA256 packaging\windows\bundle\MindedAAOS\MindedAAOS.exe

# 3. Launch the Application
& "packaging\windows\bundle\MindedAAOS\MindedAAOS.exe"
```

---

## Built Executable Locations

- **Full Offline Bundle Folder**:
  `C:\Users\bhask\Downloads\MindEd\packaging\windows\bundle\MindedAAOS\`
- **Standard Launcher**:
  `C:\Users\bhask\Downloads\MindEd\packaging\windows\bundle\MindedAAOS\MindedAAOS.exe`
- **Versioned Executable**:
  `C:\Users\bhask\Downloads\MindEd\packaging\windows\bundle\MindedAAOS\MindEd_AAOS_v1.0.0_Windows_x64.exe`
