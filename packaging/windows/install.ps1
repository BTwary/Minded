param(
  [string]$BundlePath = (Join-Path $PSScriptRoot 'bundle\MindedAAOS')
)
$ErrorActionPreference = 'Stop'

Write-Host "Minded AA-OS offline desktop installer"

if (-not (Test-Path $BundlePath -PathType Container)) {
  throw "Self-contained desktop bundle not found at '$BundlePath'. Build the certified offline bundle before installation."
}

$Exe = Join-Path $BundlePath 'MindedAAOS.exe'
$Frontend = Join-Path $BundlePath 'frontend'
$Marker = Join-Path $BundlePath 'OFFLINE_MODE'
if (-not (Test-Path $Exe -PathType Leaf)) { throw "MindedAAOS.exe is missing from the release bundle." }
if (-not (Test-Path $Frontend -PathType Container)) { throw "Static frontend bundle is missing from the release bundle." }
if (-not (Test-Path $Marker -PathType Leaf)) { throw "Offline release marker is missing." }

$InstallRoot = Join-Path $env:LOCALAPPDATA 'Minded\AAOS\app'
if (Test-Path $InstallRoot) { Remove-Item -Recurse -Force $InstallRoot }
New-Item -ItemType Directory -Force -Path $InstallRoot | Out-Null
Copy-Item (Join-Path $BundlePath '*') $InstallRoot -Recurse -Force

# The application itself creates mutable state under %LOCALAPPDATA%\Minded\AAOS.
# No Python/Node/npm/pip installation or network access is performed here.
$ShortcutDir = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs\Minded'
New-Item -ItemType Directory -Force -Path $ShortcutDir | Out-Null
$ShortcutPath = Join-Path $ShortcutDir 'Minded AA-OS.lnk'
$Shell = New-Object -ComObject WScript.Shell
$Shortcut = $Shell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = (Join-Path $InstallRoot 'MindedAAOS.exe')
$Shortcut.WorkingDirectory = $InstallRoot
$Shortcut.Save()

Write-Host "Offline desktop application installed to $InstallRoot"
Write-Host "No Python, Node, npm, pip, cloud service, or internet connection is required at runtime."
Write-Host "Launch it from the Start Menu shortcut 'Minded AA-OS' -- it opens as its own application window, not a browser tab."
