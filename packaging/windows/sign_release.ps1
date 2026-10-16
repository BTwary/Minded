<#
Authenticode-sign every Windows release artifact and fail closed unless every signature verifies.

  .\sign_release.ps1 -Path <bundle\MindedAAOS or folder with the exes> -Thumbprint <cert thumbprint>
  .\sign_release.ps1 -Path ... -PfxPath cert.pfx -PfxPasswordEnv MINDED_SIGN_PFX_PASSWORD

The certificate is never stored in the repository: pass a thumbprint from the machine's certificate
store, or a PFX path plus the NAME of an environment variable holding its password. A timestamp is
always applied so signatures outlive the certificate. Without a certificate this script throws; it
never produces unsigned "signed" output.
#>
param(
  [Parameter(Mandatory = $true)][string]$Path,
  [string]$Thumbprint = "",
  [string]$PfxPath = "",
  [string]$PfxPasswordEnv = "",
  [string]$TimestampUrl = "http://timestamp.digicert.com"
)
$ErrorActionPreference = 'Stop'

if (-not $Thumbprint -and -not $PfxPath) { throw "Provide -Thumbprint or -PfxPath: refusing to continue without a code-signing certificate." }
if ($PfxPath -and -not (Test-Path $PfxPath -PathType Leaf)) { throw "PFX not found: $PfxPath" }
if (-not (Test-Path $Path)) { throw "Path not found: $Path" }

$SignTool = (Get-Command signtool.exe -ErrorAction SilentlyContinue).Source
if (-not $SignTool) {
  $SignTool = Get-ChildItem "${env:ProgramFiles(x86)}\Windows Kits\10\bin" -Recurse -Filter signtool.exe -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -match '\\x64\\' } | Sort-Object FullName -Descending | Select-Object -First 1 -ExpandProperty FullName
}
if (-not $SignTool) { throw "signtool.exe not found (install the Windows SDK)." }

# Everything an end user can execute: the product exes, launcher, uninstaller and the setup wizard.
$Targets = Get-ChildItem -Path $Path -Recurse -File -Include *.exe, *.dll, *.pyd -ErrorAction Stop |
  Where-Object { $_.FullName -notmatch '\\_internal\\' -or $_.Extension -eq '.exe' }
if (-not $Targets) { throw "No signable files found under $Path." }

$Args = @('sign', '/fd', 'SHA256', '/tr', $TimestampUrl, '/td', 'SHA256')
if ($Thumbprint) { $Args += @('/sha1', $Thumbprint) }
else {
  $Args += @('/f', $PfxPath)
  if ($PfxPasswordEnv) { $Args += @('/p', [Environment]::GetEnvironmentVariable($PfxPasswordEnv)) }
}
foreach ($File in $Targets) {
  & $SignTool @Args $File.FullName
  if ($LASTEXITCODE -ne 0) { throw "Signing failed: $($File.FullName)" }
}

& (Join-Path $PSScriptRoot 'verify_release_signature.ps1') -Path $Path
if ($LASTEXITCODE -ne 0) { throw "Post-sign verification failed." }
Write-Host "All $($Targets.Count) file(s) signed and verified."
