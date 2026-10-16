<#
Fail unless every release executable under -Path carries a VALID, timestamped Authenticode signature.
Run this on the clean test machine as well: it is the acceptance check for "signed release".
#>
param([Parameter(Mandatory = $true)][string]$Path)
$ErrorActionPreference = 'Stop'

$Exes = Get-ChildItem -Path $Path -Recurse -File -Filter *.exe | Where-Object { $_.FullName -notmatch '\\_internal\\' }
if (-not $Exes) { Write-Host "SIGNATURE_AUDIT: FAIL (no executables under $Path)"; exit 1 }

$Bad = @()
foreach ($File in $Exes) {
  $Sig = Get-AuthenticodeSignature -FilePath $File.FullName
  if ($Sig.Status -ne 'Valid') { $Bad += "$($File.Name): $($Sig.Status)"; continue }
  if (-not $Sig.TimeStamperCertificate) { $Bad += "$($File.Name): signature has no timestamp" }
}
if ($Bad.Count -gt 0) {
  Write-Host "SIGNATURE_AUDIT: FAIL"
  $Bad | ForEach-Object { Write-Host "  - $_" }
  exit 1
}
Write-Host "SIGNATURE_AUDIT: PASS ($($Exes.Count) executables)"
exit 0
