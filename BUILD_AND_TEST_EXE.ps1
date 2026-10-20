[CmdletBinding()]
param(
    [switch]$SkipFocusedTests,
    [switch]$SkipLaunch,
    [string]$PythonExe = ""
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$Artifacts = Join-Path $Root 'artifacts\exe-test'
$LogPath = Join-Path $Artifacts "build_test_$Stamp.log"
$BundleRoot = Join-Path $Root 'packaging\windows\bundle'
$Bundle = Join-Path $BundleRoot 'MindedAAOS'
$Certification = Join-Path $Artifacts "offline_certification_$Stamp.json"
$TestResult = 'SKIPPED'
$BuildResult = 'NOT_STARTED'
$LaunchResult = 'NOT_REQUESTED'
$TranscriptStarted = $false

New-Item -ItemType Directory -Force -Path $Artifacts | Out-Null
try {
    Start-Transcript -Path $LogPath -Force | Out-Null
    $TranscriptStarted = $true

    Write-Host 'MindEd AA-OS: Windows EXE build and test' -ForegroundColor Cyan
    Write-Host "Source root: $Root"
    Write-Host "Log file:    $LogPath"

    if ($env:OS -ne 'Windows_NT') {
        throw 'This is a Windows-only EXE build. Run this script on Windows 10/11 x64, not Linux or macOS.'
    }

    $Node = Get-Command node.exe -ErrorAction SilentlyContinue
    $Npm = Get-Command npm -ErrorAction SilentlyContinue
    if (-not $Node -or -not $Npm) {
        throw 'Node.js and npm are required for the build step. Install a supported Node.js LTS release and reopen PowerShell.'
    }
    Write-Host "Node: $(& $Node.Source --version); npm: $(& $Npm.Source --version)"

    $Venv = Join-Path $Root '.venv-exe-test'
    $BuildPython = Join-Path $Venv 'Scripts\python.exe'

    if ($PythonExe) {
        if (-not (Test-Path $PythonExe -PathType Leaf)) { throw "PythonExe does not exist: $PythonExe" }
        $BasePython = (Resolve-Path $PythonExe).Path
        $Version = & $BasePython -c 'import sys; print(sys.version_info.major, sys.version_info.minor, sep=chr(46))'
        if ($LASTEXITCODE -ne 0 -or $Version.Trim() -ne '3.12') {
            throw "PythonExe must be CPython 3.12; detected $Version"
        }
        if (-not (Test-Path $BuildPython)) {
            & $BasePython -m venv $Venv
            if ($LASTEXITCODE -ne 0) { throw 'Failed creating isolated .venv-exe-test environment.' }
        }
    } else {
        $PyLauncher = Get-Command py.exe -ErrorAction SilentlyContinue
        if (-not $PyLauncher) { throw 'Python Launcher (py.exe) not found. Install 64-bit CPython 3.12 and select the Python Launcher.' }
        & $PyLauncher.Source -3.12 -c 'import sys; print(sys.version)'
        if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 not found. Install 64-bit CPython 3.12, then rerun this script.' }
        if (-not (Test-Path $BuildPython)) {
            & $PyLauncher.Source -3.12 -m venv $Venv
            if ($LASTEXITCODE -ne 0) { throw 'Failed creating isolated .venv-exe-test environment.' }
        }
    }

    if (-not (Test-Path $BuildPython -PathType Leaf)) { throw "Isolated build Python missing: $BuildPython" }
    $Version = & $BuildPython -c 'import sys; print(sys.version_info.major, sys.version_info.minor, sep=chr(46))'
    if ($LASTEXITCODE -ne 0 -or $Version.Trim() -ne '3.12') { throw "Build environment must use Python 3.12; detected $Version" }
    Write-Host "Build Python: $BuildPython (Python $($Version.Trim()))"

    if (-not (Test-Path (Join-Path $Root 'requirements-lock.txt'))) { throw 'requirements-lock.txt is missing.' }
    if (-not (Test-Path (Join-Path $Root 'packaging\windows\requirements-desktop.txt'))) { throw 'Windows desktop requirements lock is missing.' }
    if (-not (Test-Path (Join-Path $Root 'apps\web\package-lock.json'))) { throw 'apps\web\package-lock.json is missing.' }

    Write-Host 'Installing hash-locked analytical dependencies. This can take several minutes...' -ForegroundColor Yellow
    & $BuildPython -m pip install --require-hashes -r (Join-Path $Root 'requirements-lock.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Analytical requirements installation failed. See the transcript log.' }
    & $BuildPython -m pip install --require-hashes -r (Join-Path $Root 'packaging\windows\requirements-desktop.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Desktop requirements installation failed. See the transcript log.' }

    if (-not $SkipFocusedTests) {
        $TestPaths = @(
            'tests\independent_release\test_desktop_startup_and_parser_hardening.py',
            'tests\independent_release\test_audit_2026_10_17_filter_regressions.py',
            'tests\test_professional_report_contract.py',
            'tests\benchmark\test_report_benchmark_validator.py',
            'tests\test_offline_desktop_contract.py'
        )
        $Missing = @($TestPaths | Where-Object { -not (Test-Path (Join-Path $Root $_)) })
        if ($Missing.Count -gt 0) { throw "Focused regression test files missing: $($Missing -join ', ')" }
        Write-Host 'Running focused filter, parser, reporting and desktop regressions...' -ForegroundColor Yellow
        Push-Location $Root
        try {
            & $BuildPython -m pytest -q @TestPaths
            if ($LASTEXITCODE -eq 0) {
                $TestResult = 'PASS'
                Write-Host 'Focused regression suite: PASS' -ForegroundColor Green
            } else {
                $TestResult = "FAIL (exit $LASTEXITCODE)"
                Write-Warning 'Focused regressions failed. The script will still attempt the EXE build for manual testing, but this is not a release pass.'
            }
        } finally { Pop-Location }
    }

    Write-Host 'Building the Windows native desktop bundle from current source...' -ForegroundColor Yellow
    $WorkRoot = Join-Path $env:TEMP ("MindedAAOS-Exe-Test-" + [guid]::NewGuid().ToString('N'))
    try {
        & (Join-Path $Root 'packaging\windows\build_exe_scope.ps1') -PythonRuntimePath $BuildPython -OfflineBundleRoot $BundleRoot -PyInstallerWorkRoot $WorkRoot
        if ($LASTEXITCODE -ne 0) { throw "build_exe_scope.ps1 failed with exit code $LASTEXITCODE" }
    } finally {
        if (Test-Path $WorkRoot) { Remove-Item -Recurse -Force $WorkRoot -ErrorAction SilentlyContinue }
    }

    $Exe = Join-Path $Bundle 'MindedAAOS.exe'
    if (-not (Test-Path $Exe -PathType Leaf)) { throw "Expected EXE not found at $Exe" }
    $Hash = (Get-FileHash -Algorithm SHA256 $Exe).Hash
    Write-Host "Built EXE: $Exe"
    Write-Host "SHA-256:   $Hash"

    & $BuildPython (Join-Path $Root 'scripts\offline_desktop_certification.py') --bundle $BundleRoot --json $Certification
    if ($LASTEXITCODE -ne 0) { throw "Offline bundle certification failed. See $Certification" }
    $BuildResult = 'PASS'

    if (-not $SkipLaunch) {
        Write-Host 'Launching the EXE for manual GUI testing...' -ForegroundColor Yellow
        $LogFile = Join-Path $env:LOCALAPPDATA 'Minded\AAOS\logs\desktop_startup.log'
        $BeforeWrite = if (Test-Path $LogFile) { (Get-Item $LogFile).LastWriteTimeUtc } else { [datetime]::MinValue }
        $Process = Start-Process -FilePath $Exe -PassThru
        Start-Sleep -Seconds 8
        if ($Process.HasExited) {
            $LaunchResult = "PROCESS_EXITED (code $($Process.ExitCode))"
            Write-Warning 'The EXE exited quickly. Check desktop_startup.log and the native error dialog, if shown.'
        } else {
            $LaunchResult = 'PROCESS_RUNNING; GUI requires human confirmation'
            Write-Host 'EXE process is running. Confirm the MindEd native window opened and can be used.' -ForegroundColor Green
        }
        if (Test-Path $LogFile) {
            $LogInfo = Get-Item $LogFile
            if ($LogInfo.LastWriteTimeUtc -ge $BeforeWrite) {
                Write-Host "Startup log: $LogFile"
                Get-Content $LogFile -Tail 25
            }
        } else {
            Write-Warning "Startup log not found yet: $LogFile"
        }
    }

    $Summary = [ordered]@{
        product = 'MindEd AA-OS'
        built_at_local = (Get-Date).ToString('o')
        source_root = $Root
        python = $Version.Trim()
        focused_regressions = $TestResult
        exe_build_and_offline_certification = $BuildResult
        launch_status = $LaunchResult
        executable = if (Test-Path (Join-Path $Bundle 'MindedAAOS.exe')) { Join-Path $Bundle 'MindedAAOS.exe' } else { $null }
        sha256 = if (Test-Path (Join-Path $Bundle 'MindedAAOS.exe')) { (Get-FileHash -Algorithm SHA256 (Join-Path $Bundle 'MindedAAOS.exe')).Hash } else { $null }
        certification_json = $Certification
        log = $LogPath
        note = 'A running process is not proof of successful GUI or analytical behavior. Complete the manual checklist in HOW_TO_TEST_THE_EXE.md.'
    }
    $SummaryPath = Join-Path $Artifacts "exe_test_summary_$Stamp.json"
    $Summary | ConvertTo-Json -Depth 5 | Set-Content -Path $SummaryPath -Encoding UTF8

    Write-Host ''
    Write-Host 'Build/test summary' -ForegroundColor Cyan
    Write-Host "Focused regressions: $TestResult"
    Write-Host "EXE build/certification: $BuildResult"
    Write-Host "Launch: $LaunchResult"
    Write-Host "Summary JSON: $SummaryPath"
    if ($TestResult -like 'FAIL*') { Write-Warning 'The EXE was built for inspection, but the overall acceptance gate is NOT PASSED because a focused regression failed.' }
} catch {
    Write-Error $_
    Write-Host "Build/test did not complete. Review: $LogPath" -ForegroundColor Red
    exit 1
} finally {
    if ($TranscriptStarted) { Stop-Transcript | Out-Null }
}
