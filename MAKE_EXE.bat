@echo off
setlocal
cd /d "%~dp0"
echo ========================================================
echo MindEd AA-OS: Building Windows Desktop Offline Executable
echo ========================================================
echo.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0BUILD_AND_TEST_EXE.ps1" -SkipLaunch
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================================
    echo SUCCESS: Executable created at:
    echo %~dp0packaging\windows\bundle\MindedAAOS\MindedAAOS.exe
    echo ========================================================
) else (
    echo.
    echo ========================================================
    echo BUILD FAILED. Review the error output above.
    echo ========================================================
)
pause
