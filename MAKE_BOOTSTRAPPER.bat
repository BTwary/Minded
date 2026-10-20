@echo off
setlocal
cd /d "%~dp0"
echo ========================================================
echo MindEd AA-OS: Building Lightweight Setup (MindedSetup.exe)
echo ========================================================
echo.
python "%~dp0packaging\windows\installer\build_bootstrapper.py"
if %ERRORLEVEL% NEQ 0 (
    echo Retrying with Python 3.12 build environment...
    "%~dp0.venv-exe-test\Scripts\python.exe" "%~dp0packaging\windows\installer\build_bootstrapper.py"
)
pause
