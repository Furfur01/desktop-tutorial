@echo off
setlocal
title Atmos20
cd /d "%~dp0"
set "PYTHONPATH=%CD%\src"

powershell.exe -NoProfile -Command "try { $r = Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:7861/' -TimeoutSec 1; if ($r.Content -match 'Atmos20') { exit 0 }; exit 1 } catch { exit 1 }"
if not errorlevel 1 (
    start "" "http://127.0.0.1:7861/"
    exit /b 0
)

if not exist ".venv\Scripts\python.exe" (
    py -3 --version >nul 2>&1
    if not errorlevel 1 (
        py -3 -m venv .venv
    ) else (
        python --version >nul 2>&1
        if errorlevel 1 goto no_python
        python -m venv .venv
    )
    if errorlevel 1 goto setup_failed
)

".venv\Scripts\python.exe" -c "import atmos20, numpy, matplotlib, PIL, scipy" >nul 2>&1
if errorlevel 1 (
    echo Installing Atmos20 dependencies for the first run...
    ".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt
    if errorlevel 1 goto setup_failed
)

echo Starting Atmos20 at http://127.0.0.1:7861/
echo Keep this window open. Press Ctrl+C to stop.
start "" powershell.exe -NoProfile -WindowStyle Hidden -Command "for ($i=0; $i -lt 60; $i++) { try { $r = Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:7861/' -TimeoutSec 1; if ($r.Content -match 'Atmos20') { Start-Process 'http://127.0.0.1:7861/'; exit } } catch {}; Start-Sleep -Seconds 1 }"
".venv\Scripts\python.exe" -u windy_app.py --host 127.0.0.1 --port 7861
exit /b %errorlevel%

:no_python
echo Python 3.10 or newer is required: https://www.python.org/downloads/windows/
pause
exit /b 1

:setup_failed
echo Atmos20 setup failed. Check the error above and try again.
pause
exit /b 1
