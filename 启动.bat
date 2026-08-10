@echo off
title Support/Resistance Detection Web App
cd /d "%~dp0"

echo.
echo ============================================================
echo   Support/Resistance Detection Web App
echo   Dir: %CD%
echo ============================================================
echo.

REM Check Python
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] python not found. Please install Python 3 and add to PATH.
    pause
    exit /b 1
)

REM Check deps, auto install if missing
python -c "import flask, pyarrow" 2>nul
if errorlevel 1 (
    echo [INFO] Missing dependencies, installing requirements.txt ...
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Install failed. Run manually: python -m pip install -r requirements.txt
        pause
        exit /b 1
    )
    echo.
)

echo Starting server, browser will open http://127.0.0.1:5000
echo Press Ctrl+C to stop.
echo.

REM Open browser after 3 seconds
start "" /b powershell -NoProfile -Command "Start-Sleep -Seconds 3; Start-Process 'http://127.0.0.1:5000'"

python app.py

pause
