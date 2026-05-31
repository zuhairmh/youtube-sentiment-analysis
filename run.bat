@echo off
title YouTube Sentiment Lens Launcher
echo ===================================================
echo     YouTube Sentiment Lens Launcher - Antigravity
echo ===================================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not added to your system PATH.
    echo Please install Python 3.8+ and try again.
    pause
    exit /b 1
)

:: Check for existing virtual environment
if not exist .venv (
    echo [INFO] Creating local virtual environment (.venv)...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [WARNING] Failed to create virtual environment. Running in global context...
        goto :global_run
    )
)

echo [INFO] Activating virtual environment...
call .venv\Scripts\activate

echo [INFO] Installing/verifying dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

goto :launch

:global_run
echo [INFO] Installing/verifying dependencies globally...
python -m pip install -r requirements.txt

:launch
echo [INFO] Starting Flask backend server...
echo [INFO] Opening dashboard in browser...
start http://127.0.0.1:5000
python main.py

pause
