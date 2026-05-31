@echo off
title YouTube Sentiment Lens Launcher

echo ===================================================
echo     YouTube Sentiment Lens Launcher - Antigravity
echo ===================================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 goto :no_python

:: Check if virtual environment exists
if exist .venv goto :activate_venv

echo [INFO] Creating local virtual environment (.venv)...
python -m venv .venv
if errorlevel 1 goto :venv_failed

:activate_venv
echo [INFO] Activating virtual environment...
call .venv\Scripts\activate
if errorlevel 1 goto :activation_failed

:: If sentinel file exists, skip package installation
if exist .venv\installed.sentinel goto :launch

echo [INFO] Installing dependencies (first-time setup)...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 goto :install_failed

:: Create sentinel file to skip installation next time
echo. > .venv\installed.sentinel
goto :launch

:venv_failed
echo [WARNING] Failed to create virtual environment.
echo [INFO] Attempting global run...
goto :global_run

:activation_failed
echo [WARNING] Failed to activate virtual environment.
echo [INFO] Attempting global run...
goto :global_run

:global_run
:: Check if global modules are already installed by importing flask
python -c "import flask" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing dependencies globally...
    python -m pip install -r requirements.txt
    if errorlevel 1 goto :install_failed
)
goto :launch_global

:launch
echo [INFO] Starting Flask backend server...
python main.py
goto :end

:launch_global
echo [INFO] Starting Flask backend server globally...
python main.py
goto :end

:no_python
echo [ERROR] Python is not installed or not added to your system PATH.
echo Please install Python 3.8+ from https://www.python.org/ and tick "Add Python to PATH".
echo.
pause
exit /b 1

:install_failed
echo [ERROR] Dependency installation failed.
echo Please check your internet connection and try again.
echo.
pause
exit /b 1

:end
pause
