@echo off
title F1 Fantasy Engine
color 0C

cd /d "%~dp0"

echo  =============================================
echo    F1 FANTASY PREDICTION TOOL - GUI ENGINE
echo  =============================================
echo.
echo Starting FastAPI Backend Server...
echo Please keep this window open while using the application.
echo Close this window to shut down the server.
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.10+
    pause
    exit /b 1
)

:: Start the server in the foreground
python -m engine.serving.server

pause
