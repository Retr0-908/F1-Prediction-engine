@echo off
title F1 Fantasy Prediction Tool
color 0C
echo.
echo  =============================================
echo    F1 FANTASY PREDICTION TOOL  - 2026 Season
echo  =============================================
echo.

cd /d "%~dp0"

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo  ERROR: Python not found. Please install Python 3.10+
    echo  Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

:menu
cls
echo  =============================================
echo    F1 FANTASY PREDICTION TOOL  - 2026 Season
echo  =============================================
echo.
echo  [1] Run Predictor (Interactive)
echo  [2] Run Predictor (Non-interactive)
echo  [3] Update Dependencies (pip)
echo  [4] Exit
echo.

set /p opt="Select an option (1-4): "

if "%opt%"=="1" goto run_interactive
if "%opt%"=="2" goto run_auto
if "%opt%"=="3" goto update_deps
if "%opt%"=="4" goto end
echo Invalid option. Please try again.
pause
goto menu

:run_interactive
echo  Starting prediction tool...
echo.
python main.py %*
echo.
pause
goto menu

:run_auto
echo  Starting prediction tool (non-interactive)...
echo.
python main.py --auto
echo.
pause
goto menu

:update_deps
:: Install/check ALL deps from requirements.txt (the old inline list omitted
:: fastapi/uvicorn/pulp/tensorflow/juliacall/sse-starlette)
echo  Checking dependencies (from requirements.txt)...
pip install -q -r "%~dp0requirements.txt" 2>nul
echo  Installing headless browser support...
python -m playwright install chromium >nul 2>&1

echo.
pause
goto menu

:end
echo Exiting F1 Fantasy Prediction Tool.
exit /b 0
