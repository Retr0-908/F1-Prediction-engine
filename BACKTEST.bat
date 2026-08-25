@echo off
title F1 Backtester
color 0A
cd /d "%~dp0"
echo.
echo  =============================================
echo    F1 PREDICTION BACKTESTER
echo  =============================================
echo.
echo  Running backtest on: 2024 and 2025 races
echo.
python -m engine.analysis.backtest --years 2024 2025
echo.
pause
