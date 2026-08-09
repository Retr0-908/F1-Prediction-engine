@echo off
title F1 Cache Verification
color 0B

cd /d "%~dp0"

echo ==================================================
echo    F1 FANTASY - CACHE VERIFICATION & DOWNLOAD     
echo ==================================================
echo.
echo This script will verify that all required historical
echo race data (2021-Present) is downloaded and cached.
echo Missing data will be fetched from Jolpica/OpenF1.
echo.
echo Depending on how much data is missing, this may take
echo 5-15 minutes due to API rate limits.
echo.
pause

python warm_cache.py

echo.
pause
