@echo off
chcp 65001 >nul
cd /d "%~dp0"
title defect-vision toolbox
python tools\toolbox.py
if errorlevel 1 (
  echo.
  echo ==== program exited with an error, see above ====
)
echo.
pause
