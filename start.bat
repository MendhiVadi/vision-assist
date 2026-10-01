@echo off
rem Vision Assist launcher: starts the model server and opens the web page automatically.
cd /d "%~dp0"
title Vision Assist

where python >nul 2>nul || (echo Python was not found. Install Python 3.10+ first. & pause & exit /b 1)

rem Install the package on first run (or if it was removed).
python -c "import vision_assist" >nul 2>nul || (
    echo First run: installing dependencies, this can take a while...
    python -m pip install -r requirements.txt && python -m pip install -e . || (echo Install failed. & pause & exit /b 1)
)

rem Stop a previous Vision Assist server so you always get the latest code.
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and $_.CommandLine -like '*vision_assist.api*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }" >nul 2>nul

echo Loading the model, your browser will open when it is ready...
echo Close this window to stop Vision Assist.
echo.
python -m vision_assist.api --model models/yolo11m.pt --lens-model models/yoloe-v8l-seg-pf.pt
pause
