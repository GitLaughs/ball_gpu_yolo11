@echo off
setlocal EnableExtensions
chcp 65001 >nul 2>&1

set "ROOT_DIR=%~dp0"
pushd "%ROOT_DIR%" >nul

title Basketball Shot Detection - Launcher
color 0E

echo ============================================================
echo   Basketball Shot Detection - Launcher
echo ============================================================
echo.

call :detect_venv
if not defined VENV_DIR (
    echo [INFO] Virtual environment was not found.
    echo        Starting install_gpu.bat for first-time setup...
    echo.
    call "%ROOT_DIR%install_gpu.bat"
    if errorlevel 1 (
        echo [ERROR] Installation was not completed successfully.
        echo.
        pause
        popd >nul
        exit /b 1
    )
    call :detect_venv
)

if not defined VENV_DIR (
    echo [ERROR] No usable virtual environment was found after installation.
    echo.
    pause
    popd >nul
    exit /b 1
)

set "VENV_PY=%ROOT_DIR%%VENV_DIR%\Scripts\python.exe"
if not exist "%VENV_PY%" (
    echo [ERROR] Virtual environment Python not found:
    echo         %VENV_PY%
    echo.
    pause
    popd >nul
    exit /b 1
)

if not exist "static" mkdir "static"
if not exist "uploads" mkdir "uploads"
if not exist "exports" mkdir "exports"
if not exist "logs" mkdir "logs"
if not exist "static\index.html" if exist "static\index_old.html" (
    copy /Y "static\index_old.html" "static\index.html" >nul
)

if not exist "static\index.html" (
    echo [ERROR] Frontend page not found: static\index.html
    echo         Please check the repository files.
    echo.
    pause
    popd >nul
    exit /b 1
)

if exist "best_yolo11.pt" (
    echo [OK] Using model: best_yolo11.pt
) else if exist "best.pt" (
    echo [OK] Using model: best.pt
) else (
    echo [ERROR] Model file not found.
    echo         Put best.pt or best_yolo11.pt in the project root.
    echo.
    pause
    popd >nul
    exit /b 1
)

echo Checking runtime dependencies...
"%VENV_PY%" -c "import flask, cv2, torch, ultralytics"
if errorlevel 1 (
    echo [ERROR] Runtime dependencies are incomplete.
    echo         Please run install_gpu.bat or install_cpu.bat again.
    echo.
    pause
    popd >nul
    exit /b 1
)

for /f "usebackq delims=" %%I in (`"%VENV_PY%" -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"` ) do set "DEVICE_NAME=%%I"
if not defined DEVICE_NAME set "DEVICE_NAME=CPU"

echo [OK] Runtime device: %DEVICE_NAME%
echo [OK] Working directory: %ROOT_DIR%
echo.
echo Opening browser: http://127.0.0.1:5000
echo Press Ctrl+C in this window to stop the service.
echo.

start "" http://127.0.0.1:5000
"%VENV_PY%" app.py

echo.
echo Service stopped.
pause
popd >nul
exit /b 0

:detect_venv
set "VENV_DIR="
if exist "venv\Scripts\python.exe" set "VENV_DIR=venv"
if not defined VENV_DIR if exist ".venv\Scripts\python.exe" set "VENV_DIR=.venv"
exit /b 0
