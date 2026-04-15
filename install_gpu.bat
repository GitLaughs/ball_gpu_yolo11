@echo off
setlocal EnableExtensions
chcp 65001 >nul 2>&1

set "ROOT_DIR=%~dp0"
pushd "%ROOT_DIR%" >nul

title Basketball Shot Detection - GPU Installer
color 0A

echo ============================================================
echo   Basketball Shot Detection - One Click GPU Install
echo ============================================================
echo.

set "PY_CMD=python"
%PY_CMD% --version >nul 2>&1
if errorlevel 1 (
    set "PY_CMD=py -3"
    %PY_CMD% --version >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Python 3 was not found.
        echo Install Python 3.10+ and enable "Add Python to PATH".
        echo Download: https://www.python.org/downloads/
        echo.
        pause
        exit /b 1
    )
)

echo [1/7] Python detected:
%PY_CMD% --version
echo.

echo [2/7] Checking NVIDIA GPU...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [WARN] NVIDIA GPU or driver was not detected.
    echo        If this machine is CPU-only, use install_cpu.bat instead.
    set /p CONTINUE=Continue GPU package installation anyway? [Y/N]: 
    if /I not "%CONTINUE%"=="Y" (
        echo Installation cancelled.
        echo.
        pause
        exit /b 0
    )
) else (
    echo [OK] NVIDIA GPU detected:
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
)
echo.

set "VENV_DIR="
if exist "venv\Scripts\python.exe" set "VENV_DIR=venv"
if not defined VENV_DIR if exist ".venv\Scripts\python.exe" set "VENV_DIR=.venv"
if not defined VENV_DIR set "VENV_DIR=venv"

echo [3/7] Preparing virtual environment: %VENV_DIR%
if exist "%VENV_DIR%\Scripts\python.exe" (
    echo [OK] Reusing existing virtual environment.
) else (
    %PY_CMD% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        echo.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
)
echo.

set "VENV_PY=%ROOT_DIR%%VENV_DIR%\Scripts\python.exe"
if not exist "%VENV_PY%" (
    echo [ERROR] Virtual environment Python not found:
    echo         %VENV_PY%
    echo.
    pause
    exit /b 1
)

echo [4/7] Upgrading pip tools...
"%VENV_PY%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo [ERROR] Failed to upgrade pip/setuptools/wheel.
    echo.
    pause
    exit /b 1
)
echo.

echo [5/7] Installing PyTorch with CUDA 12.1...
"%VENV_PY%" -m pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo [ERROR] PyTorch GPU installation failed.
    echo         Please check your network or CUDA driver compatibility.
    echo.
    pause
    exit /b 1
)
echo.

echo [6/7] Installing project dependencies...
"%VENV_PY%" -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python-headless >nul 2>&1
"%VENV_PY%" -m pip install Flask==3.1.0 Flask-Cors==4.0.0
if errorlevel 1 goto :install_failed
"%VENV_PY%" -m pip install opencv-contrib-python==4.10.0.84
if errorlevel 1 goto :install_failed
"%VENV_PY%" -m pip install numpy==1.26.4 scipy==1.16.3 matplotlib==3.9.2 Pillow==11.1.0
if errorlevel 1 goto :install_failed
"%VENV_PY%" -m pip install ultralytics==8.3.227 tqdm psutil requests PyYAML colorama
if errorlevel 1 goto :install_failed
echo [OK] Core packages installed.
echo.

echo [7/7] Preparing runtime files...
if not exist "static" mkdir "static"
if not exist "uploads" mkdir "uploads"
if not exist "exports" mkdir "exports"
if not exist "logs" mkdir "logs"
if not exist "static\index.html" if exist "static\index_old.html" (
    copy /Y "static\index_old.html" "static\index.html" >nul
    echo [OK] Restored static\index.html from static\index_old.html
)
echo.

echo Verifying installation...
"%VENV_PY%" -c "import flask, cv2, torch, ultralytics; print('Python      : OK'); print('Flask       :', flask.__version__); print('OpenCV      :', cv2.__version__); print('PyTorch     :', torch.__version__); print('CUDA Ready  :', torch.cuda.is_available()); print('Ultralytics :', ultralytics.__version__)"
if errorlevel 1 (
    echo [ERROR] Verification failed. Please review the install output above.
    echo.
    pause
    exit /b 1
)
echo.

if exist "best_yolo11.pt" (
    echo [OK] Model found: best_yolo11.pt
) else if exist "best.pt" (
    echo [OK] Model found: best.pt
) else (
    echo [WARN] Model file was not found in the project root.
    echo        Put best.pt or best_yolo11.pt next to app.py before running.
)
echo.

echo ============================================================
echo   Installation completed.
echo   Start the project by double-clicking start.bat
echo   Web URL: http://127.0.0.1:5000
echo ============================================================
echo.
pause
popd >nul
exit /b 0

:install_failed
echo [ERROR] Dependency installation failed.
echo         Please review the output above and retry.
echo.
pause
popd >nul
exit /b 1
