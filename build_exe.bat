@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Python virtual environment not found: .venv\Scripts\python.exe
    exit /b 1
)

echo [INFO] Building Windows executable with PyInstaller...
".venv\Scripts\python.exe" -m PyInstaller ^
  --noconfirm ^
  --clean ^
  --name ball_gpu_yolo11 ^
  --onedir ^
  --add-data "static;static" ^
  --add-data "best.pt;." ^
  --collect-submodules ultralytics ^
  --collect-data ultralytics ^
  --hidden-import cv2 ^
  --hidden-import flask_cors ^
  app.py

if errorlevel 1 (
    echo [ERROR] Build failed.
    exit /b %errorlevel%
)

echo [OK] Build completed.
echo [OK] Output folder: dist\ball_gpu_yolo11\
exit /b 0
