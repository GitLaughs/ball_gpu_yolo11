@REM filepath: check_env.bat
@echo off
chcp 65001 >nul 2>&1
title 🏀 环境检测工具
color 0F

echo ══════════════════════════════════════════════════════════
echo          🏀 AI 篮球投篮分析系统 - 环境检测
echo ══════════════════════════════════════════════════════════
echo.

echo ── 系统信息 ──
echo   OS: %OS%
ver
echo.

echo ── Python ──
python --version 2>nul || echo   [未安装]
echo.

echo ── NVIDIA GPU ──
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>nul || echo   [未检测到 NVIDIA GPU]
echo.

echo ── 虚拟环境 ──
if exist venv\Scripts\activate.bat (
    echo   ✓ 虚拟环境存在
    call venv\Scripts\activate.bat

    echo.
    echo ── Python 包版本 ──
    python -c "import torch; print(f'  PyTorch:    {torch.__version__}')" 2>nul || echo   PyTorch:    [未安装]
    python -c "import torch; print(f'  CUDA可用:   {torch.cuda.is_available()}')" 2>nul
    python -c "import torch; print(f'  GPU名称:    {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>nul
    python -c "import cv2; print(f'  OpenCV:     {cv2.__version__}')" 2>nul || echo   OpenCV:     [未安装]
    python -c "import ultralytics; print(f'  Ultralytics:{ultralytics.__version__}')" 2>nul || echo   Ultralytics:[未安装]
    python -c "import flask; print(f'  Flask:      {flask.__version__}')" 2>nul || echo   Flask:      [未安装]
    python -c "import numpy; print(f'  NumPy:      {numpy.__version__}')" 2>nul || echo   NumPy:      [未安装]
) else (
    echo   ✗ 虚拟环境不存在，请先运行 install.bat
)
echo.

echo ── 项目文件 ──
if exist app.py          (echo   ✓ app.py) else (echo   ✗ app.py [缺失])
if exist best.pt         (echo   ✓ best.pt) else if exist best_yolo11.pt (echo   ✓ best_yolo11.pt) else (echo   ⚠ 模型文件 [缺失])
if exist static\index.html (echo   ✓ static/index.html) else (echo   ✗ static/index.html [缺失])
if exist engine\config.py  (echo   ✓ engine/) else (echo   ✗ engine/ [缺失])
if exist ball.avi        (echo   ✓ ball.avi) else (echo   ℹ ball.avi [可选])
echo.

echo ══════════════════════════════════════════════════════════
pause