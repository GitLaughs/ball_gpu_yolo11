@REM filepath: install_cpu.bat
@echo off
chcp 65001 >nul 2>&1
title 🏀 AI 篮球投篮分析系统 - 安装程序 (CPU版)
color 0B

echo ══════════════════════════════════════════════════════════
echo    🏀 AI 篮球投篮分析系统 - 一键安装脚本 (CPU 版)
echo ══════════════════════════════════════════════════════════
echo.
echo    ℹ 此脚本安装 CPU 版本的 PyTorch（无需 NVIDIA GPU）
echo.

REM ── 检查 Python ──
echo [1/5] 检查 Python 环境...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未检测到 Python！
    echo        请从 https://www.python.org/downloads/ 下载安装 Python 3.10+
    echo        安装时请勾选 "Add Python to PATH"
    pause
    exit /b 1
)
python --version
echo       ✓ Python 已安装
echo.

REM ── 创建虚拟环境 ──
echo [2/5] 创建 Python 虚拟环境...
if exist venv (
    echo       虚拟环境已存在，跳过创建。
) else (
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [错误] 虚拟环境创建失败！
        pause
        exit /b 1
    )
    echo       ✓ 虚拟环境创建成功
)
echo.

REM ── 激活虚拟环境 ──
call venv\Scripts\activate.bat

REM ── 升级 pip ──
echo [3/5] 升级 pip...
python -m pip install --upgrade pip setuptools wheel -q
echo       ✓ pip 已升级
echo.

REM ── 安装 PyTorch (CPU) ──
echo [4/5] 安装 PyTorch (CPU 版本)...
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
if %errorlevel% neq 0 (
    echo [错误] PyTorch 安装失败！请检查网络连接。
    pause
    exit /b 1
)
echo       ✓ PyTorch CPU 版已安装
echo.

REM ── 安装其他依赖 ──
echo [5/5] 安装其他项目依赖...
pip install flask==3.1.0 flask-cors==4.0.0 -q
pip install opencv-contrib-python==4.10.0.84 -q
pip install numpy==1.26.4 scipy matplotlib -q
pip install ultralytics==8.3.227 -q
pip install tqdm psutil requests pyyaml colorama pillow -q
echo       ✓ 所有依赖安装完成
echo.

REM ── 创建必要目录 ──
if not exist static  mkdir static
if not exist uploads mkdir uploads
if not exist exports mkdir exports
if not exist logs    mkdir logs

REM ── 验证安装 ──
echo ──────────────────────────────────────────────────────────
echo   验证安装结果...
echo ──────────────────────────────────────────────────────────
python -c "import torch; print(f'  PyTorch: {torch.__version__} (CPU)')"
python -c "import cv2; print(f'  OpenCV:  {cv2.__version__}')"
python -c "import ultralytics; print(f'  YOLO:    {ultralytics.__version__}')"
python -c "import flask; print(f'  Flask:   {flask.__version__}')"
echo.

if exist best.pt (
    echo   ✓ 模型文件 best.pt 已找到
) else (
    echo   ⚠ 未找到模型文件！请将 best.pt 放在项目根目录下。
)
echo.

echo ══════════════════════════════════════════════════════════
echo   ✅ 安装完成！(CPU 版本)
echo.
echo   启动: 双击 start.bat 或执行 python app.py
echo   访问: http://127.0.0.1:5000
echo.
echo   ⚠ CPU 模式下处理速度较慢，建议使用较短的视频。
echo ══════════════════════════════════════════════════════════
pause
