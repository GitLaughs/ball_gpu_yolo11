@REM filepath: install_gpu.bat
@echo off
chcp 65001 >nul 2>&1
title 🏀 AI 篮球投篮分析系统 - 安装程序 (GPU版)
color 0A

echo ══════════════════════════════════════════════════════════
echo    🏀 AI 篮球投篮分析系统 - 一键安装脚本 (GPU 版)
echo ══════════════════════════════════════════════════════════
echo.

REM ── 检查 Python ──
echo [1/6] 检查 Python 环境...
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

REM ── 检查 NVIDIA GPU ──
echo [2/6] 检查 NVIDIA GPU...
nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo [警告] 未检测到 NVIDIA GPU 或驱动未安装！
    echo        将继续安装 CUDA 版 PyTorch，但运行时可能回退到 CPU 模式。
    echo        如果您没有 NVIDIA GPU，建议使用 install_cpu.bat 安装。
    echo.
    set /p CONTINUE="是否继续？(Y/N): "
    if /i not "%CONTINUE%"=="Y" (
        echo 已取消安装。请使用 install_cpu.bat 安装 CPU 版本。
        pause
        exit /b 0
    )
) else (
    echo       ✓ NVIDIA GPU 已检测到
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
)
echo.

REM ── 创建虚拟环境 ──
echo [3/6] 创建 Python 虚拟环境...
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
echo [4/6] 升级 pip...
python -m pip install --upgrade pip setuptools wheel -q
echo       ✓ pip 已升级
echo.

REM ── 安装 PyTorch (CUDA 12.1) ──
echo [5/6] 安装 PyTorch (CUDA 12.1 版本)...
echo       这可能需要几分钟，请耐心等待...
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
if %errorlevel% neq 0 (
    echo [错误] PyTorch 安装失败！请检查网络连接。
    echo        也可手动执行:
    echo        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pause
    exit /b 1
)
echo       ✓ PyTorch CUDA 版已安装
echo.

REM ── 安装其他依赖 ──
echo [6/6] 安装其他项目依赖...
pip install flask==3.1.0 flask-cors==4.0.0 -q
pip install opencv-contrib-python==4.10.0.84 -q
pip install numpy==1.26.4 scipy matplotlib -q
pip install ultralytics==8.3.227 -q
pip install tqdm psutil requests pyyaml colorama pillow -q
if %errorlevel% neq 0 (
    echo [警告] 部分依赖安装可能失败，请检查上方输出。
) else (
    echo       ✓ 所有依赖安装完成
)
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
python -c "import torch; print(f'  PyTorch:  {torch.__version__}'); print(f'  CUDA:     {torch.cuda.is_available()}'); print(f'  GPU:      {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
python -c "import cv2; print(f'  OpenCV:   {cv2.__version__}')"
python -c "import ultralytics; print(f'  YOLO:     {ultralytics.__version__}')"
python -c "import flask; print(f'  Flask:    {flask.__version__}')"
echo.

REM ── 检查模型文件 ──
if exist best.pt (
    echo   ✓ 模型文件 best.pt 已找到
) else (
    echo   ⚠ 未找到模型文件！请将 best.pt 放在项目根目录下。
)
echo.

echo ══════════════════════════════════════════════════════════
echo   ✅ 安装完成！
echo.
echo   启动方式:
echo     方式1: 双击 start.bat
echo     方式2: 执行以下命令
echo            venv\Scripts\activate
echo            python app.py
echo.
echo   访问地址: http://127.0.0.1:5000
echo ══════════════════════════════════════════════════════════
pause
