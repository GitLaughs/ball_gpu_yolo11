@echo off
chcp 65001 >nul 2>&1
setlocal EnableDelayedExpansion

title 环境检测工具

echo  ================================================================
echo   篮球投篮检测系统 - 环境检测工具
echo  ================================================================
echo.

:: ---- 系统信息 ----
echo [系统信息]
echo   OS: Windows
for /f "tokens=4-5 delims=. " %%i in ('ver') do echo   Version: %%i.%%j
echo   Architecture: %PROCESSOR_ARCHITECTURE%
echo.

:: ---- Python ----
echo [Python 环境]
python --version >nul 2>&1
if !errorlevel! equ 0 (
    python --version
    echo   Path: 
    where python
) else (
    echo   [MISSING] Python 未安装
)
echo.

:: ---- 虚拟环境 ----
echo [虚拟环境]
if exist "%~dp0.venv\Scripts\activate.bat" (
    echo   [OK] 虚拟环境存在: .venv
    call "%~dp0.venv\Scripts\activate.bat"
    python --version
) else (
    echo   [MISSING] 虚拟环境未创建，请运行 install.bat
)
echo.

:: ---- GPU / CUDA ----
echo [GPU / CUDA]
nvidia-smi >nul 2>&1
if !errorlevel! equ 0 (
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
) else (
    echo   未检测到 NVIDIA GPU（将使用 CPU 模式）
)
echo.

:: ---- 项目文件检查 ----
echo [项目文件]
for %%f in (app.py requirements.txt trajectory_analyzer.py) do (
    if exist "%~dp0%%f" (
        echo   [OK] %%f
    ) else (
        echo   [MISSING] %%f
    )
)
echo.

echo [模型文件]
set MODEL_OK=0
for %%f in (best.pt best_yolo11.pt) do (
    if exist "%~dp0%%f" (
        echo   [OK] %%f
        set MODEL_OK=1
    )
)
if !MODEL_OK! equ 0 echo   [MISSING] 未找到模型文件！

echo.
echo [视频文件]
set VID_COUNT=0
for %%f in (ball.avi ball.mp4 basketball1.mp4 basketball2.mp4) do (
    if exist "%~dp0%%f" (
        echo   [OK] %%f
        set /a VID_COUNT+=1
    )
)
if !VID_COUNT! equ 0 echo   [MISSING] 未找到视频文件，请上传或放置视频
echo.

:: ---- 依赖包检查（如果有虚拟环境）----
if exist "%~dp0.venv\Scripts\python.exe" (
    echo [Python 依赖]
    "%~dp0.venv\Scripts\python.exe" -c "
pkgs = ['flask','cv2','torch','numpy','ultralytics','scipy','flask_cors']
labels = {'cv2':'OpenCV'}
for p in pkgs:
    name = labels.get(p, p)
    try:
        m = __import__(p)
        ver = getattr(m, '__version__', 'OK')
        print(f'  [OK] {name}: {ver}')
    except ImportError:
        print(f'  [MISSING] {name}')
"
    echo.
    echo [PyTorch CUDA 状态]
    "%~dp0.venv\Scripts\python.exe" -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'  GPU Memory: {mem:.1f} GB')
"
)

echo.
echo  ================================================================
echo   检测完成！如有问题请参考 README.md 故障排除章节
echo  ================================================================
pause