@echo off
chcp 65001 >nul 2>&1
setlocal EnableDelayedExpansion

:: ============================================================
::  Basketball Shot Detection System - Windows Installer
::  篮球投篮检测系统 - Windows 一键安装脚本
::  Version: 1.0.0
:: ============================================================

title Basketball Shot Detection System - Installer

:: 颜色定义（通过 PowerShell 输出彩色文字）
set "PS=powershell -NoProfile -Command"

call :print_banner
call :check_admin
call :check_system
call :check_python
call :check_git
call :check_cuda
call :create_venv
call :install_requirements
call :verify_install
call :create_shortcuts
call :print_success

pause
exit /b 0

:: ============================================================
::  函数定义
:: ============================================================

:print_banner
echo.
echo  ================================================================
echo   ^|  BASKETBALL SHOT DETECTION SYSTEM                          ^|
echo   ^|  篮球投篮检测系统 - 一键安装程序                           ^|
echo   ^|  Powered by YOLOv11 + Stereo Vision + Flask               ^|
echo  ================================================================
echo.
goto :eof

:check_admin
echo [步骤 1/8] 检查管理员权限...
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo [警告] 当前不是管理员权限，部分功能可能受限
    echo        建议：右键 install.bat，选择"以管理员身份运行"
    echo.
    choice /C YN /M "是否继续安装（Y=继续，N=退出）"
    if !errorlevel! equ 2 exit /b 1
) else (
    echo [OK] 管理员权限确认
)
echo.
goto :eof

:check_system
echo [步骤 2/8] 检查系统环境...
:: 检查 Windows 版本
for /f "tokens=4-5 delims=. " %%i in ('ver') do set VERSION=%%i.%%j
echo [INFO] Windows 版本: %VERSION%
:: 检查磁盘空间（至少需要 10GB）
for /f "tokens=3" %%a in ('dir /-c "%~dp0" ^| find "可用字节"') do set FREE_BYTES=%%a
echo [INFO] 系统架构: %PROCESSOR_ARCHITECTURE%
echo [OK] 系统检查完成
echo.
goto :eof

:check_python
echo [步骤 3/8] 检查 Python 环境...

:: 优先检查 python 命令
python --version >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PY_VER=%%v
    echo [INFO] 检测到 Python !PY_VER!
    
    :: 检查版本是否 >= 3.9
    for /f "tokens=1,2 delims=." %%a in ("!PY_VER!") do (
        set PY_MAJOR=%%a
        set PY_MINOR=%%b
    )
    
    if !PY_MAJOR! lss 3 (
        call :error_exit "Python 版本过低（需要 3.9+），当前: !PY_VER!" "请从 https://www.python.org/downloads/ 下载 Python 3.11"
    )
    if !PY_MAJOR! equ 3 if !PY_MINOR! lss 9 (
        call :error_exit "Python 版本过低（需要 3.9+），当前: !PY_VER!" "请从 https://www.python.org/downloads/ 下载 Python 3.11"
    )
    
    set PYTHON_CMD=python
    echo [OK] Python !PY_VER! 满足要求（>= 3.9）
    goto :python_found
)

:: 尝试 python3
python3 --version >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=python3
    for /f "tokens=2" %%v in ('python3 --version 2^>^&1') do set PY_VER=%%v
    echo [OK] Python3 !PY_VER! 已找到
    goto :python_found
)

:: Python 未找到
echo [错误] 未检测到 Python！
echo.
echo  解决方案：
echo  1. 访问 https://www.python.org/downloads/
echo  2. 下载 Python 3.11.x（推荐）
echo  3. 安装时勾选 "Add Python to PATH"
echo  4. 重新运行此安装脚本
echo.
echo  快速安装（需要 winget）：
echo     winget install Python.Python.3.11
echo.
pause
exit /b 1

:python_found
echo.
goto :eof

:check_git
echo [步骤 4/8] 检查 Git...
git --version >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=3" %%v in ('git --version') do set GIT_VER=%%v
    echo [OK] Git !GIT_VER! 已安装
) else (
    echo [警告] Git 未安装（非必须，但推荐用于更新代码）
    echo  可选安装：winget install Git.Git
    echo  或访问：https://git-scm.com/download/win
)
echo.
goto :eof

:check_cuda
echo [步骤 5/8] 检查 CUDA / GPU 环境...
set CUDA_AVAILABLE=0
set CUDA_VER=None

:: 检查 nvidia-smi
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=9" %%v in ('nvidia-smi --query-gpu^=driver_version --format^=csv^,noheader 2^>nul') do set GPU_DRIVER=%%v
    for /f "tokens=*" %%v in ('nvidia-smi ^| findstr "CUDA Version"') do set CUDA_LINE=%%v
    echo [OK] 检测到 NVIDIA GPU
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>nul
    echo [INFO] !CUDA_LINE!
    set CUDA_AVAILABLE=1
) else (
    echo [警告] 未检测到 NVIDIA GPU 或驱动未安装
    echo        系统将使用 CPU 模式运行（速度较慢）
)
echo.

:: 根据 CUDA 情况设置 PyTorch 安装命令
if !CUDA_AVAILABLE! equ 1 (
    echo [INFO] 将安装 CUDA 版 PyTorch（CUDA 12.1）
    set TORCH_INSTALL=torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
) else (
    echo [INFO] 将安装 CPU 版 PyTorch
    set TORCH_INSTALL=torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
)
echo.
goto :eof

:create_venv
echo [步骤 6/8] 创建 Python 虚拟环境...
set VENV_DIR=%~dp0.venv

if exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [INFO] 检测到已有虚拟环境，跳过创建
    echo        路径: %VENV_DIR%
) else (
    echo [INFO] 正在创建虚拟环境: %VENV_DIR%
    %PYTHON_CMD% -m venv "%VENV_DIR%"
    if !errorlevel! neq 0 (
        call :error_exit "虚拟环境创建失败" "请检查 Python 是否正确安装，或手动运行：python -m venv .venv"
    )
    echo [OK] 虚拟环境创建成功
)

:: 激活虚拟环境
call "%VENV_DIR%\Scripts\activate.bat"
echo [OK] 虚拟环境已激活

:: 升级 pip
echo [INFO] 升级 pip...
python -m pip install --upgrade pip setuptools wheel >nul 2>&1
echo [OK] pip 已升级

echo.
goto :eof

:install_requirements
echo [步骤 7/8] 安装项目依赖...
echo.

set VENV_PIP=%~dp0.venv\Scripts\pip.exe

:: --- 第一步：安装 PyTorch ---
echo [7.1] 安装 PyTorch...
echo      命令: pip install %TORCH_INSTALL%
echo      （此步骤可能需要 5-15 分钟，请耐心等待...）
echo.
"%VENV_PIP%" install %TORCH_INSTALL%
if !errorlevel! neq 0 (
    echo [警告] PyTorch 安装失败，尝试 CPU 版本...
    "%VENV_PIP%" install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
    if !errorlevel! neq 0 (
        call :error_exit "PyTorch 安装失败" "请检查网络连接后重试，或手动运行: pip install torch"
    )
)
echo [OK] PyTorch 安装完成

:: --- 第二步：安装核心依赖 ---
echo.
echo [7.2] 安装核心依赖...
"%VENV_PIP%" install ^
    Flask==3.1.0 ^
    Flask-Cors==4.0.0 ^
    opencv-contrib-python==4.10.0.84 ^
    numpy==1.26.4 ^
    ultralytics==8.3.227
if !errorlevel! neq 0 (
    call :error_exit "核心依赖安装失败" "请检查网络连接，或使用国内镜像：pip install -i https://pypi.tuna.tsinghua.edu.cn/simple"
)
echo [OK] 核心依赖安装完成

:: --- 第三步：安装辅助依赖 ---
echo.
echo [7.3] 安装辅助依赖...
"%VENV_PIP%" install ^
    scipy==1.16.3 ^
    matplotlib==3.9.2 ^
    psutil==7.1.3 ^
    tqdm==4.67.1 ^
    PyYAML==6.0.3 ^
    requests==2.32.5 ^
    Pillow==11.1.0 ^
    Werkzeug==3.1.3
if !errorlevel! neq 0 (
    echo [警告] 部分辅助依赖安装失败，但不影响核心功能
)
echo [OK] 辅助依赖安装完成

:: --- 第四步：安装 CuPy（可选，仅 CUDA）---
if !CUDA_AVAILABLE! equ 1 (
    echo.
    echo [7.4] 安装 CuPy（GPU 加速，可选）...
    "%VENV_PIP%" install cupy-cuda12x >nul 2>&1
    if !errorlevel! equ 0 (
        echo [OK] CuPy 安装成功
    ) else (
        echo [警告] CuPy 安装失败（不影响运行，仅跳过 GPU 加速优化）
    )
)

echo.
echo [OK] 所有依赖安装完成
echo.
goto :eof

:verify_install
echo [步骤 8/8] 验证安装...
set VENV_PYTHON=%~dp0.venv\Scripts\python.exe

:: 验证核心包
set VERIFY_SCRIPT=^
import sys; ^
errors = []; ^
^
try: import flask; print(f'  [OK] Flask {flask.__version__}') ^
except: errors.append('Flask'); print('  [FAIL] Flask'); ^
^
try: import cv2; print(f'  [OK] OpenCV {cv2.__version__}') ^
except: errors.append('OpenCV'); print('  [FAIL] OpenCV'); ^
^
try: import torch; print(f'  [OK] PyTorch {torch.__version__}'); print(f'  [INFO] CUDA available: {torch.cuda.is_available()}') ^
except: errors.append('PyTorch'); print('  [FAIL] PyTorch'); ^
^
try: import numpy; print(f'  [OK] NumPy {numpy.__version__}') ^
except: errors.append('NumPy'); print('  [FAIL] NumPy'); ^
^
try: import ultralytics; print(f'  [OK] Ultralytics {ultralytics.__version__}') ^
except: errors.append('ultralytics'); print('  [FAIL] Ultralytics'); ^
^
print(); ^
if errors: print(f'缺少包: {errors}'); sys.exit(1) ^
else: print('所有核心依赖验证通过！')

"%VENV_PYTHON%" -c "%VERIFY_SCRIPT%"

if !errorlevel! neq 0 (
    echo.
    echo [警告] 部分依赖验证失败，请查看上方错误信息
    echo        尝试手动安装缺少的包后再次运行
) else (
    echo [OK] 安装验证通过
)
echo.
goto :eof

:create_shortcuts
echo [快捷方式] 创建运行脚本...

:: 创建 start.bat（如果不存在）
if not exist "%~dp0start.bat" (
    echo 已有 start.bat，跳过
)

:: 创建 check_model.bat 用于检查模型文件
if not exist "%~dp0check_model.bat" (
(
echo @echo off
echo chcp 65001 ^>nul
echo call "%~dp0.venv\Scripts\activate.bat"
echo python -c "from ultralytics import YOLO; m=YOLO('best.pt'); print('Model OK')"
echo pause
) > "%~dp0check_model.bat"
)

echo [OK] 快捷脚本已准备完成
echo.
goto :eof

:print_success
echo.
echo  ================================================================
echo   安装完成！Installation Complete!
echo  ================================================================
echo.
echo  下一步操作：
echo  ─────────────────────────────────────────────────────────────
echo  1. 确认模型文件存在：  best.pt（或 best_yolo11.pt）
echo  2. 确认视频文件存在：  ball.avi / ball.mp4 / basketball1.mp4
echo  3. 启动系统：          双击 start.bat
echo  4. 打开浏览器：        http://127.0.0.1:5002
echo  ─────────────────────────────────────────────────────────────
echo.
echo  如有问题，请查看 README.md 中的故障排除章节
echo.
goto :eof

:error_exit
echo.
echo  ================================================================
echo   [错误] %~1
echo  ================================================================
echo   解决方案: %~2
echo  ================================================================
echo.
pause
exit /b 1