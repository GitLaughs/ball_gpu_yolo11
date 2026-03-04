@echo off
chcp 65001 >nul 2>&1
setlocal EnableDelayedExpansion

title Basketball Shot Detection System - Running

echo  ================================================================
echo   篮球投篮检测系统 启动中...
echo   Basketball Shot Detection System Starting...
echo  ================================================================
echo.

:: 检查虚拟环境
set VENV_DIR=%~dp0.venv
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [错误] 未找到虚拟环境！
    echo        请先运行 install.bat 安装依赖
    pause
    exit /b 1
)

:: 检查 app.py
if not exist "%~dp0app.py" (
    echo [错误] 未找到 app.py！
    echo        请确认你在项目根目录运行此脚本
    pause
    exit /b 1
)

:: 检查模型文件
set MODEL_FOUND=0
if exist "%~dp0best_yolo11.pt" (
    set MODEL_FOUND=1
    echo [OK] 模型文件: best_yolo11.pt
)
if exist "%~dp0best.pt" (
    set MODEL_FOUND=1
    echo [OK] 模型文件: best.pt
)
if !MODEL_FOUND! equ 0 (
    echo [警告] 未找到模型文件 ^(best.pt 或 best_yolo11.pt^)
    echo        系统可能无法正常检测，但 Web 界面仍会启动
)

:: 检查视频文件
set VIDEO_FOUND=0
for %%f in (ball.avi ball.mp4 basketball1.mp4 basketball2.mp4) do (
    if exist "%~dp0%%f" (
        set VIDEO_FOUND=1
        echo [OK] 默认视频: %%f
    )
)
if !VIDEO_FOUND! equ 0 (
    echo [警告] 未找到默认视频文件，请通过 Web 界面上传视频
)

echo.

:: 激活虚拟环境
call "%VENV_DIR%\Scripts\activate.bat"

:: 创建必要目录
if not exist "%~dp0uploads" mkdir "%~dp0uploads"
if not exist "%~dp0exports" mkdir "%~dp0exports"
if not exist "%~dp0logs" mkdir "%~dp0logs"
if not exist "%~dp0static" mkdir "%~dp0static"

:: 切换到项目目录
cd /d "%~dp0"

:: 等待 1 秒后自动打开浏览器
echo [INFO] 服务器将在 http://127.0.0.1:5002 启动
echo [INFO] 2 秒后自动打开浏览器...
echo [INFO] 按 Ctrl+C 停止服务器
echo.
echo ────────────────────────────────────────────────────
echo  服务器日志输出:
echo ────────────────────────────────────────────────────

start "" /min cmd /c "timeout /t 2 >nul && start http://127.0.0.1:5002"

:: 启动 Flask 服务器，日志写入文件同时显示在控制台
python app.py 2>&1 | tee logs\server_%date:~0,4%%date:~5,2%%date:~8,2%.log

if !errorlevel! neq 0 (
    echo.
    echo [错误] 服务器启动失败，错误码: !errorlevel!
    echo        请查看上方错误日志
    echo        详细解决方案：README.md → 故障排除
    pause
)