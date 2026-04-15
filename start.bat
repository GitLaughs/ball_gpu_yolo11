@REM filepath: start.bat
@echo off
chcp 65001 >nul 2>&1
title 🏀 AI 篮球投篮分析系统
color 0E

echo ══════════════════════════════════════════════════════════
echo          🏀 AI 篮球投篮分析系统 - 启动中...
echo ══════════════════════════════════════════════════════════
echo.

REM ── 检查虚拟环境 ──
if not exist venv\Scripts\activate.bat (
    echo [错误] 未找到虚拟环境！请先运行 install_gpu.bat 或 install_cpu.bat 进行安装。
    pause
    exit /b 1
)

REM ── 激活虚拟环境 ──
call venv\Scripts\activate.bat

REM ── 检查模型文件 ──
if exist best.pt (
    echo   ✓ 使用模型: best.pt
) else (
    echo [错误] 未找到模型文件 best.pt！
    echo        请将 YOLO 模型权重文件放在项目根目录下。
    pause
    exit /b 1
)

REM ── 检查视频 ──
if exist ball.avi (
    echo   ✓ 默认视频: ball.avi
) else (
    echo   ℹ 未找到 ball.avi，可通过 Web 界面上传视频。
)
echo.

REM ── 显示设备信息 ──
python -c "import torch; d='CUDA: '+torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'; print(f'  运行设备: {d}')" 2>nul
echo.

echo ──────────────────────────────────────────────────────────
echo   启动 Web 服务器...
echo   浏览器访问: http://127.0.0.1:5000
echo   按 Ctrl+C 停止服务
echo ──────────────────────────────────────────────────────────
echo.

REM ── 自动打开浏览器 ──
start "" http://127.0.0.1:5000

REM ── 启动应用 ──
python app.py

echo.
echo   服务已停止。
pause
