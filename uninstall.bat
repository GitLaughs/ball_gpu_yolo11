@echo off
chcp 65001 >nul 2>&1
setlocal EnableDelayedExpansion

title Basketball Shot Detection System - Uninstaller

echo  ================================================================
echo   篮球投篮检测系统 - 卸载程序
echo  ================================================================
echo.
echo  将要删除以下内容：
echo    - .venv^/  （Python 虚拟环境）
echo    - __pycache__^/ （Python 缓存）
echo    - logs^/    （日志文件）
echo.
echo  以下内容将保留：
echo    - 你的代码文件
echo    - 上传的视频（uploads^/）
echo    - 模型文件（best.pt）
echo.
choice /C YN /M "确认卸载？（Y=是，N=取消）"
if !errorlevel! equ 2 (
    echo 取消卸载
    pause
    exit /b 0
)

echo.
echo [1/4] 删除虚拟环境...
if exist "%~dp0.venv" (
    rmdir /s /q "%~dp0.venv"
    echo [OK] .venv 已删除
) else (
    echo [跳过] .venv 不存在
)

echo [2/4] 清理 Python 缓存...
for /d /r "%~dp0" %%d in (__pycache__) do (
    if exist "%%d" rmdir /s /q "%%d"
)
echo [OK] 缓存已清理

echo [3/4] 清理日志文件...
if exist "%~dp0logs" (
    rmdir /s /q "%~dp0logs"
    echo [OK] logs 已删除
)

echo [4/4] 清理临时文件...
del /q "%~dp0*.pyc" >nul 2>&1
echo [OK] 临时文件已清理

echo.
echo [完成] 卸载完成！代码文件和视频已完整保留
pause