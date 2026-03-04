@echo off

echo 正在创建虚拟环境...
python -m venv .venv

echo 正在激活虚拟环境...
.venv\Scripts\activate.bat

echo 正在更新pip...
.venv\Scripts\pip.exe install --upgrade pip

echo 正在安装依赖项...
.venv\Scripts\pip.exe install -r requirements.txt

echo 安装完成！
pause
