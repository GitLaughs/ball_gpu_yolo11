# 篮球检测与投篮分析系统

## 项目简介

这是一个基于YOLOv11和OpenCV的篮球检测与投篮分析系统，能够实时检测篮球和篮筐，分析投篮轨迹，并判断投篮是否命中。

## 功能特性

- 实时篮球和篮筐检测
- 3D轨迹分析
- 投篮速度和角度计算
- 投篮命中判断
- Web界面展示

## 系统要求

- Windows 10/11 64位系统
- Python 3.9+
- NVIDIA GPU (推荐，用于加速PyTorch推理)
- CUDA 11.0+ (如果使用GPU)

## 安装步骤

### 方法一：使用一键安装脚本

1. 双击运行 `install_dependencies.bat` 文件
2. 等待安装过程完成
3. 安装完成后，会显示"安装完成！"的信息
4. 按任意键关闭窗口

### 方法二：手动安装

1. 创建虚拟环境：
   ```bash
   python -m venv .venv
   ```

2. 激活虚拟环境：
   ```bash
   .venv\Scripts\activate.bat
   ```

3. 更新pip：
   ```bash
   pip install --upgrade pip
   ```

4. 安装依赖项：
   ```bash
   pip install -r requirements.txt
   ```

## 使用方法

1. 启动服务器：
   ```bash
   .venv\Scripts\python.exe app.py
   ```

2. 打开浏览器访问：
   ```
   http://127.0.0.1:5002
   ```

3. 在Web界面中：
   - 点击"上传视频"按钮选择要分析的视频文件
   - 点击"开始分析"按钮启动视频处理
   - 查看实时分析结果，包括投篮轨迹、速度和角度

## 视频格式要求

- 支持的格式：avi, mp4, mov, mkv, flv, wmv
- 分辨率：1280x480 (左右各640x480的双目拼接视频)

## 项目结构

```
ball_gpu_yolo11/
├── .venv/            # 虚拟环境
├── engine/           # 核心引擎
│   ├── config.py     # 配置文件
│   ├── models.py     # 数据模型
│   ├── shot_engine.py    # 投篮分析引擎
│   └── vision_engine.py  # 视觉处理引擎
├── static/           # 静态文件
│   └── index.html    # Web界面
├── uploads/          # 上传的视频文件
├── app.py            # Flask应用
├── ball.avi          # 示例视频
├── ball.mp4          # 示例视频
├── basketball1.avi   # 示例视频
├── basketball2.avi   # 示例视频
├── convert_videos.py # 视频格式转换脚本
├── install_dependencies.bat # 一键安装脚本
├── requirements.txt  # 依赖项列表
└── trajectory_analyzer.py  # 轨迹分析器
```

## 依赖项

- Flask
- Flask-CORS
- OpenCV
- NumPy
- PyTorch
- Ultralytics (YOLOv11)
- Matplotlib
- Pillow
- PyYAML
- Requests
- SciPy
- TorchVision
- Psutil
- Polars

## 故障排除

### 1. 视频流显示"Waiting for video data"

- 确保点击了"开始分析"按钮启动视频处理
- 检查视频文件是否存在且格式正确
- 检查模型文件是否存在

### 2. GPU加速问题

- 确保已安装正确版本的CUDA和cuDNN
- 检查PyTorch是否正确安装并配置了GPU支持

### 3. 视频文件无法打开

- 确保视频文件格式支持
- 确保视频文件路径正确

## 注意事项

- 本系统使用YOLOv11模型进行目标检测，需要提前下载模型文件
- 模型文件应命名为`best_yolo11.pt`或`best.pt`，并放在项目根目录
- 视频文件应是左右拼接的双目视频，分辨率为1280x480

## 许可证

本项目仅供学习和研究使用。
