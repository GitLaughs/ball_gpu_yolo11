<div align="center">

# 🏀 Basketball Shot Detection System
### 篮球投篮检测系统

**双目立体视觉 × YOLOv11 × 实时 3D 轨迹分析**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.1.0-green?logo=flask)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-orange?logo=pytorch)
![YOLO](https://img.shields.io/badge/YOLOv11-ultralytics-purple)
![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900?logo=nvidia)
![OpenCV](https://img.shields.io/badge/OpenCV-4.10-blue?logo=opencv)
![License](https://img.shields.io/badge/License-MIT-yellow)

</div>

---

## 📖 目录

- [🎯 项目介绍](#-项目介绍)
- [✨ 设计亮点](#-设计亮点)
- [🏗️ 系统架构](#️-系统架构)
- [📁 目录结构](#-目录结构)
- [⚙️ 环境要求](#️-环境要求)
- [🚀 一键安装](#-一键安装)
- [▶️ 启动运行](#️-启动运行)
- [🖥️ Web 界面使用](#️-web-界面使用)
- [🔌 API 接口文档](#-api-接口文档)
- [🧩 模块详解](#-模块详解)
- [🔧 配置说明](#-配置说明)
- [🐛 故障排除](#-故障排除)
- [📊 性能参考](#-性能参考)

---

## 🎯 项目介绍

本系统是一套**实时篮球投篮检测与分析平台**，通过双目摄像头（或双目合拍视频）获取三维空间信息，利用 YOLOv11 检测篮球和篮筐，结合自研的鲁棒轨迹分析算法，实时计算：

| 指标       | 说明                                         |
| ---------- | -------------------------------------------- |
| 🎯 进球判定 | 基于 3D 空间穿越算法，检测球是否通过篮圈平面 |
| 💨 出手速度 | 通过 3D 轨迹时序回归精确估算（m/s）          |
| 📐 出手角度 | X-Z 平面内的初始飞行角度（°）                |
| 📈 飞行轨迹 | 鲁棒加权二次曲线拟合 + 预测延伸              |
| 📍 3D 坐标  | 球和篮筐的实时三维位置（m）                  |

---

## ✨ 设计亮点

### 1. 🔭 双目立体视觉深度感知（VisionEngine）

```
左摄像头帧 ─┐
            ├─► 立体校正(remap) ─► SGBM 视差计算 ─► reprojectImageTo3D ─► 3D 点云
右摄像头帧 ─┘
```

- 使用 **StereoSGBM** 三通道模式（`STEREO_SGBM_MODE_SGBM_3WAY`）计算稠密视差
- 通过预标定的内参/外参矩阵做双目校正，确保极线对齐
- 在每个检测框中心取 **5×5 中位数深度窗口**（`sample_window=5`），有效抑制视差噪声
- 视差图乘以 16，然后 `reprojectImageTo3D` 获得真实米制坐标

### 2. 🎯 YOLOv11 目标检测（自训练模型）

- 使用 `best.pt` / `best_yolo11.pt` 自训练 YOLO 模型
- 双目视频输入为 **左右拼接（side-by-side）** 格式，宽度 = 2× 单眼宽度
- 仅对**左摄像头帧**做目标检测，右摄像头用于深度计算，节省推理时间
- 类别 ID 映射：`0 → hoop`（篮筐），`1 → basketball`（篮球）

### 3. 🏀 智能投篮状态机（ShotEngine）

```
检测到球 → [ACTIVE 状态] → 积累点序列（最多80帧）
    ↓
球消失超过 0.35s → 触发分析
    ↓
TrajectoryAnalyzer 分析 → ShotResult
    ↓
进球判定：3D 穿越检测（球轨迹是否穿过篮圈高度平面且在篮圈半径内）
```

**篮筐位置缓冲机制**：即使篮筐在关键帧未被检测到，系统会保留最近 60 帧的篮筐位置用于进球判定，大幅提升鲁棒性。

### 4. 📐 鲁棒轨迹分析算法（TrajectoryAnalyzer）

这是本系统的**核心算法亮点**，完全自实现（不依赖 SciPy）：

```
原始点序列
    │
    ├─ 1. 去重复点（L1距离 < 1e-6）
    ├─ 2. MAD 离群点去除（Z方向 + 径向，4.5σ/5.0σ 阈值）
    ├─ 3. Savitzky-Golay 平滑（自实现，7点2阶多项式）
    ├─ 4. IRLS 鲁棒加权二次拟合（Huber权重，8次迭代）
    └─ 5. 速度/角度估计（末尾8点线性回归）
```

| 算法组件       | 作用                        | 参数                   |
| -------------- | --------------------------- | ---------------------- |
| MAD 去噪       | 去除跳变噪声点              | `outlier_mad_z=4.5`    |
| Savitzky-Golay | 平滑轨迹，保持峰值          | `window=7, poly=2`     |
| IRLS + Huber   | 对残余噪声鲁棒的曲线拟合    | `huber_k=1.5, iters=8` |
| 线性回归估速   | 用末尾K点回归，比差分更稳定 | `K=8`                  |

### 5. 🌐 实时 Web 可视化（Flask + MJPEG）

- 三路 MJPEG 视频流：左摄像头（带检测框）、右摄像头、深度图（JET 伪彩）
- 通过 `/api/status` 轮询实时更新系统状态（FPS、帧数、GPU 显存）
- 支持视频上传（6种格式），上传后立即切换视频源
- 投篮历史记录、轨迹图表（X-Z 平面抛物线可视化）

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    Web Browser                          │
│         http://127.0.0.1:5002                          │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐              │
│   │Left Cam  │ │Right Cam │ │Depth View│  MJPEG流     │
│   └──────────┘ └──────────┘ └──────────┘              │
│   ┌─────────────────────────────────────┐              │
│   │     Shot Analysis / Status          │  REST API   │
│   └─────────────────────────────────────┘              │
└─────────────────────┬───────────────────────────────────┘
                      │ HTTP
┌─────────────────────▼───────────────────────────────────┐
│                 Flask App (app.py)                       │
│  ┌─────────────┐  AppState  ┌──────────────────────┐   │
│  │  /api/start │ ────────── │  detection_loop()    │   │
│  │  /api/pause │            │  (daemon thread)     │   │
│  │  /api/stop  │            └──────────────────────┘   │
│  │  /api/status│                      │                 │
│  └─────────────┘                      │                 │
└──────────────────────────────────────-│─────────────────┘
                                        │
            ┌───────────────────────────┤
            │                           │
┌───────────▼──────────┐   ┌───────────▼──────────┐
│   VisionEngine        │   │    ShotEngine         │
│  ─────────────────   │   │  ─────────────────    │
│  双目校正（remap）    │   │  状态机（is_active）  │
│  SGBM 视差计算       │──►│  轨迹点积累           │
│  3D 重建             │   │  超时检测             │
│  YOLOv11 推理        │   │  进球判定             │
│  3D 坐标提取         │   └───────────┬──────────┘
└──────────────────────┘               │
                            ┌──────────▼──────────┐
                            │  TrajectoryAnalyzer  │
                            │  ─────────────────  │
                            │  去重 → 去噪 → 平滑 │
                            │  → 鲁棒拟合 → 预测  │
                            └─────────────────────┘
```

---

## 📁 目录结构

```
basketball-detection/
│
├── 📄 app.py                    # Flask 主程序，API 路由，检测主循环
├── 📄 trajectory_analyzer.py    # 鲁棒轨迹分析核心算法
├── 📄 requirements.txt          # Python 依赖列表
│
├── 🚀 install.bat               # Windows 一键安装脚本
├── ▶️  start.bat                 # 一键启动脚本
├── 🗑️  uninstall.bat             # 卸载脚本
├── 🔍 check_env.bat             # 环境检测工具
│
├── engine/                      # 核心引擎模块
│   ├── 📄 __init__.py
│   ├── 📄 config.py             # EngineConfig 配置数据类
│   ├── 📄 models.py             # 数据模型（Detection3D, ShotResult 等）
│   ├── 📄 vision_engine.py      # 视觉引擎（双目 + YOLO）
│   └── 📄 shot_engine.py        # 投篮分析引擎（状态机）
│
├── static/
│   └── 📄 index.html            # Web 前端界面（单页应用）
│
├── 📹 ball.mp4                  # 默认测试视频（双目拼接格式）
├── 📹 basketball1.mp4           # 测试视频 1
├── 📹 basketball2.mp4           # 测试视频 2
│
├── 🤖 best.pt                   # YOLOv11 训练模型（必须）
│   (或 best_yolo11.pt)
│
├── uploads/                     # 用户上传视频目录（自动创建）
├── exports/                     # 导出数据目录（自动创建）
├── logs/                        # 运行日志目录（自动创建）
│
└── .venv/                       # Python 虚拟环境（安装后生成）
```

---

## ⚙️ 环境要求

### 必须条件

| 组件         | 最低要求        | 推荐配置        |
| ------------ | --------------- | --------------- |
| **操作系统** | Windows 10 64位 | Windows 11 64位 |
| **Python**   | 3.9             | 3.11.x          |
| **RAM**      | 8 GB            | 16 GB+          |
| **存储**     | 10 GB 可用空间  | 20 GB+          |
| **CPU**      | 4 核            | 8 核+           |

### GPU 加速（强烈推荐）

| 组件     | 要求                             |
| -------- | -------------------------------- |
| **GPU**  | NVIDIA GPU（GTX 1060 及以上）    |
| **VRAM** | 4 GB+（推荐 8 GB+）              |
| **CUDA** | 12.1+（由 install.bat 自动匹配） |
| **驱动** | NVIDIA Driver 525.0+             |

> ⚠️ **CPU 模式警告**：无 GPU 时系统可运行但 FPS 极低（<5 FPS），不适合实时检测

### 软件依赖

| 软件          | 用途     | 下载                                            |
| ------------- | -------- | ----------------------------------------------- |
| Python 3.11   | 运行环境 | [python.org](https://www.python.org/downloads/) |
| Git（可选）   | 代码更新 | [git-scm.com](https://git-scm.com/download/win) |
| NVIDIA Driver | GPU 支持 | [nvidia.com](https://www.nvidia.com/drivers)    |

---

## 🚀 一键安装

### 方法一：双击安装（推荐）

```
1. 将项目文件夹放到任意位置（路径不要包含中文或空格）
2. 右键点击 install.bat
3. 选择"以管理员身份运行"
4. 按照提示等待安装完成（约 10-20 分钟）
```

### 方法二：命令行安装

```cmd
# 以管理员身份打开 CMD 或 PowerShell
cd C:\path\to\basketball-detection

# 运行安装脚本
install.bat

# 安装完成后验证环境
check_env.bat
```

### 安装过程说明

```
[步骤 1] 检查管理员权限
[步骤 2] 检查 Windows 系统信息
[步骤 3] 检查 Python 版本（必须 ≥ 3.9）
[步骤 4] 检查 Git 安装
[步骤 5] 检查 CUDA/GPU 环境 → 决定安装 GPU 版或 CPU 版 PyTorch
[步骤 6] 创建 Python 虚拟环境（.venv/）
[步骤 7] 安装所有依赖（分批安装，出错不会中断）
[步骤 8] 验证安装结果
```

### 手动安装（install.bat 失败时）

```cmd
# 1. 创建虚拟环境
python -m venv .venv

# 2. 激活虚拟环境
.venv\Scripts\activate

# 3. 升级 pip
pip install --upgrade pip

# 4. 安装 PyTorch（CUDA 版）
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 ^
    --index-url https://download.pytorch.org/whl/cu121

# 4b. 或者安装 CPU 版（无 GPU 时）
pip install torch==2.5.1 torchvision==0.20.1

# 5. 安装其余依赖
pip install Flask==3.1.0 Flask-Cors==4.0.0 opencv-contrib-python==4.10.0.84 ^
    ultralytics==8.3.227 numpy==1.26.4 scipy==1.16.3 Pillow==11.1.0

# 6. 使用国内镜像（网速慢时）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## ▶️ 启动运行

### 快速启动

```
双击 start.bat
→ 自动激活虚拟环境
→ 检查模型和视频文件
→ 启动 Flask 服务器
→ 自动打开浏览器 http://127.0.0.1:5002
```

### 命令行启动

```cmd
# 激活虚拟环境
.venv\Scripts\activate

# 启动服务器
python app.py
```

### 启动成功标志

```
Server: http://127.0.0.1:5002
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5002
```

---

## 🖥️ Web 界面使用

### 主界面布局

```
┌─────────────────────────────────────────────────┐
│  🏀 Basketball Shot Detection     [Start][Stop]  │
├────────────┬────────────┬────────────────────────┤
│ 左摄像头   │ 右摄像头   │   📊 系统状态          │
│  LIVE ●   │  LIVE ●   │   FPS / 帧数 / GPU    │
├────────────┴────────────┤   🏀 检测计数         │
│      深度伪彩视图        ├───────────────────────┤
│      LIVE ●            │   🎯 投篮分析          │
│                        │   速度 / 角度 / 坐标   │
├─────────────────────────┤   轨迹图表（X-Z）     │
│  📋 投篮历史记录         └───────────────────────┘
└─────────────────────────────────────────────────┘
```

### 操作步骤

```
1. ▶ Start     → 开始视频处理和检测
2. ⏸ Pause    → 暂停（不断开，可继续）
3. ⏹ Stop     → 停止并释放 GPU 显存
4. 📁 Upload   → 上传本地视频文件（支持 mp4/avi/mov/mkv）
```

### 视频要求

> ⚠️ 本系统使用**双目拼接格式**视频：

```
原始帧格式：
┌───────────────┬───────────────┐
│   左摄像头     │   右摄像头    │
│   640 × 480   │   640 × 480  │
│               │               │
└───────────────┴───────────────┘
          总计: 1280 × 480
```

---

## 🔌 API 接口文档

| 接口                    | 方法 | 说明               |
| ----------------------- | ---- | ------------------ |
| `/api/start`            | POST | 开始检测           |
| `/api/pause`            | POST | 暂停/恢复检测      |
| `/api/stop`             | POST | 停止检测，释放资源 |
| `/api/status`           | GET  | 获取实时状态       |
| `/api/shot_analysis`    | GET  | 获取最新投篮分析   |
| `/api/upload`           | POST | 上传视频文件       |
| `/api/video_feed/left`  | GET  | 左摄像头 MJPEG 流  |
| `/api/video_feed/right` | GET  | 右摄像头 MJPEG 流  |
| `/api/video_feed/depth` | GET  | 深度图 MJPEG 流    |

### `/api/status` 响应示例

```json
{
  "success": true,
  "is_running": true,
  "is_paused": false,
  "device": "NVIDIA GeForce RTX 3080",
  "cuda_version": "12.1",
  "stats": {
    "fps": 28.5,
    "current_frame": 1520,
    "total_frames": 3600,
    "process_time": 35.2,
    "gpu_memory": 1.82,
    "basketball_count": 1,
    "hoops_count": 1,
    "current_video": "basketball1.mp4"
  }
}
```

### `/api/shot_analysis` 响应示例

```json
{
  "is_scored": true,
  "shot_speed": 7.23,
  "shot_angle": 52.4,
  "shot_position": {"x": 1.2, "y": 1.8, "z": 3.5},
  "hoop_position": {"x": 0.1, "y": 3.05, "z": 6.1},
  "is_active_shot": false,
  "sequence_length": 34,
  "predicted_trajectory": [[0.2, 3.1], [0.5, 3.8], ...],
  "actual_trajectory": [[1.2, 3.5], [0.9, 4.1], ...],
  "trajectory_analysis": {
    "fit": {
      "coef": {"a": -0.42, "b": 1.85, "c": 0.31},
      "rmse": 0.042
    },
    "actual": {"velocity": 7.23, "angle": 52.4}
  }
}
```

---

## 🧩 模块详解

### `engine/config.py` — 统一配置

```python
@dataclass
class EngineConfig:
    conf_threshold: float = 0.5      # YOLO 置信度阈值
    iou_threshold: float = 0.45      # NMS IoU 阈值
    width: int = 640                 # 单目宽度
    height: int = 480                # 单目高度
    sample_window: int = 5           # 深度采样窗口（奇数）
    hoop_radius_m: float = 0.15      # 篮圈半径（米）
    min_points: int = 8              # 轨迹最少采样点
    max_points: int = 80             # 轨迹最多保留点
    lost_timeout_s: float = 0.35     # 球消失超时（秒）
    require_downward: bool = True    # 进球需要向下速度
```

### `engine/models.py` — 数据结构

| 类                | 字段                                | 说明                  |
| ----------------- | ----------------------------------- | --------------------- |
| `Detection3D`     | cls, conf, bbox, xyz, distance      | 单目标检测结果+3D坐标 |
| `FrameDetections` | frame_id, timestamp, detections     | 单帧所有检测结果      |
| `ShotResult`      | is_scored, speed, angle, trajectory | 完整投篮分析结果      |
| `ShotState`       | is_active, points, hoop_history     | 投篮状态机内部状态    |

---

## 🔧 配置说明

### 调整检测灵敏度

编辑 `engine/config.py`：

```python
# 降低误检（提高阈值，减少虚警）
conf_threshold: float = 0.65

# 放宽进球判定（适合广角摄像头）
hoop_radius_m: float = 0.23
hoop_radius_tol: float = 1.5

# 短视频场景（减少最少点数）
min_points: int = 5

# 高速摄像机（减少超时时间）
lost_timeout_s: float = 0.2
```

### 修改服务端口

编辑 `app.py` 最后一行：

```python
app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)
#                        ^^^^  修改这里
```

### 自定义双目标定参数

编辑 `app.py` 中 `build_stereo_params()` 函数，替换你自己的标定矩阵：

```python
left_camera_matrix = np.array([[fx, 0, cx],
                                [0, fy, cy],
                                [0,  0,  1]])
# T 单位为 mm（毫米）
T = np.array([-基线距离_mm, ty, tz])
```

---

## 🐛 故障排除

### ❌ 错误 1：`python` 不是内部命令

**症状**：
```
'python' 不是内部或外部命令，也不是可运行的程序
```

**解决方案**：

```
方法1：重新安装 Python，安装时务必勾选 "Add Python to PATH"

方法2：手动添加 PATH
  1. 按 Win+X 选择"系统"
  2. 点击"高级系统设置" → "环境变量"
  3. 在"系统变量"中找到 Path，添加：
     C:\Users\你的用户名\AppData\Local\Programs\Python\Python311\
     C:\Users\你的用户名\AppData\Local\Programs\Python\Python311\Scripts\

方法3：使用完整路径
  C:\Users\你的用户名\AppData\Local\Programs\Python\Python311\python.exe app.py
```

---

### ❌ 错误 2：PyTorch 安装失败 / 网络超时

**症状**：
```
ERROR: Could not install packages due to an OSError
ReadTimeoutError / ConnectionError
```

**解决方案**：

```cmd
# 方法1：使用清华镜像（推荐国内用户）
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 ^
    -i https://pypi.tuna.tsinghua.edu.cn/simple ^
    --index-url https://download.pytorch.org/whl/cu121

# 方法2：先安装 CPU 版（验证环境），再换 GPU 版
pip install torch torchvision

# 方法3：离线安装
# 1. 在有网络的机器从 https://download.pytorch.org/whl/cu121 下载 .whl 文件
# 2. 复制到目标机器
# 3. pip install torch-2.5.1+cu121-cp311-cp311-win_amd64.whl

# 方法4：增大 pip 超时时间
pip install torch --timeout=600 --retries=5
```

---

### ❌ 错误 3：`torch.cuda.is_available()` 返回 `False`

**症状**：界面显示 `Device: CPU`，GPU 闲置

**诊断步骤**：

```python
# 在虚拟环境中运行
import torch
print(torch.__version__)          # 应包含 +cu121
print(torch.cuda.is_available())  # 应为 True
print(torch.version.cuda)         # 应为 12.1
```

**解决方案**：

```
原因1：安装的是 CPU 版 PyTorch
  → pip uninstall torch torchvision torchaudio
  → pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

原因2：NVIDIA 驱动版本过低（需 ≥ 525.0）
  → 下载最新驱动：https://www.nvidia.com/drivers
  → 安装后重启系统

原因3：CUDA Toolkit 版本不匹配
  → 检查：nvcc --version
  → PyTorch 2.5.1+cu121 需要 CUDA 12.1+
  → 下载：https://developer.nvidia.com/cuda-downloads

原因4：多 GPU 系统，CUDA 设备 ID 错误
  → 在 app.py 中修改：DEVICE = "cuda:1"（尝试其他 GPU）
```

---

### ❌ 错误 4：`ModuleNotFoundError: No module named 'cv2'`

**症状**：
```
ModuleNotFoundError: No module named 'cv2'
```

**解决方案**：

```cmd
# 激活虚拟环境后安装
.venv\Scripts\activate

# 方法1：安装 contrib 版（包含额外算法）
pip install opencv-contrib-python==4.10.0.84

# 方法2：如果上面失败
pip install opencv-python==4.12.0.88

# 注意：不要同时安装两个版本！
pip uninstall opencv-python opencv-contrib-python  # 先卸载
pip install opencv-contrib-python==4.10.0.84       # 再安装

# 验证
python -c "import cv2; print(cv2.__version__)"
```

---

### ❌ 错误 5：视频无法打开 / `cv2.VideoCapture failed`

**症状**：
```
[ERROR] cv2.VideoCapture failed to open: ./ball.avi
[ERROR] Video width 640 < 1280. process_frame expects side-by-side stereo!
```

**解决方案**：

```
原因1：视频文件不存在
  → 确认文件位于项目根目录
  → check_env.bat 会列出所有检测到的视频文件

原因2：视频格式不是双目拼接格式（宽度应为 1280）
  → 使用 convert_videos.py 转换（如果项目包含此文件）
  → 或将两路视频用 ffmpeg 合并：
    ffmpeg -i left.mp4 -i right.mp4 \
      -filter_complex hstack \
      -c:v libx264 stereo_combined.mp4

原因3：视频编解码器缺失
  → 安装 K-Lite Codec Pack：https://codecguide.com/download_kl.htm
  → 或安装 LAV Filters
  
原因4：路径包含中文
  → 将项目移到纯英文路径，例如 C:\Projects\basketball-detection\
```

---

### ❌ 错误 6：`Model file not found: ./best.pt`（续）

**解决方案**：

```
1. 确认模型文件在项目根目录：
   C:\Projects\basketball-detection\best.pt
   C:\Projects\basketball-detection\best_yolo11.pt

2. 如果模型文件名不同，修改 app.py：
   MODEL_PATHS = [
       "./best.pt",
       "./best_yolo11.pt",
       "./your_model_name.pt",  ← 添加你的模型文件名
   ]

3. 用 YOLOv11 默认模型临时测试（会自动下载）：
   python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

4. 检查文件是否损坏：
   python -c "
   import torch
   model = torch.load('./best.pt', map_location='cpu')
   print('Model loaded OK')
   "
```

---

### ❌ 错误 7：端口 5002 被占用

**症状**：
```
OSError: [Errno 98] Address already in use: ('0.0.0.0', 5002)
[WinError 10048] 通常每个套接字地址只能使用一次
```

**解决方案**：

```cmd
:: 查看占用端口的进程
netstat -ano | findstr :5002

:: 结果示例（最后一列是 PID）：
:: TCP  0.0.0.0:5002  0.0.0.0:0  LISTENING  12345

:: 杀死进程
taskkill /PID 12345 /F

:: 或者直接杀死所有 python 进程
taskkill /F /IM python.exe

:: 或者修改端口（app.py 最后一行）
app.run(host="0.0.0.0", port=5003)
:: 然后访问 http://127.0.0.1:5003
```

---

### ❌ 错误 8：`ImportError: DLL load failed`（Windows）

**症状**：
```
ImportError: DLL load failed while importing cv2: 找不到指定的模块
ImportError: DLL load failed while importing torch_cuda
```

**解决方案**：

```
原因1：Visual C++ 运行时缺失
  → 下载安装：https://aka.ms/vs/17/release/vc_redist.x64.exe
  → 安装后重启计算机

原因2：PyTorch DLL 与 CUDA 版本不匹配
  → 重新安装匹配版本的 PyTorch：
    pip uninstall torch torchvision torchaudio
    pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

原因3：杀毒软件拦截 DLL 加载
  → 将项目目录添加到杀毒软件白名单
  → 临时关闭实时保护后重试

原因4：路径中有空格或特殊字符
  → 将项目移到 C:\Projects\basketball-detection\
  → 路径不要包含中文、空格、括号等
```

---

### ❌ 错误 9：`numpy` 版本冲突

**症状**：
```
AttributeError: module 'numpy' has no attribute 'bool'
RuntimeError: module compiled against API version 0x10 but this version of numpy is 0xf
```

**解决方案**：

```cmd
:: NumPy 2.x 与部分包不兼容，回退到 1.x
pip uninstall numpy -y
pip install numpy==1.26.4

:: 验证
python -c "import numpy; print(numpy.__version__)"
:: 应输出 1.26.4

:: 如果 ultralytics 报错，同步更新
pip install ultralytics==8.3.227 --upgrade
```

---

### ❌ 错误 10：Web 界面打开但视频流黑屏

**症状**：
- 浏览器能访问 http://127.0.0.1:5002
- 点击 Start 后视频区域全黑
- 控制台无报错

**诊断与解决**：

```
步骤1：检查服务器日志
  → 查看 logs/ 目录下最新的日志文件
  → 或在 CMD 窗口查看实时输出

步骤2：检查视频文件分辨率
  python -c "
  import cv2
  cap = cv2.VideoCapture('./ball.mp4')
  w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
  h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
  fps = cap.get(cv2.CAP_PROP_FPS)
  print(f'Resolution: {w}x{h}, FPS: {fps}')
  "
  → 宽度应为 1280（双目拼接），高度应为 480

步骤3：检查 MJPEG 流是否正常
  → 直接在浏览器访问：http://127.0.0.1:5002/api/video_feed/left
  → 如果显示图像则 Flask 正常，问题在前端 JS

步骤4：清除浏览器缓存
  → 按 Ctrl+Shift+Del 清除缓存
  → 或用无痕模式打开 http://127.0.0.1:5002

步骤5：换用其他浏览器
  → 推荐 Chrome 或 Edge（最新版）
  → Safari 对 MJPEG 支持较弱
```

---

### ❌ 错误 11：CUDA 内存不足（OOM）

**症状**：
```
torch.cuda.OutOfMemoryError: CUDA out of memory.
RuntimeError: CUDA error: out of memory
```

**解决方案**：

```python
# 方法1：在 app.py 中减小推理尺寸（engine/config.py）
imgsz: int = 416    # 从 640 降低到 416 或 320

# 方法2：强制使用 CPU（临时方案）
DEVICE = "cpu"      # 在 app.py 顶部修改

# 方法3：清理 CUDA 缓存（在检测循环中定期执行）
import torch
torch.cuda.empty_cache()

# 方法4：关闭其他占用 GPU 内存的程序
# → 关闭浏览器标签页（部分占用 GPU）
# → 关闭其他 AI/游戏程序

# 方法5：查看 GPU 显存占用
nvidia-smi
```

---

### ❌ 错误 12：`scipy` 或 `ultralytics` 安装报错

**症状**：
```
ERROR: Could not build wheels for scipy
error: Microsoft Visual C++ 14.0 or greater is required
```

**解决方案**：

```
原因：Windows 缺少 C++ 编译器

方法1：安装 Build Tools（推荐）
  → 下载：https://visualstudio.microsoft.com/visual-cpp-build-tools/
  → 安装"C++ build tools"工作负载
  → 重启后重试 pip install scipy

方法2：使用预编译 wheel
  → pip install scipy --prefer-binary
  → pip install scipy==1.16.3 --only-binary=:all:

方法3：conda 安装（如果有 Anaconda）
  → conda install scipy
  → conda install -c conda-forge ultralytics
```

---

### ❌ 错误 13：系统 FPS 极低（< 5 FPS）

**性能调优方案**：

```python
# 在 engine/config.py 中调整以下参数：

# 1. 降低推理分辨率（最显著提速）
imgsz: int = 320          # 从 640 → 320（速度翻倍，精度略降）

# 2. 使用半精度推理（需要 GPU）
half_precision: bool = True   # FP16 推理，速度提升 30-50%

# 3. 跳帧处理
skip_frames: int = 2          # 每3帧处理1帧

# 4. 关闭深度图渲染（如果不需要）
enable_depth_view: bool = False

# 5. 降低 SGBM 精度（深度计算更快）
sgbm_num_disp: int = 64       # 从 128 → 64
sgbm_block_size: int = 11     # 从 9 → 11（较大窗口更快）
```

```cmd
:: 确认 GPU 被正确使用
nvidia-smi -l 1
:: 观察 GPU 利用率，应在 50% 以上
```

---

### ❌ 错误 14：`flask_cors` 导入失败

**症状**：
```
ModuleNotFoundError: No module named 'flask_cors'
```

**解决方案**：

```cmd
.venv\Scripts\activate
pip install Flask-Cors==4.0.0

:: 验证
python -c "from flask_cors import CORS; print('OK')"
```

---

### ❌ 错误 15：视频上传失败

**症状**：
- 上传后提示 "Upload failed"
- 服务器报 `413 Request Entity Too Large`

**解决方案**：

```python
# 在 app.py 中增加上传限制
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB

# 确认 uploads 目录存在且有写权限
import os
os.makedirs("./uploads", exist_ok=True)
```

```cmd
:: 手动创建目录
mkdir uploads
mkdir exports
mkdir logs
```

---

### 🛠️ 通用调试命令

```cmd
:: 检查完整环境
check_env.bat

:: 单独测试 YOLO 模型加载
.venv\Scripts\python.exe -c "
from ultralytics import YOLO
model = YOLO('./best.pt')
print('Model classes:', model.names)
print('Model device:', next(model.model.parameters()).device)
"

:: 单独测试视频读取
.venv\Scripts\python.exe -c "
import cv2
cap = cv2.VideoCapture('./ball.mp4')
ret, frame = cap.read()
print('Read OK:', ret)
print('Frame shape:', frame.shape if ret else 'N/A')
"

:: 单独测试 Flask 启动（最小化）
.venv\Scripts\python.exe -c "
from flask import Flask
app = Flask(__name__)
print('Flask OK')
"

:: 查看详细 pip 错误
.venv\Scripts\pip.exe install Flask -v 2>&1 | more

:: 强制重装所有依赖
.venv\Scripts\pip.exe install -r requirements.txt --force-reinstall
```

---

### 📋 错误速查表

| 错误关键词                      | 快速解决                               |
| ------------------------------- | -------------------------------------- |
| `'python' 不是内部命令`         | 重装 Python，勾选 Add to PATH          |
| `No module named 'cv2'`         | `pip install opencv-contrib-python`    |
| `No module named 'torch'`       | 重新安装 PyTorch                       |
| `cuda.is_available() = False`   | 更新 NVIDIA 驱动 / 重装 GPU 版 PyTorch |
| `DLL load failed`               | 安装 VC++ 运行时                       |
| `Address already in use`        | `taskkill /F /IM python.exe`           |
| `Out of memory`                 | 降低 `imgsz` 到 320                    |
| `numpy has no attribute 'bool'` | `pip install numpy==1.26.4`            |
| `Model file not found`          | 将 best.pt 放到项目根目录              |
| `Video width < 1280`            | 确认视频为双目拼接格式（1280×480）     |
| `413 Request Entity Too Large`  | 增大 `MAX_CONTENT_LENGTH`              |
| `ReadTimeoutError`              | 使用清华镜像安装                       |
| `Microsoft Visual C++ required` | 安装 VS Build Tools                    |
| `No module named 'flask_cors'`  | `pip install Flask-Cors`               |
| `Permission denied`             | 以管理员身份运行                       |

---

## 📊 性能参考

| 硬件配置      | FPS（imgsz=640） | FPS（imgsz=320） | 延迟   |
| ------------- | ---------------- | ---------------- | ------ |
| RTX 4090 + i9 | ~65 FPS          | ~120 FPS         | <16ms  |
| RTX 3080 + i7 | ~42 FPS          | ~85 FPS          | ~24ms  |
| RTX 2070 + i7 | ~28 FPS          | ~55 FPS          | ~36ms  |
| GTX 1660 + i5 | ~18 FPS          | ~35 FPS          | ~55ms  |
| CPU only (i7) | ~3 FPS           | ~6 FPS           | >300ms |

> **推荐**：至少使用 GTX 1660 级别 GPU，实时检测需要 ≥ 20 FPS

---

## 📝 更新日志

```
v1.0.0 (2025-01)
  + 双目立体视觉深度感知
  + YOLOv11 篮球/篮筐检测
  + 鲁棒轨迹分析（IRLS + Huber + SG平滑）
  + Flask 实时 Web 界面
  + 三路 MJPEG 视频流
  + 进球/速度/角度实时分析
  + Windows 一键安装脚本
```

---

## 📄 License

MIT License - 详见 [LICENSE](LICENSE) 文件

---

<div align="center">

**遇到问题？** 查看 [故障排除](#-故障排除) | 运行 `check_env.bat`

Made with ❤️ for Basketball Analytics

</div>