<div align="center">

# 🏀 AI 篮球投篮分析系统
### Basketball Shot Detection System

**双目立体视觉 × YOLOv11 × 实时 3D 轨迹分析**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.5-ee4c2c?logo=pytorch" />
  <img src="https://img.shields.io/badge/YOLOv11-ultralytics-00FFFF?logo=yolo" />
  <img src="https://img.shields.io/badge/Flask-3.1-black?logo=flask" />
  <img src="https://img.shields.io/badge/OpenCV-4.10-5C3EE8?logo=opencv" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>

<p align="center">
  <b>实时双目立体视觉 · 深度学习目标检测 · 投篮轨迹跟踪与命中判定</b>
</p>

</div>

---

## 📖 目录

- [🎯 项目介绍](#-项目介绍)
- [✨ 核心特性](#-核心特性)
- [🏗️ 系统架构](#️-系统架构)
- [📁 目录结构](#-目录结构)
- [⚙️ 环境要求](#️-环境要求)
- [🚀 快速开始](#-快速开始)
- [🖥️ Web 界面使用](#️-web-界面使用)
- [🔌 API 接口文档](#-api-接口文档)
- [🧩 模块详解](#-模块详解)
- [🔧 配置说明](#-配置说明)
- [🐛 故障排除](#-故障排除)
- [📊 性能参考](#-性能参考)
- [📝 更新日志](#-更新日志)

---

## 🎯 项目介绍

本系统是一个基于 **双目立体视觉** 和 **YOLO 深度学习目标检测** 的 AI 篮球投篮分析平台。通过离线视频预处理与实时回放分析相结合的方式，实现对投篮动作的自动识别、轨迹追踪、弧度计算和命中判定。

系统提供一个美观的 Web 可视化界面，支持实时查看左/右相机画面、深度图、投篮轨迹、命中统计等信息。

通过双目摄像头（或双目合拍视频）获取三维空间信息，利用 YOLOv11 检测篮球和篮筐，结合自研的鲁棒轨迹分析算法，实时计算：

| 指标       | 说明                                         |
| ---------- | -------------------------------------------- |
| 🎯 进球判定 | 基于 3D 空间穿越算法，检测球是否通过篮圈平面 |
| 💨 出手速度 | 通过 3D 轨迹时序回归精确估算（m/s）          |
| 📐 出手角度 | X-Z 平面内的初始飞行角度（°）                |
| 📈 飞行轨迹 | 鲁棒加权二次曲线拟合 + 预测延伸              |
| 📍 3D 坐标  | 球和篮筐的实时三维位置（m）                  |

---

## ✨ 核心特性

| 特性                 | 说明                                                         |
| -------------------- | ------------------------------------------------------------ |
| 🎯 **YOLO 目标检测**  | 基于 YOLOv11 自定义模型，精准检测篮球和篮筐                  |
| 📷 **双目立体视觉**   | 利用双目相机标定参数，进行立体校正与深度估计                 |
| 🔁 **离线预处理管线** | 四阶段流水线：逐帧检测 → 跨帧跟踪 → 轨迹平滑 → 插值补帧      |
| 🏀 **智能投篮检测**   | 五阶段状态机（IDLE → RISING → TRACKING → RESULT → COOLDOWN） |
| 📊 **命中判定**       | 基于轨迹穿越篮筐平面检测，支持空心球/打板球/弹框球判定       |
| 🌐 **Web 可视化**     | Flask + MJPEG 实时推流，响应式前端界面                       |
| 📈 **实时统计**       | FPS、延迟、GPU 显存、命中率等数据实时显示                    |

### 设计亮点

#### 1. 🔭 双目立体视觉深度感知（VisionEngine）

```
左摄像头帧 ─┐
            ├─► 立体校正(remap) ─► SGBM 视差计算 ─► reprojectImageTo3D ─► 3D 点云
右摄像头帧 ─┘
```

- 使用 **StereoSGBM** 三通道模式（`STEREO_SGBM_MODE_SGBM_3WAY`）计算稠密视差
- 通过预标定的内参/外参矩阵做双目校正，确保极线对齐
- 在每个检测框中心取 **5×5 中位数深度窗口**（`sample_window=5`），有效抑制视差噪声
- 视差图乘以 16，然后 `reprojectImageTo3D` 获得真实米制坐标

#### 2. 🎯 YOLOv11 目标检测（自训练模型）

- 使用 `best.pt` / `best_yolo11.pt` 自训练 YOLO 模型
- 双目视频输入为 **左右拼接（side-by-side）** 格式，宽度 = 2× 单眼宽度
- 仅对**左摄像头帧**做目标检测，右摄像头用于深度计算，节省推理时间
- 类别 ID 映射：`0 → hoop`（篮筐），`1 → basketball`（篮球）

#### 3. 🏀 智能投篮状态机（ShotEngine）

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

#### 4. 📐 鲁棒轨迹分析算法（TrajectoryAnalyzer）

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

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                     Web 前端 (index.html)                │
│         MJPEG视频流 · REST API · 实时状态轮询             │
└───────────────────────┬─────────────────────────────────┘
                        │ HTTP
┌───────────────────────▼─────────────────────────────────┐
│                   Flask 后端 (app.py)                     │
│              路由控制 · 视频上传 · 流推送                   │
└──┬──────────┬──────────┬──────────┬──────────────────────┘
   │          │          │          │
   ▼          ▼          ▼          ▼
┌──────┐ ┌────────┐ ┌────────┐ ┌──────────┐
│预处理 │ │视觉引擎│ │投篮引擎│ │ 配置管理  │
│器    │ │        │ │        │ │          │
└──────┘ └────────┘ └────────┘ └──────────┘
```

### 处理流程

```
视频输入 ──► 离线预处理(4阶段) ──► 回放+立体视觉 ──► 投篮检测(5阶段) ──► Web展示
               │                      │                  │
               ├─ Phase1: YOLO检测     ├─ 立体校正         ├─ IDLE: 等待
               ├─ Phase2: IoU跟踪      ├─ 视差计算         ├─ RISING: 上升
               ├─ Phase3: 轨迹平滑     ├─ 深度估计         ├─ TRACKING: 跟踪
               └─ Phase4: 导出         └─ 3D坐标           ├─ RESULT: 结果
                                                           └─ COOLDOWN: 冷却
```

---

## 📁 目录结构

```
basketball-shot-analysis/
│
├── app.py                   # Flask 主应用 — 路由、视频流、检测线程
├── best.pt                  # YOLO 模型权重文件
├── trajectory_analyzer.py   # 鲁棒轨迹分析核心算法
├── convert_videos.py        # 视频格式转换工具
├── requirements.txt         # Python 依赖列表
├── README.md                # 项目说明文档
│
├── engine/                  # 核心引擎模块
│   ├── config.py            # 全局配置参数（阈值、窗口大小等）
│   ├── models.py            # 数据模型定义（检测、轨迹、帧数据）
│   ├── preprocessor.py      # 离线视频预处理器（检测+跟踪+平滑+插值）
│   ├── shot_engine.py       # 投篮检测状态机（出手识别+命中判定）
│   └── vision_engine.py     # 双目视觉引擎（立体匹配+深度估计+绘制）
│
├── static/
│   └── index.html           # Web 前端界面
│
├── uploads/                 # 用户上传的视频文件
├── exports/                 # 导出数据目录
├── logs/                    # 日志文件目录
│
├── install.bat              # Windows 一键安装脚本（GPU版）
├── install_cpu.bat          # Windows CPU版安装脚本
├── run.bat                  # Windows 一键启动脚本
├── check_env.bat            # 环境检测工具
│
├── ball.avi                 # 示例视频（双目拼接格式）
├── basketball1.mp4          # 测试视频 1
└── basketball2.mp4          # 测试视频 2
```

---

## ⚙️ 环境要求

| 项目         | 最低要求                  | 推荐配置                    |
| ------------ | ------------------------- | --------------------------- |
| **操作系统** | Windows 10 / Ubuntu 20.04 | Windows 11 / Ubuntu 22.04   |
| **Python**   | 3.9                       | 3.10 – 3.12                 |
| **GPU**      | — (可用CPU)               | NVIDIA GTX 1060+ (6GB+显存) |
| **CUDA**     | —                         | CUDA 12.1 + cuDNN 8.x       |
| **内存**     | 8 GB                      | 16 GB+                      |
| **硬盘**     | 10 GB 可用空间            | 20 GB+                      |
| **CPU**      | 4 核                      | 8 核+                       |

### GPU 加速（强烈推荐）

| 组件     | 要求                             |
| -------- | -------------------------------- |
| **GPU**  | NVIDIA GPU（GTX 1060 及以上）    |
| **VRAM** | 4 GB+（推荐 8 GB+）              |
| **CUDA** | 12.1+（由 install.bat 自动匹配） |
| **驱动** | NVIDIA Driver 525.0+             |

> ⚠️ **CPU 模式警告**：无 GPU 时系统可运行但 FPS 极低（<5 FPS），不适合实时检测

---

## 🚀 快速开始

### 方式一：Windows 一键安装（推荐）

1. **下载项目**
   ```bash
   git clone https://github.com/your-username/basketball-shot-analysis.git
   cd basketball-shot-analysis
   ```

2. **运行安装脚本**
   - **有 NVIDIA GPU**：双击 `install.bat`
   - **无 GPU / 仅 CPU**：双击 `install_cpu.bat`

3. **启动系统**
   ```
   双击 run.bat
   ```

4. **打开浏览器访问**
   ```
   http://127.0.0.1:5002
   ```

### 方式二：手动安装

#### 1. 创建虚拟环境

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

#### 2. 安装 PyTorch

**GPU 版本（CUDA 12.1）：**

```bash
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

**CPU 版本：**

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
```

#### 3. 安装其他依赖

```bash
pip install -r requirements.txt
```

> ⚠️ 如果使用 CPU 版本，请先编辑 `requirements.txt`，注释掉带 `+cu121` 的 torch 相关行。使用国内镜像可加速安装：
> ```bash
> pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
> ```

#### 4. 放置模型文件

将训练好的 YOLO 模型权重文件 `best.pt`（或 `best_yolo11.pt`）放在项目根目录下。

#### 5. 启动

```bash
python app.py
```

浏览器访问 `http://127.0.0.1:5002`

---

## 📹 视频要求

本系统使用 **双目立体相机** 拍摄的视频，视频帧为左右相机画面 **水平拼接** 格式：

```
┌──────────────┬──────────────┐
│   左相机      │   右相机      │
│  (640×480)   │  (640×480)   │
└──────────────┴──────────────┘
       总分辨率: 1280×480
```

支持的视频格式：`avi`、`mp4`、`mov`、`mkv`、`flv`、`wmv`

可通过 Web 界面上传视频，或将视频文件直接放在项目根目录下。

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

1. **选择视频**：点击「📁 选择视频」上传本地视频文件
2. **开始分析**：点击「▶ 开始分析」启动检测流程
3. **预处理阶段**：系统自动进行离线逐帧检测（可在遮罩层查看进度）
4. **回放阶段**：预处理完成后自动进入回放，实时显示：
   - 📹 左相机画面（含检测框 + 轨迹叠加）
   - 📷 右相机画面
   - 🌈 深度图（JET 伪彩色）
   - 🎯 投篮检测统计（命中/出手/命中率）
5. **暂停/停止**：可随时暂停或停止分析

---

## 🔌 API 接口文档

| 端点                    | 方法 | 说明             |
| ----------------------- | ---- | ---------------- |
| `/api/start`            | POST | 启动检测         |
| `/api/pause`            | POST | 暂停/恢复        |
| `/api/stop`             | POST | 停止检测         |
| `/api/status`           | GET  | 获取系统状态     |
| `/api/shot_analysis`    | GET  | 获取投篮分析数据 |
| `/api/upload`           | POST | 上传视频文件     |
| `/api/video_feed/left`  | GET  | 左相机 MJPEG 流  |
| `/api/video_feed/right` | GET  | 右相机 MJPEG 流  |
| `/api/video_feed/depth` | GET  | 深度图 MJPEG 流  |

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
  "predicted_trajectory": [[0.2, 3.1], [0.5, 3.8]],
  "actual_trajectory": [[1.2, 3.5], [0.9, 4.1]],
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
    # 检测参数
    conf_threshold: float = 0.35    # 置信度阈值
    iou_threshold: float = 0.45     # NMS IoU 阈值
    max_det: int = 300              # 最大检测数

    # 图像尺寸（单目）
    width: int = 640
    height: int = 480

    # 跟踪参数
    track_iou_thresh: float = 0.20  # 跟踪IoU阈值
    track_max_gap: int = 15         # 最大丢失帧数
    track_min_len: int = 3          # 最短轨迹长度

    # 平滑参数
    smooth_window: int = 5          # 滑动平均窗口
    median_filter_size: int = 3     # 中位数滤波窗口
    interpolate_max_gap: int = 10   # 最大插值间隔

    # 深度采样
    sample_window: int = 5          # 深度采样窗口（奇数）
    hoop_radius_m: float = 0.15     # 篮圈半径（米）
    min_points: int = 8             # 轨迹最少采样点
    max_points: int = 80            # 轨迹最多保留点
    lost_timeout_s: float = 0.35    # 球消失超时（秒）
```

### `engine/models.py` — 数据结构

| 类                | 字段                                | 说明                  |
| ----------------- | ----------------------------------- | --------------------- |
| `Detection3D`     | cls, conf, bbox, xyz, distance      | 单目标检测结果+3D坐标 |
| `FrameDetections` | frame_id, timestamp, detections     | 单帧所有检测结果      |
| `ShotResult`      | is_scored, speed, angle, trajectory | 完整投篮分析结果      |
| `ShotState`       | is_active, points, hoop_history     | 投篮状态机内部状态    |

### 离线预处理管线 (`preprocessor.py`)

| 阶段    | 功能          | 说明                                                 |
| ------- | ------------- | ---------------------------------------------------- |
| Phase 1 | YOLO 逐帧检测 | 每帧编码为 JPG 再解码，模拟图片输入减少噪声          |
| Phase 2 | 跨帧 IoU 关联 | 贪心匹配算法，基于 IoU + 中心距离构建目标轨迹        |
| Phase 3 | 轨迹平滑      | 中位数滤波去噪 → 滑动平均平滑 → 线性插值补缺         |
| Phase 4 | 格式导出      | 同帧同类去重，输出 `{frame_id: [StableDetection2D]}` |

### 投篮检测状态机 (`shot_engine.py`)

```
IDLE ──(检测到连续上升)──► RISING ──(越过顶点/到达篮筐附近)──► TRACKING
  ▲                                                              │
  │                                                              ▼
  └──────── COOLDOWN ◄──── RESULT ◄────(轨迹结束/穿越篮筐)─────┘
```

**关键改进 (v5)：**
- 使用连续下降帧计数器，避免单帧速度波动导致的误判
- 篮筐附近区域保护机制，防止 RISING 阶段被错误重置
- 弹地过滤：检测上升前是否有快速下落特征
- 多穿越扫描进球判定，支持打板球/弹框球

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

## 🧠 模型训练

本系统使用 [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) 框架，检测类别为：

| 类别 ID | 类别名     | 说明 |
| ------- | ---------- | ---- |
| 0       | hoop       | 篮筐 |
| 1       | basketball | 篮球 |

如需自行训练模型，请参考 Ultralytics 官方文档准备数据集并训练：

```bash
yolo train model=yolo11n.pt data=basketball.yaml epochs=100 imgsz=640
```

---

## 🐛 故障排除

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

<details>
<summary><b>Q: 提示 CUDA 不可用？</b></summary>

1. 确认已安装 NVIDIA 驱动和 CUDA Toolkit 12.1
2. 运行 `nvidia-smi` 检查 GPU 状态
3. 确认安装的是 CUDA 版本的 PyTorch：

   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```
4. 如果仍不可用，系统会自动回退到 CPU 模式运行

</details>

<details>
<summary><b>Q: 安装依赖时报错？</b></summary>

- 确保 Python 版本为 3.9 - 3.12
- 使用虚拟环境避免包冲突
- 如果 `opencv-contrib-python` 和 `opencv-python` 冲突，只保留一个：

  ```bash
  pip uninstall opencv-python opencv-contrib-python -y
  pip install opencv-contrib-python==4.10.0.84
  ```

</details>

<details>
<summary><b>Q: 视频分析没有检测结果？</b></summary>

- 确认视频为双目拼接格式（1280×480）
- 确认 `best.pt` 模型文件存在于项目根目录
- 尝试降低置信度阈值 `conf_threshold`

</details>

<details>
<summary><b>Q: 画面卡顿 / FPS 很低？</b></summary>

- 推荐使用 NVIDIA GPU 加速
- 缩短视频长度或降低分辨率
- 在 `engine/config.py` 中降低推理分辨率：`imgsz: int = 320`

</details>

<details>
<summary><b>Q: PyTorch 安装失败 / 网络超时？</b></summary>

使用国内镜像（推荐国内用户）：

```cmd
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 ^
    -i https://pypi.tuna.tsinghua.edu.cn/simple ^
    --index-url https://download.pytorch.org/whl/cu121
```

或增大 pip 超时时间：`pip install torch --timeout=600 --retries=5`

</details>

<details>
<summary><b>Q: Web 界面打开但视频流黑屏？</b></summary>

1. 检查视频文件分辨率是否为 1280×480（双目拼接）
2. 直接访问 `http://127.0.0.1:5002/api/video_feed/left` 验证流是否正常
3. 清除浏览器缓存或换用 Chrome/Edge 浏览器
4. 查看 `logs/` 目录下的日志文件

</details>

### 🛠️ 通用调试命令

```cmd
:: 检查完整环境
check_env.bat

:: 单独测试 YOLO 模型加载
.venv\Scripts\python.exe -c "
from ultralytics import YOLO
model = YOLO('./best.pt')
print('Model classes:', model.names)
"

:: 单独测试视频读取
.venv\Scripts\python.exe -c "
import cv2
cap = cv2.VideoCapture('./ball.avi')
ret, frame = cap.read()
print('Read OK:', ret, '| Shape:', frame.shape if ret else 'N/A')
"

:: 强制重装所有依赖
.venv\Scripts\pip.exe install -r requirements.txt --force-reinstall
```

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
v2.0.0
  + 新增离线四阶段预处理管线（逐帧检测 → 跨帧跟踪 → 轨迹平滑 → 插值补帧）
  + 五阶段投篮状态机（IDLE/RISING/TRACKING/RESULT/COOLDOWN）
  + 空心球/打板球/弹框球智能判定
  + install_cpu.bat CPU版一键安装脚本
  + check_env.bat 环境检测工具

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

## 🙏 致谢

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) — 目标检测框架
- [OpenCV](https://opencv.org/) — 计算机视觉库
- [PyTorch](https://pytorch.org/) — 深度学习框架
- [Flask](https://flask.palletsprojects.com/) — Web 框架

---

## 📄 License

本项目采用 [MIT License](LICENSE) 开源协议。

---

<div align="center">

**遇到问题？** 查看 [故障排除](#-故障排除) | 运行 `check_env.bat`

Made with ❤️ for Basketball Analytics

</div>