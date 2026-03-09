# 🏀 AI 篮球投篮分析系统

### Basketball Shot Detection System

> **双目立体视觉 × YOLOv11 × 抛物线物理分析 × 实时 3D 轨迹分析**
>
> 实时双目立体视觉 · 深度学习目标检测 · 抛物线拟合出手识别 · 投篮轨迹跟踪与命中判定

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

本系统是一个基于 **双目立体视觉** 和 **YOLO 深度学习目标检测** 的 AI 篮球投篮分析平台。
通过 **五阶段离线预处理**（含抛物线物理分析）与实时回放分析相结合的方式，
实现对投篮动作的自动识别、轨迹追踪、弧度计算和命中判定。

系统提供一个美观的 Web 可视化界面，支持实时查看左/右相机画面、深度图、投篮轨迹、命中统计等信息。

通过双目摄像头（或双目合拍视频）获取三维空间信息，利用 YOLOv11 检测篮球和篮筐，
结合自研的 **抛物线拟合出手检测** 和 **鲁棒进球判定算法**，实时计算：

| 指标       | 说明                                                       |
| ---------- | ---------------------------------------------------------- |
| 🎯 进球判定 | 基于多穿越感知 + 网弹/框弹容忍算法，精确判定进球           |
| 🏀 出手检测 | Phase 5 抛物线拟合自由飞行段检测，物理驱动出手帧识别       |
| 📈 飞行轨迹 | 五阶段预处理：检测 → 跟踪 → 平滑 → 导出 → 物理分析         |
| 📐 弧线高度 | 出手点到弧顶的像素高度差                                   |
| 📍 3D 坐标  | 球和篮筐的实时三维位置（m），双目立体视觉 + 流水线并行加速 |

---

## ✨ 核心特性

| 特性                | 说明                                                         |
| ------------------- | ------------------------------------------------------------ |
| 🎯 YOLO 目标检测     | 基于 YOLOv11 自定义模型，精准检测篮球和篮筐                  |
| 📷 双目立体视觉      | 利用双目相机标定参数，进行立体校正与深度估计                 |
| 🔁 五阶段离线预处理  | 检测 → 跟踪 → 平滑 → 导出 → **抛物线物理分析**               |
| 🧮 抛物线出手检测    | Phase 5 像素重力估算 + 自由飞行段拟合 + 出手帧精确定位       |
| 🏀 智能投篮状态机    | 五阶段状态机（IDLE → RISING → TRACKING → RESULT → COOLDOWN） |
| 📊 v7 进球判定       | 网弹容忍 + 框弹多穿越感知 + 弹跳延长跟踪 + 实时穿越快速确认  |
| ⚡ stereo 流水线并行 | SGBM 立体匹配移至后台线程，OpenCV C++ 释放 GIL，主线程用前一帧结果 |
| 🌐 Web 可视化        | Flask + MJPEG 实时推流，响应式前端界面                       |
| 📈 实时统计          | FPS、延迟、GPU 显存、命中率等数据实时显示                    |

---

### 设计亮点

#### 1. 🔭 双目立体视觉深度感知 + 流水线并行（VisionEngine v2）

```text
左摄像头帧 ─┐                    ┌─ 当前帧: 提交 stereo 到后台线程
            ├─► 立体校正 ─► SGBM ─► 3D
右摄像头帧 ─┘                    └─ 主线程使用前一帧缓存结果（流水线并行, 零等待）
```

- 使用 `ThreadPoolExecutor` 将 SGBM 立体匹配移至后台线程
- OpenCV C++ 操作释放 GIL，不受 Python GIL 限制
- 主线程使用前一帧的深度缓存结果，实现流水线并行
- 无检测帧跳过 `reprojectImageTo3D`，减少不必要计算
- 在每个检测框中心取 7×7 中位数深度窗口，有效抑制视差噪声

#### 2. 🎯 YOLOv11 目标检测 + 批量推理

- 使用 `best.pt` / `best_yolo11.pt` 自训练 YOLO 模型
- **Phase 1 批量推理**：多帧打包一次 `model.predict()`，GPU 利用率更高
- 仅对左摄像头帧做目标检测，右摄像头用于深度计算
- 类别 ID 映射：`0 → hoop`（篮筐），`1 → basketball`（篮球）

#### 3. 🧮 Phase 5 抛物线物理分析（全新）

```text
篮球轨迹 ─► 像素重力估算   ─► 自由飞行段检测  ─► 出手帧定位
              │                   │                  │
              滑窗二次拟合        水平加速度约束      篮筐区域排除
              R² ≥ 0.95          垂直 R² ≥ 0.90     弹地排除
              g ∈ [0.05,2.0]     步进采样加速        40% 画面约束
              Vandermonde         矩阵乘法批量        上升动作验证
              伪逆预计算          拟合 + R² 计算
```

- **像素重力估算**：滑动窗口二次多项式拟合，提取抛物线的二次项系数作为像素重力
- **自由飞行段检测**：水平加速度 ≤ 重力 × 20%，垂直 R² ≥ 0.90
- **出手帧定位**：自由飞行段起始、篮筐区域排除、弹地排除、上升动作验证
- **性能优化**：预计算 Vandermonde 伪逆矩阵，矩阵乘法批量拟合 + R² 计算

#### 4. 🏀 v7 投篮状态机（ShotEngine）

```text
Phase 5 抛物线信号 → [RISING] → 越过弧顶 → [TRACKING] → 进球/未中判定

v7 增强：
  · 网弹容忍
  · 框弹多穿越
  · 弹跳延长跟踪
  · 实时穿越快速确认
```

- **出手检测**：由 Phase 5 抛物线拟合信号驱动（`is_physics_release`）
- **进球判定 v7**：网弹容忍（球在篮筐 X 范围内弹网回弹仍判进球）
- **多穿越感知**：框弹 ≥2 次穿越自动放宽判定标准
- **弹跳延长跟踪**：篮筐区域内检测方向交替，自动延长跟踪超时

---

## 🏗️ 系统架构

```text
┌─────────────────────────────────────────────────────────┐
│                   Web 前端 (index.html)                  │
│           MJPEG视频流 · REST API · 实时状态轮询           │
└───────────────────────┬─────────────────────────────────┘
                        │ HTTP
┌───────────────────────▼─────────────────────────────────┐
│                  Flask 后端 (app.py)                     │
│               路由控制 · 视频上传 · 流推送                │
└──┬──────────┬──────────┬──────────┬──────────────────────┘
   │          │          │          │
   ▼          ▼          ▼          ▼
┌──────┐ ┌────────┐ ┌────────┐ ┌──────────┐
│预处理│ │视觉引擎│ │投篮引擎│ │ 配置管理 │
│  器  │ │(并行)  │ │ (v7)  │ │          │
└──────┘ └────────┘ └────────┘ └──────────┘
```

### 处理流程

```text
视频输入 ──► 离线预处理(5阶段) ──► 回放+立体视觉(并行) ──► 投篮检测(v7) ──► Web展示
                  │                        │                      │
                  ├─ Phase1: YOLO批量检测   ├─ 后台线程立体校正     ├─ IDLE:     等待物理出手信号
                  ├─ Phase2: IoU跟踪        ├─ 后台线程SGBM匹配     ├─ RISING:   弧线上升
                  ├─ Phase3: 轨迹平滑       ├─ 主线程用缓存深度     ├─ TRACKING: 弧线跟踪
                  ├─ Phase4: 导出           └─ 按需3D重投影         │            (网弹/框弹/弹跳容忍)
                  └─ Phase5: 抛物线出手                            ├─ RESULT:   结果
                       · 像素重力估算                              └─ COOLDOWN: 冷却
                       · 自由飞行段检测
                       · 出手帧定位
```

---

## 📁 目录结构

```text
basketball-shot-analysis/
│
├── app.py                  # Flask 主应用 — 路由、视频流、检测线程
├── best.pt                 # YOLO 模型权重文件
├── requirements.txt        # Python 依赖列表
├── README.md               # 项目说明文档（本文件）
│
├── engine/                 # 核心引擎模块
│   ├── config.py           # 全局配置参数（Phase 1-5 全部阈值）
│   ├── models.py           # 数据模型定义（含 PhysicsMetadata）
│   ├── preprocessor.py     # 离线视频预处理器（5阶段：检测+跟踪+平滑+导出+物理分析）
│   ├── shot_engine.py      # 投篮检测状态机 v7（物理出手+网弹/框弹容忍+弹跳延长）
│   └── vision_engine.py    # 双目视觉引擎 v2（流水线并行 stereo + 深度估计）
│
├── static/
│   └── index.html          # Web 前端界面
│
├── uploads/                # 用户上传的视频文件
├── exports/                # 导出数据目录
├── logs/                   # 日志文件目录
│
├── install.bat             # Windows 一键安装脚本（GPU 版）
├── install_cpu.bat         # Windows CPU 版安装脚本
├── run.bat                 # Windows 一键启动脚本
├── check_env.bat           # 环境检测工具
│
├── ball.avi                # 示例视频（双目拼接格式）
├── basketball1.mp4         # 测试视频 1
└── basketball2.mp4         # 测试视频 2
```

---

## ⚙️ 环境要求

| 项目     | 最低要求                  | 推荐配置                      |
| -------- | ------------------------- | ----------------------------- |
| 操作系统 | Windows 10 / Ubuntu 20.04 | Windows 11 / Ubuntu 22.04     |
| Python   | 3.9                       | 3.10 – 3.12                   |
| GPU      | —（可用 CPU）             | NVIDIA GTX 1060+（6GB+ 显存） |
| CUDA     | —                         | CUDA 12.1 + cuDNN 8.x         |
| 内存     | 8 GB                      | 16 GB+                        |
| 硬盘     | 10 GB 可用空间            | 20 GB+                        |
| CPU      | 4 核                      | 8 核+                         |

### GPU 加速（强烈推荐）

| 组件 | 要求                             |
| ---- | -------------------------------- |
| GPU  | NVIDIA GPU（GTX 1060 及以上）    |
| VRAM | 4 GB+（推荐 8 GB+）              |
| CUDA | 12.1+（由 install.bat 自动匹配） |
| 驱动 | NVIDIA Driver 525.0+             |

> ⚠️ **CPU 模式警告**：无 GPU 时系统可运行但 FPS 极低（< 5 FPS），不适合实时检测。

---

## 🚀 快速开始

### 方式一：Windows 一键安装（推荐）

**Step 1. 下载项目**

```bash
git clone https://github.com/your-username/basketball-shot-analysis.git
cd basketball-shot-analysis
```

**Step 2. 运行安装脚本**

- 有 NVIDIA GPU → 双击 `install.bat`
- 无 GPU / 仅 CPU → 双击 `install_cpu.bat`

**Step 3. 启动系统**

双击 `run.bat`，然后浏览器访问 `http://127.0.0.1:5002`

---

### 方式二：手动安装

**Step 1. 创建虚拟环境**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

**Step 2. 安装 PyTorch**

```bash
# GPU 版本（CUDA 12.1）
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# CPU 版本
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
```

**Step 3. 安装其他依赖**

```bash
pip install -r requirements.txt
```

**Step 4. 放置模型文件**

将 `best.pt`（或 `best_yolo11.pt`）放在项目根目录。

**Step 5. 启动**

```bash
python app.py
```

浏览器访问 `http://127.0.0.1:5002`

---

### 📹 视频格式要求

本系统使用双目立体相机拍摄的视频，视频帧为**左右相机画面水平拼接**格式：

```text
┌──────────────┬──────────────┐
│    左相机    │    右相机    │
│  (640×480)   │  (640×480)   │
└──────────────┴──────────────┘
        总分辨率: 1280×480
```

支持格式：`avi` · `mp4` · `mov` · `mkv` · `flv` · `wmv`

---

## 🖥️ Web 界面使用

### 主界面布局

```text
┌─────────────────────────────────────────────────────┐
│   🏀 Basketball Shot Detection      [Start] [Stop]   │
├────────────┬────────────┬────────────────────────────┤
│  左摄像头  │  右摄像头  │      📊 系统状态            │
│  LIVE ●   │  LIVE ●   │   FPS / 帧数 / GPU          │
├────────────┴────────────┤      🏀 检测计数            │
│      深度伪彩视图        ├────────────────────────────┤
│      LIVE ●            │      🎯 投篮分析            │
│                        │   RELEASE标记 / 弧高         │
├─────────────────────────┤   命中率 / 投篮历史          │
│    📋 投篮历史记录       └────────────────────────────┘
└─────────────────────────────────────────────────────┘
```

### 操作步骤

1. **选择视频**：点击「📁 选择视频」上传本地视频文件
2. **开始分析**：点击「▶ 开始分析」启动检测流程
3. **预处理阶段**：系统自动进行五阶段离线预处理（可在遮罩层查看进度）
   - Phase 1–4：检测、跟踪、平滑、导出
   - Phase 5：抛物线物理分析（像素重力估算 + 出手帧检测）
4. **回放阶段**：预处理完成后自动进入流水线并行回放，实时显示：
   - 📹 左相机画面（含检测框 + 轨迹 + RELEASE 出手标记）
   - 📷 右相机画面
   - 🌈 深度图（JET 伪彩色，后台线程计算）
   - 🎯 投篮检测统计（命中 / 出手 / 命中率）
5. **暂停 / 停止**：可随时暂停或停止分析

---

## 🔌 API 接口文档

| 端点                    | 方法 | 说明             |
| ----------------------- | ---- | ---------------- |
| `/api/start`            | POST | 启动检测         |
| `/api/pause`            | POST | 暂停 / 恢复      |
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
  "is_preprocessing": false,
  "preprocess_progress": "",
  "device": "NVIDIA GeForce RTX 3080",
  "stats": {
    "fps": 28.5,
    "current_frame": 1520,
    "total_frames": 3600,
    "process_time": 35.2,
    "gpu_memory": 1.82,
    "basketball_count": 1,
    "hoops_count": 1,
    "current_video": "basketball1.mp4",
    "phase": "playback"
  },
  "shot_info": {
    "phase": "tracking",
    "total_shots": 5,
    "total_scored": 3,
    "accuracy": "60%",
    "last_shot": {
      "is_scored": true,
      "arc_height": 156,
      "duration": 1.23,
      "release": [320, 350],
      "apex": [400, 120],
      "frame_start": 1200,
      "frame_end": 1280,
      "time": "14:32:15"
    },
    "shot_history": []
  }
}
```

---

## 🧩 模块详解

### `engine/config.py` — 统一配置（含 Phase 5 参数）

```python
@dataclass
class EngineConfig:
    # ═══ 检测参数 ═══
    conf_threshold: float = 0.35        # 置信度阈值
    iou_threshold: float  = 0.45        # NMS IoU 阈值
    max_det: int          = 300         # 最大检测数

    # ═══ 图像尺寸（单目）═══
    width:  int = 640
    height: int = 480

    # ═══ Phase 1: 批量检测 ═══
    detect_batch_size: int = 8          # YOLO 批量推理大小

    # ═══ Phase 2: 跟踪 ═══
    track_iou_thresh:     float = 0.20  # 跟踪 IoU 阈值
    track_max_gap:        int   = 15    # 最大丢失帧数
    track_min_len:        int   = 3     # 最短轨迹长度
    track_max_center_dist:int   = 120   # 最大中心距离

    # ═══ Phase 3: 平滑 ═══
    smooth_window:        int = 5       # 滑动平均窗口
    median_filter_size:   int = 3       # 中位数滤波窗口
    size_smooth_window:   int = 9       # 尺寸平滑窗口
    interpolate_max_gap:  int = 10      # 最大插值间隔

    # ═══ Phase 5: 物理分析（新增）═══
    # 像素重力估算
    gravity_fit_window_sec:     float = 0.30
    gravity_fit_r2_min:         float = 0.95
    gravity_min_px_per_frame2:  float = 0.05
    gravity_max_px_per_frame2:  float = 2.0
    # 自由飞行检测
    horiz_accel_ratio:          float = 0.20
    free_flight_window_sec:     float = 0.15
    free_flight_r2_min:         float = 0.90
    # 出手帧筛选
    release_hoop_margin_x:      float = 1.5
    release_hoop_margin_y_up:   float = 1.0
    release_hoop_margin_y_dn:   float = 1.5
    release_bounce_consec_frames:int  = 3
    release_bounce_fall_speed:  float = 3.0
    release_min_rise_px:        float = 30
    release_min_cy_ratio:       float = 0.40  # 出手点必须在画面 40% 以下

    # ═══ 深度采样 ═══
    sample_window: int = 7              # 深度采样窗口
```

### `engine/models.py` — 数据结构

| 类                  | 关键字段                                             | 说明                        |
| ------------------- | ---------------------------------------------------- | --------------------------- |
| `RawDetection2D`    | cls, conf, bbox, center                              | Phase 1 原始检测结果        |
| `StableDetection2D` | track_id, cls, conf, bbox, **is_release_point**      | Phase 4 稳定检测 + 出手标记 |
| `Detection3D`       | track_id, cls, conf, bbox, xyz, **is_release_point** | 含 3D 坐标的检测结果        |
| `FrameDetections`   | frame_id, timestamp, detections                      | 单帧所有检测结果            |
| `PhysicsMetadata`   | estimated_gravity_px, **release_frames**             | Phase 5 物理分析结果        |

### 离线预处理管线（`preprocessor.py`）— 五阶段

| 阶段        | 功能               | 说明                                                 | 优化                            |
| ----------- | ------------------ | ---------------------------------------------------- | ------------------------------- |
| **Phase 1** | YOLO 批量检测      | 多帧打包 `model.predict()`，GPU 批量推理             | `detect_batch_size=8`           |
| **Phase 2** | 跨帧 IoU 关联      | 贪心匹配，基于 IoU + 中心距离构建目标轨迹            | —                               |
| **Phase 3** | 轨迹平滑           | numpy 向量化中值滤波 → 滑动平均 → 线性插值补缺       | `sliding_window_view`           |
| **Phase 4** | 格式导出           | 同帧同类去重，输出 `{frame_id: [StableDetection2D]}` | 出手帧标记                      |
| **Phase 5** | **抛物线物理分析** | 像素重力估算 → 自由飞行段检测 → 出手帧定位           | Vandermonde 伪逆 + 矩阵批量拟合 |

#### Phase 5 详细流程

```text
                       ┌──────────────────┐
  所有篮球轨迹 ───────►│   像素重力估算    │
                       │   滑窗二次拟合    │
                       │   R² ≥ 0.95      │
                       │ g ∈ [0.05, 2.0]  │
                       └────────┬─────────┘
                                │ 全局 g_px（中位数）
                                ▼
                       ┌──────────────────┐
  每条篮球轨迹 ───────►│  自由飞行段检测   │
                       │ 水平加速度 ≤ 20%g │
                       │  垂直 R² ≥ 0.90  │
                       │   步进采样加速    │
                       └────────┬─────────┘
                                │ [(start_frame, end_frame), ...]
                                ▼
                       ┌──────────────────┐
                       │    出手帧定位     │
                       │  · 篮筐区域排除   │
                       │  · 弹地历史排除   │
                       │  · 画面 40% 约束  │
                       │  · 上升动作验证   │
                       └────────┬─────────┘
                                │ release_frame_id
                                ▼
                          PhysicsMetadata
```

### 投篮检测状态机 v7（`shot_engine.py`）

```text
Phase 5 抛物线信号
  (is_physics_release)
        │
        ▼
┌──────┐   物理出手触发    ┌──────────┐
│ IDLE │ ───────────────► │  RISING  │
│ 等待  │                  │  上升检测 │
└──┬───┘                  └────┬─────┘
   ▲                           │ 开始下落 + 弧高足够 / 到达篮筐附近
   │ 冷却结束且球远离篮筐        ▼
┌──┴──────┐               ┌──────────────────────┐
│COOLDOWN │◄──── 判定 ────│    TRACKING  (v7)    │
│  冷却    │     完成       │                      │
│ (90帧)  │               │  · 实时穿越检测        │
└─────────┘               │  · 快速进球确认        │
                           │  · 网弹容忍     ★ 新  │
                           │  · 框弹多穿越   ★ 新  │
                           │  · 弹跳延长跟踪 ★ 新  │
                           └──────────────────────┘
```

#### v7 进球判定核心改进

| 场景             | v4 处理                 | v7 改进                                      |
| ---------------- | ----------------------- | -------------------------------------------- |
| 空心入网         | 快速确认（5 帧）        | 保持不变                                     |
| **网弹回弹**     | ❌ 回弹高度超限 → MISSED | ✅ X 仍在篮筐范围 + 回弹高度 < 1.2hh → SCORED |
| **框弹多穿越**   | 每次穿越独立检查        | ✅ 穿越 ≥2 次放宽判定，给更多等待时间         |
| **弹跳期间终止** | 可能被超时终止          | ✅ 活跃弹跳检测，自动延长跟踪超时 25 帧       |
| 弹板进球         | 穿越失效不终止跟踪      | 保持不变                                     |
| 打框弹出         | 实时 + 终态双重验证     | 保持不变                                     |

### 双目视觉引擎 v2（`vision_engine.py`）

```text
主线程:                              后台 stereo 线程:
┌───────────────┐                  ┌──────────────────────┐
│  收取上一帧结果 │                  │     remap 立体校正    │
│  (缓存深度)   │                  │     SGBM 视差计算     │
│               │◄──── 完成 ───── │     深度着色          │
│  提交新帧 ────│───── 提交 ──────►│  reprojectImageTo3D   │
│               │                  │    (仅有检测时)       │
│  使用缓存深度  │                  └──────────────────────┘
│  2D → 3D 映射 │                  OpenCV C++ 释放 GIL
│  绘制检测框   │                  不受 Python GIL 限制
│  绘制出手标记  │
└───────────────┘
```

---

## 🔧 配置说明

### 调整 Phase 5 物理分析

编辑 `engine/config.py`：

```python
# 降低重力估算灵敏度（适合慢速摄像机）
gravity_fit_window_sec: float = 0.50
gravity_fit_r2_min:     float = 0.90

# 放宽自由飞行检测（更多出手被检测到）
free_flight_r2_min:  float = 0.85
horiz_accel_ratio:   float = 0.30

# 调整出手位置约束
release_min_cy_ratio: float = 0.35  # 允许画面 35% 以上出手
release_min_rise_px:  float = 20    # 降低最小上升高度
```

### 调整进球判定灵敏度

编辑 `engine/shot_engine.py` 的 `__init__`：

```python
# 放宽进球 X 对齐容差（适合广角镜头）
self.SCORE_X_TOL = 0.55

# 增大网弹容忍（适合松散球网）
self.NET_BOUNCE_X_TOL  = 0.70
self.NET_BOUNCE_MAX_RISE = 1.5

# 增加弹跳延长帧数
self.BOUNCE_EXTEND_FRAMES = 35
```

### 修改服务端口

编辑 `app.py` 最后一行：

```python
app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)
```

### 自定义双目标定参数

编辑 `app.py` 中 `build_stereo()` 函数，替换自己的标定矩阵：

```python
lm = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])  # 左内参
T  = np.array([-基线距离_mm, ty, tz])                  # 单位：mm
```

---

## 🧠 模型训练

本系统使用 Ultralytics YOLO 框架，检测类别如下：

| 类别 ID | 类别名       | 说明 |
| ------- | ------------ | ---- |
| 0       | `hoop`       | 篮筐 |
| 1       | `basketball` | 篮球 |

如需自行训练模型：

```bash
yolo train model=yolo11n.pt data=basketball.yaml epochs=100 imgsz=640
```

---

## 🐛 故障排除

### 📋 错误速查表

| 错误关键词                      | 快速解决                               |
| ------------------------------- | -------------------------------------- |
| `'python' 不是内部命令`         | 重装 Python，勾选 **Add to PATH**      |
| `No module named 'cv2'`         | `pip install opencv-contrib-python`    |
| `No module named 'torch'`       | 重新安装 PyTorch                       |
| `cuda.is_available() = False`   | 更新 NVIDIA 驱动 / 重装 GPU 版 PyTorch |
| `DLL load failed`               | 安装 VC++ 运行时                       |
| `Address already in use`        | `taskkill /F /IM python.exe`           |
| `Out of memory`                 | 降低 `detect_batch_size` 到 4          |
| `numpy has no attribute 'bool'` | `pip install numpy==1.26.4`            |
| `Model file not found`          | 将 `best.pt` 放到项目根目录            |
| `Video width < 1280`            | 确认视频为双目拼接格式（1280×480）     |

---

### 🔍 GPU 不可用

1. 确认已安装 NVIDIA 驱动和 CUDA Toolkit 12.1
2. 运行 `nvidia-smi` 检查 GPU 状态
3. 验证 PyTorch CUDA 可用性：

   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

4. 若仍不可用，系统会自动回退到 CPU 模式运行

### 🔍 出手检测不触发

1. 确认视频中有完整的投篮动作（出手 → 飞行 → 落地/入筐）
2. 检查控制台输出的重力估算值是否合理（通常 0.1–1.0 px/frame²）
3. 尝试降低自由飞行 R² 阈值：`free_flight_r2_min: float = 0.80`
4. 尝试降低出手位置约束：`release_min_cy_ratio: float = 0.30`
5. 确认篮球轨迹长度足够（≥10 帧），短轨迹无法估算重力

### 🔍 FPS 过低

- 推荐使用 NVIDIA GPU 加速
- v2 视觉引擎已启用流水线并行，确认日志输出 `"流水线 stereo 已启用"`
- 降低 `detect_batch_size` 到 4 以减少 GPU 内存峰值
- 确认没有其他程序占用 GPU

### 🔍 依赖安装失败

- 确保 Python 版本为 3.9–3.12
- 使用虚拟环境避免包冲突
- 若 `opencv-contrib-python` 与 `opencv-python` 冲突：

  ```bash
  pip uninstall opencv-python opencv-contrib-python -y
  pip install opencv-contrib-python==4.10.0.84
  ```

### 🔍 视频流无画面

1. 检查视频文件分辨率是否为 1280×480（双目拼接）
2. 直接访问 `http://127.0.0.1:5002/api/video_feed/left` 验证流是否正常
3. 清除浏览器缓存或换用 Chrome / Edge 浏览器

### 🛠️ 通用调试命令

```cmd
:: 检查完整环境
check_env.bat

:: 单独测试 YOLO 模型加载
python -c "from ultralytics import YOLO; m = YOLO('./best.pt'); print('classes:', m.names)"

:: 单独测试视频读取
python -c "import cv2; cap=cv2.VideoCapture('./ball.avi'); ret,f=cap.read(); print('OK:', ret, f.shape if ret else 'N/A')"
```

---

## 📊 性能参考

| 硬件配置      | 预处理（Phase 1–5） | 回放 FPS（含并行 stereo） | 延迟     |
| ------------- | ------------------- | ------------------------- | -------- |
| RTX 4090 + i9 | ~45 s（3600 帧）    | ~65 FPS                   | < 16 ms  |
| RTX 3080 + i7 | ~80 s（3600 帧）    | ~42 FPS                   | ~24 ms   |
| RTX 2070 + i7 | ~120 s（3600 帧）   | ~28 FPS                   | ~36 ms   |
| GTX 1660 + i5 | ~180 s（3600 帧）   | ~18 FPS                   | ~55 ms   |
| CPU only (i7) | ~600 s（3600 帧）   | ~3 FPS                    | > 300 ms |

> **推荐**：至少使用 GTX 1660 级别 GPU，实时检测需要 ≥ 20 FPS。
>
> **v2 并行优化效果**：stereo 流水线并行相比 v1 同步方式，回放 FPS 提升约 15–25%。

---

## 📝 更新日志

### v3.0.0（当前版本）

- ★ **Phase 5 抛物线物理分析** — 像素重力估算 + 自由飞行段检测 + 出手帧定位
- ★ 出手检测由 Phase 5 抛物线拟合信号驱动（`is_physics_release`）
- ★ **进球判定 v7**：
  - 网弹容忍 — 球穿过篮筐后弹网回弹，X 仍在篮筐范围判进球
  - 框弹多穿越感知 — 穿越 ≥2 次放宽判定标准
  - 弹跳延长跟踪 — 篮筐区域弹跳自动延长超时 25 帧
  - 活跃弹跳期间不触发任何终止条件
- ★ **stereo 流水线并行加速**：
  - SGBM 立体匹配移至后台线程（`ThreadPoolExecutor`）
  - OpenCV C++ 释放 GIL，主线程零等待
  - 无检测帧跳过 `reprojectImageTo3D`
- \+ Phase 5 性能优化：Vandermonde 伪逆预计算 + 矩阵批量拟合
- \+ Phase 1 批量 YOLO 推理（`detect_batch_size=8`）
- \+ Phase 3 numpy 向量化中值滤波（`sliding_window_view`）
- \+ `PhysicsMetadata` 数据模型
- \+ `VisionEngine.shutdown()` 后台线程清理
- \+ RELEASE 出手标记可视化（星标 + 圆圈）

### v2.0.0

- \+ 新增离线四阶段预处理管线（逐帧检测 → 跨帧跟踪 → 轨迹平滑 → 插值补帧）
- \+ 五阶段投篮状态机（IDLE / RISING / TRACKING / RESULT / COOLDOWN）
- \+ 空心球 / 打板球 / 弹框球智能判定
- \+ 实时穿越检测 + 快速进球确认（5 帧）
- \+ `install_cpu.bat` CPU 版一键安装脚本
- \+ `check_env.bat` 环境检测工具

### v1.0.0（2025-01）

- \+ 双目立体视觉深度感知
- \+ YOLOv11 篮球 / 篮筐检测
- \+ 鲁棒轨迹分析（IRLS + Huber + SG 平滑）
- \+ Flask 实时 Web 界面
- \+ 三路 MJPEG 视频流
- \+ 进球 / 速度 / 角度实时分析
- \+ Windows 一键安装脚本

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

> **遇到问题？** 查看 [🐛 故障排除](#-故障排除) 章节，或运行 `check_env.bat` 进行环境自检。
>
> Made with ❤️ for Basketball Analytics