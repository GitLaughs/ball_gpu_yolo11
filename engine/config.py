### engine/config.py
from dataclasses import dataclass


@dataclass
class EngineConfig:
    # ─── 视频/模型 ───
    conf_threshold: float = 0.35
    iou_threshold: float = 0.45
    max_det: int = 300

    # ─── 图像尺寸 ───
    width: int = 640
    height: int = 480

    # ─── 深度采样 ───
    sample_window: int = 7

    # ═══ 第一阶段: 离线检测 ═══
    detect_save_jpg: bool = True
    detect_jpg_quality: int = 95

    # ═══ 第一阶段: 跟踪关联 ═══
    track_iou_thresh: float = 0.20
    track_max_gap: int = 15
    track_max_center_dist: int = 120
    track_min_len: int = 3

    # ═══ 第一阶段: 平滑参数 ═══
    median_filter_size: int = 3
    smooth_window: int = 5
    size_smooth_window: int = 9
    interpolate_max_gap: int = 10