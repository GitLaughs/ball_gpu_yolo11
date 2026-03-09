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

    # ═══ Phase 1: 离线检测 ═══
    detect_save_jpg: bool = False
    detect_jpg_quality: int = 95
    detect_batch_size: int = 8

    # ═══ Phase 2: 跟踪关联 ═══
    track_iou_thresh: float = 0.20
    track_max_gap: int = 15
    track_max_center_dist: int = 120
    track_min_len: int = 3

    # ═══ Phase 3: 平滑参数 ═══
    median_filter_size: int = 3
    smooth_window: int = 5
    size_smooth_window: int = 9
    interpolate_max_gap: int = 10

    # ═══ Phase 5: 物理分析参数 ═══
    # ── 像素重力估算 ──
    gravity_fit_window_sec: float = 0.30
    gravity_fit_r2_min: float = 0.95
    gravity_min_px_per_frame2: float = 0.05
    gravity_max_px_per_frame2: float = 2.0

    # ── 自由飞行检测 ──
    horiz_accel_ratio: float = 0.20
    free_flight_window_sec: float = 0.15
    free_flight_r2_min: float = 0.90

    # ── 出手帧筛选 ──
    release_hoop_margin_x: float = 1.5
    release_hoop_margin_y_up: float = 1.0
    release_hoop_margin_y_dn: float = 1.5
    release_bounce_consec_frames: int = 3
    release_bounce_fall_speed: float = 3.0
    release_min_rise_px: float = 30
    release_min_cy_ratio: float = 0.60      # ★ 画面下方 40% 不检测出手 (60% 以上才检测)