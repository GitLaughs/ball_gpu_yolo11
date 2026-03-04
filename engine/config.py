from dataclasses import dataclass

@dataclass
class EngineConfig:
    # video / model
    conf_threshold: float = 0.5
    iou_threshold: float = 0.45
    max_det: int = 300

    # stereo / image size
    width: int = 640
    height: int = 480

    # 3D sample window around bbox center (median depth)
    sample_window: int = 5  # odd number

    # hoop geometry (meters)
    hoop_radius_m: float = 0.15  # 150mm
    hoop_radius_tol: float = 1.25  # radius * tol

    # shot state machine
    min_points: int = 8
    max_points: int = 80
    lost_timeout_s: float = 0.35

    # scoring rule
    rim_y_tol_m: float = 0.08      # 8cm height tolerance
    require_downward: bool = True  # vy < 0 at crossing