from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

@dataclass
class Detection3D:
    cls: str                 # 'basketball' / 'hoop'
    conf: float
    bbox: Tuple[int, int, int, int]   # x1,y1,x2,y2
    center: Tuple[int, int]           # cx,cy
    xyz: Tuple[float, float, float]   # meters
    distance: float

@dataclass
class FrameDetections:
    frame_id: int
    timestamp: float
    detections: List[Detection3D] = field(default_factory=list)

@dataclass
class ShotResult:
    is_scored: bool
    shot_speed: float
    shot_angle: float
    shot_position: Tuple[float, float, float]
    hoop_position: Tuple[float, float, float]
    analysis_time: float
    sequence_len: int
    predicted_trajectory: List[List[float]] = field(default_factory=list)  # [[x,z],...]
    actual_trajectory: List[List[float]] = field(default_factory=list)  # [[x,z],...]
    trajectory_analysis: Dict = field(default_factory=dict)

@dataclass
class ShotState:
    is_active: bool = False
    last_seen_ts: float = 0.0
    points: List[Tuple[float, float, float, float]] = field(default_factory=list)  # (t,x,y,z)
    hoop_history: List[Tuple[float, float, float]] = field(default_factory=list)   # (x,y,z)
    last_hoop_position: Optional[Tuple[float, float, float]] = None  # 最后检测到的篮筐位置
    hoop_buffer_frames: int = 0  # 篮筐位置缓存的帧数
    hoop_buffer_limit: int = 60  # 缓存限制（60帧约2秒）