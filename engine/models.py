### engine/models.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class RawDetection2D:
    cls: str
    conf: float
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]


@dataclass
class StableDetection2D:
    track_id: int
    cls: str
    conf: float
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    is_interpolated: bool = False


@dataclass
class Detection3D:
    track_id: int
    cls: str
    conf: float
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    xyz: Tuple[float, float, float]
    distance: float
    is_interpolated: bool = False


@dataclass
class FrameDetections:
    frame_id: int
    timestamp: float
    detections: List[Detection3D] = field(default_factory=list)