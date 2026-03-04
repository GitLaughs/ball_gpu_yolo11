from __future__ import annotations
import math
import time
from typing import Optional, Tuple, List

from trajectory_analyzer import TrajectoryAnalyzer
from engine.config import EngineConfig
from engine.models import FrameDetections, ShotResult, ShotState, Detection3D

class ShotEngine:
    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self.analyzer = TrajectoryAnalyzer()
        self.state = ShotState()
        self.last_result: Optional[ShotResult] = None

    def _avg_hoop(self) -> Optional[Tuple[float, float, float]]:
        if not self.state.hoop_history:
            return None
        xs = [p[0] for p in self.state.hoop_history]
        ys = [p[1] for p in self.state.hoop_history]
        zs = [p[2] for p in self.state.hoop_history]
        return (sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs))

    def _pick_best(self, dets: List[Detection3D], cls: str) -> Optional[Detection3D]:
        cands = [d for d in dets if d.cls == cls]
        if not cands:
            return None
        return max(cands, key=lambda d: d.conf)

    def _check_score_crossing(self, hoop: Tuple[float,float,float], points: List[Tuple[float,float,float,float]]) -> bool:
        """判定：球是否从上往下穿过 hoop_y，并且穿越时 xz 距离在篮筐半径内"""
        if len(points) < 2:
            return False
        hx, hy, hz = hoop
        r = self.cfg.hoop_radius_m * self.cfg.hoop_radius_tol

        for i in range(1, len(points)):
            t0, x0, y0, z0 = points[i-1]
            t1, x1, y1, z1 = points[i]
            # 是否跨越 hoop_y
            if (y0 - hy) * (y1 - hy) > 0:
                continue

            # 线性插值求穿越点
            denom = (y1 - y0)
            if abs(denom) < 1e-6:
                continue
            a = (hy - y0) / denom
            if not (0.0 <= a <= 1.0):
                continue
            xc = x0 + a * (x1 - x0)
            zc = z0 + a * (z1 - z0)

            # 半径门限（只在 x-z 平面）
            dx = xc - hx
            dz = zc - hz
            if math.sqrt(dx*dx + dz*dz) > r:
                continue

            # 高度容差（防止误差导致错过）
            # 实际穿越点就是 hy，这里主要给 debug 时保留
            # 向下速度约束（可选）
            if self.cfg.require_downward:
                dt = t1 - t0
                if dt <= 1e-6:
                    continue
                vy = (y1 - y0) / dt
                if vy >= 0:
                    continue

            return True
        return False

    def update(self, fd: FrameDetections) -> Optional[ShotResult]:
        """每帧喂一次；若在本次 update 里"结束投篮并完成分析"，返回 ShotResult，否则返回 None"""
        now = fd.timestamp

        hoop_det = self._pick_best(fd.detections, "hoop")
        if hoop_det is not None:
            # 检测到篮筐，更新位置并重置缓冲区
            self.state.hoop_history.append(hoop_det.xyz)
            if len(self.state.hoop_history) > 10:
                self.state.hoop_history = self.state.hoop_history[-10:]
            self.state.last_hoop_position = hoop_det.xyz
            self.state.hoop_buffer_frames = self.state.hoop_buffer_limit
        else:
            # 没检测到篮筐，减少缓冲区计数
            if self.state.hoop_buffer_frames > 0:
                self.state.hoop_buffer_frames -= 1

        ball_det = self._pick_best(fd.detections, "basketball")

        if ball_det is not None:
            # seen ball
            self.state.last_seen_ts = now
            if not self.state.is_active:
                self.state.is_active = True
                self.state.points = []

            x, y, z = ball_det.xyz
            self.state.points.append((now, x, y, z))
            if len(self.state.points) > self.cfg.max_points:
                self.state.points = self.state.points[-self.cfg.max_points:]
            return None

        # no ball in this frame
        if not self.state.is_active:
            return None

        # active but lost
        if (now - self.state.last_seen_ts) < self.cfg.lost_timeout_s:
            return None

        # finalize
        points = self.state.points
        # 优先使用缓存的篮筐位置，其次使用历史平均值，最后使用默认值
        hoop = None
        if self.state.hoop_buffer_frames > 0 and self.state.last_hoop_position is not None:
            hoop = self.state.last_hoop_position
        else:
            hoop = self._avg_hoop() or (0.0, 3.05, 5.0)  # 你原来也有默认篮筐高度/位置回退

        if len(points) < self.cfg.min_points:
            # discard
            self.state.is_active = False
            self.state.points = []
            return None

        # trajectory_analyzer：沿用你当前的格式（position_data=[x,z], hoop_position=[x,z]） :contentReference[oaicite:8]{index=8}
        traj_data = [[p[1], p[3]] for p in points]
        hoop_xz = [hoop[0], hoop[2]]
        analysis = self.analyzer.analyze_shot(position_data=traj_data, hoop_position=hoop_xz) or {}

        predicted = analysis.get("predicted_trajectory", [])
        actual = analysis.get("actual", {}) if isinstance(analysis.get("actual", {}), dict) else {}

        shot_speed = float(actual.get("velocity", 0.0))
        shot_angle = float(actual.get("angle", 0.0))

        # 出手点：取第一个点
        _, sx, sy, sz = points[0]
        shot_pos = (sx, sy, sz)

        # 进球判定：过筐高度平面 + 半径门限 + (可选)向下速度
        is_scored = self._check_score_crossing(hoop, points)

        res = ShotResult(
            is_scored=is_scored,
            shot_speed=shot_speed,
            shot_angle=shot_angle,
            shot_position=shot_pos,
            hoop_position=hoop,
            analysis_time=time.time(),
            sequence_len=len(points),
            predicted_trajectory=predicted if isinstance(predicted, list) else [],
            actual_trajectory=traj_data,
            trajectory_analysis=analysis if isinstance(analysis, dict) else {},
        )
        self.last_result = res

        # reset shot state (关键：只在结束时清空，而不是每帧清空)
        self.state.is_active = False
        self.state.points = []
        return res