# /engine/shot_engine.py
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Dict

import numpy as np

class ShotPhase(Enum):
    IDLE = "idle"
    RISING = "rising"
    TRACKING = "tracking"
    COOLDOWN = "cooldown"

@dataclass
class BallPoint2D:
    frame_id: int
    timestamp: float
    cx: int
    cy: int
    conf: float = 0.0

@dataclass
class ShotEvent:
    is_scored: bool
    trajectory: List[BallPoint2D]
    release_point: Tuple[int, int]
    apex_point: Tuple[int, int]
    end_point: Tuple[int, int]
    hoop_center: Optional[Tuple[int, int]]
    hoop_bbox: Optional[Tuple[int, int, int, int]]
    arc_height_px: float
    duration_s: float
    frame_start: int
    frame_end: int
    result_display_until: int = 0

class ShotEngine:
    def __init__(self, img_w: int = 640, img_h: int = 480):
        self.img_w = img_w
        self.img_h = img_h

        self.phase = ShotPhase.IDLE
        self.ball_history: deque[BallPoint2D] = deque(maxlen=180)
        self.shot_points: List[BallPoint2D] = []

        self.rising_count = 0
        self.apex_cy = 9999
        self.apex_idx = 0
        self._descending_count = 0

        self.last_cx: Optional[int] = None
        self.last_cy: Optional[int] = None
        self.last_frame: int = 0

        self.hoop_raw: deque = deque(maxlen=90)
        self.hoop_center: Optional[Tuple[int, int]] = None
        self.hoop_bbox: Optional[Tuple[int, int, int, int]] = None

        self.cooldown_until_frame = 0

        self.entered_hoop_zone = False
        self.hoop_zone_enter_frame = 0
        self.ball_exited_below = False

        self._bounce_count = 0
        self._last_dy_sign = 0
        self._last_bounce_frame = 0

        self.shot_history: List[ShotEvent] = []
        self.last_event: Optional[ShotEvent] = None
        self.trail: deque = deque(maxlen=120)

        self.total_shots = 0
        self.total_scored = 0

        self.MIN_RISE_SPEED = 1.5
        self.MIN_TOTAL_RISE = 40
        self.MAX_FRAME_GAP = 30
        self.MAX_FRAME_GAP_NEAR_HOOP = 80
        self.COOLDOWN_FRAMES = 90
        self.MAX_JUMP_PX = 100
        self.RESULT_DISPLAY_FRAMES = 90
        
        self.HOOP_PROXIMITY_X_FACTOR = 1.5
        self.HOOP_PROXIMITY_Y_ABOVE = 1.0
        self.HOOP_PROXIMITY_Y_BELOW = 1.0

        # ★ 全面放宽高弹跳的时间与高度冗余 
        self.TRACKING_HOOP_ZONE_TIMEOUT = 250   # 原来150 -> 250
        self.BOUNCE_EXTEND_FRAMES = 90          # 原来60 -> 90
        self.TRACKING_MAX_FRAMES = 550          # 原来400 -> 550

        self.RISING_NEED_DESCENDING = 3
        self.RISING_NEAR_HOOP_MIN_RISE = 20

    def _update_hoop(self, center, bbox):
        self.hoop_raw.append((center[0], center[1], *bbox))
        if len(self.hoop_raw) >= 3:
            data = list(self.hoop_raw)
            self.hoop_center = (int(np.median([d[0] for d in data])), int(np.median([d[1] for d in data])))
            self.hoop_bbox = (int(np.median([d[2] for d in data])), int(np.median([d[3] for d in data])), 
                              int(np.median([d[4] for d in data])), int(np.median([d[5] for d in data])))

    def _is_near_hoop(self, cx: int, cy: int) -> bool:
        if not self.hoop_bbox: return False
        hx1, hy1, hx2, hy2 = self.hoop_bbox
        hw, hh = max(hx2 - hx1, 1), max(hy2 - hy1, 1)
        mx, my_up, my_dn = hw * 1.5, hh * 1.0, hh * 1.0
        return (hx1 - mx) < cx < (hx2 + mx) and (hy1 - my_up) < cy < (hy2 + my_dn)

    def _check_scoring(self, traj: List[BallPoint2D]) -> bool:
        if not self.hoop_center or not self.hoop_bbox or len(traj) < 3: return False

        hx1, hy1, hx2, hy2 = self.hoop_bbox
        hw, hh = max(hx2 - hx1, 1), max(hy2 - hy1, 1)
        rim_y = hy1 + hh * 0.45
        x_margin_wide = hw * 0.8  # 放宽X边界用于抗划出判定
        x_margin = hw * 0.45

        # 【Step 1】直接执行严格侧边偏出(Slip Out)判定，防止假进球！
        slipped_out = False
        s_idx = next((i for i, p in enumerate(traj) if p.cy <= (min(t.cy for t in traj) + 5)), 0)
        for p in traj[s_idx:]:
            if rim_y <= p.cy <= hy2 + hh * 0.2:
                if p.cx < (hx1 - x_margin_wide) or p.cx > (hx2 + x_margin_wide):
                    slipped_out = True
                    break
        if slipped_out:
            return False

        # 【Step 2】绝对穿透检测：修复"穿透上沿后减速的空心球无法被确认的问题"
        passed_top_idx = -1
        passed_bot_idx = -1
        for i, p in enumerate(traj):
            if passed_top_idx == -1 and p.cy < hy1 + hh * 0.3 and (hx1 - x_margin_wide) < p.cx < (hx2 + x_margin_wide):
                passed_top_idx = i
            if p.cy > hy2 - hh * 0.1 and (hx1 - x_margin_wide) < p.cx < (hx2 + x_margin_wide):
                passed_bot_idx = i  # 取落入网底后最新覆盖的一点
                
        if passed_top_idx != -1 and passed_bot_idx != -1 and passed_bot_idx > passed_top_idx:
            inner_slip = False
            for p in traj[passed_top_idx:passed_bot_idx+1]:
                if p.cx < (hx1 - x_margin_wide) or p.cx > (hx2 + x_margin_wide):
                    inner_slip = True
                    break
            if not inner_slip:
                return True

        # 【Step 3】净下穿跳变推演（对应常规情况）
        down_crossings = []
        up_crossings = []
        for i in range(1, len(traj)):
            pr, cu = traj[i - 1], traj[i]
            if pr.cy <= rim_y < cu.cy: down_crossings.append(i)
            elif pr.cy >= rim_y > cu.cy: up_crossings.append(i)

        final_p = traj[-1]
        tail_near_or_below = (sum(p.cy for p in traj[-8:]) / min(len(traj), 8)) >= rim_y - hh * 0.3
        
        net_down = len(down_crossings) - len(up_crossings)
        if net_down >= 1:
            if final_p.cy < rim_y - hh * 1.5 and not ((hx1 - x_margin_wide) < final_p.cx < (hx2 + x_margin_wide)):
                return False
            return True

        last_dir_is_down = True if down_crossings and (not up_crossings or down_crossings[-1] > up_crossings[-1]) else False
        if last_dir_is_down:
            post = traj[down_crossings[-1]:] if down_crossings else []
            if post and min(p.cy for p in post) >= rim_y - hh * 0.8: return True
            if tail_near_or_below: return True

        if len(down_crossings) + len(up_crossings) >= 3 and tail_near_or_below: return True
        return False

    def _finalize_shot(self, frame_id: int) -> Optional[ShotEvent]:
        traj = self.shot_points
        if len(traj) < 10 or (traj[0].cy - min(p.cy for p in traj)) < self.MIN_TOTAL_RISE:
            self.phase = ShotPhase.IDLE; self.shot_points = []; return None

        is_scored = self._check_scoring(traj)
        apex = min(traj, key=lambda p: p.cy)
        
        evt = ShotEvent(
            is_scored=is_scored, trajectory=list(traj),
            release_point=(traj[0].cx, traj[0].cy),
            apex_point=(apex.cx, apex.cy), end_point=(traj[-1].cx, traj[-1].cy),
            hoop_center=self.hoop_center, hoop_bbox=self.hoop_bbox,
            arc_height_px=traj[0].cy - apex.cy, duration_s=traj[-1].timestamp - traj[0].timestamp if len(traj)>1 else 0,
            frame_start=traj[0].frame_id, frame_end=frame_id,
            result_display_until=frame_id + self.RESULT_DISPLAY_FRAMES,
        )

        self.shot_history.append(evt); self.last_event = evt
        self.total_shots += 1; self.total_scored += int(is_scored)
        self.phase = ShotPhase.COOLDOWN; self.cooldown_until_frame = frame_id + self.COOLDOWN_FRAMES
        
        self.shot_points = []; self.entered_hoop_zone = False; self._bounce_count = 0
        return evt

    def update(self, frame_id: int, timestamp: float, ball_det=None, hoop_det=None, is_physics_release=False) -> Optional[ShotEvent]:
        if hoop_det: self._update_hoop(hoop_det.center, hoop_det.bbox)

        if self.phase == ShotPhase.COOLDOWN:
            if frame_id >= self.cooldown_until_frame: self.phase = ShotPhase.IDLE
            if ball_det: self.trail.append((ball_det.center[0], ball_det.center[1], frame_id))
            return None

        if ball_det is not None:
            cx, cy = ball_det.center
            self.trail.append((cx, cy, frame_id))
            bp = BallPoint2D(frame_id, timestamp, cx, cy, getattr(ball_det, "conf", 0))

            if self.last_cx and (abs(cx - self.last_cx) + abs(cy - self.last_cy)) > self.MAX_JUMP_PX and (frame_id - self.last_frame) <= 2:
                self.last_frame = frame_id; return None

            self.ball_history.append(bp)
            result = None

            if self.phase == ShotPhase.IDLE:
                if is_physics_release and not self._is_near_hoop(cx, cy):
                    self.phase = ShotPhase.RISING; self.shot_points = [bp]; self.apex_cy = cy

            elif self.phase == ShotPhase.RISING:
                self.shot_points.append(bp)
                if bp.cy < self.apex_cy: self.apex_cy = bp.cy
                if self._descending_count >= self.RISING_NEED_DESCENDING or len(self.shot_points) > 50 or self._is_near_hoop(bp.cx, bp.cy):
                    self.phase = ShotPhase.TRACKING; self.entered_hoop_zone = False

            elif self.phase == ShotPhase.TRACKING:
                result = self._state_tracking(bp, frame_id)

            self.last_cx, self.last_cy, self.last_frame = cx, cy, frame_id
            return result
        else:
            if self.phase in (ShotPhase.RISING, ShotPhase.TRACKING) and (frame_id - self.last_frame) > (self.MAX_FRAME_GAP_NEAR_HOOP if self.entered_hoop_zone else self.MAX_FRAME_GAP):
                return self._finalize_shot(frame_id) if self.phase == ShotPhase.TRACKING else self._reset()
        return None

    def _reset(self):
        self.phase = ShotPhase.IDLE; self.shot_points = []; self.entered_hoop_zone = False
        
    def _state_tracking(self, bp: BallPoint2D, frame_id: int) -> Optional[ShotEvent]:
        self.shot_points.append(bp)
        if bp.cy < self.apex_cy: self.apex_cy = bp.cy

        if self.hoop_bbox and self.hoop_center:
            hx1, hy1, hx2, hy2 = self.hoop_bbox
            hw, hh = max(hx2 - hx1, 1), max(hy2 - hy1, 1)

            # ★ 给足了空间处理向上的反向跳变和超出
            in_zone = (hx1 - hw * 1.5 < bp.cx < hx2 + hw * 1.5) and (hy1 - hh * 3.5 < bp.cy < hy2 + hh * 3.0)
            if in_zone and not self.entered_hoop_zone:
                self.entered_hoop_zone = True; self.hoop_zone_enter_frame = frame_id

            if len(self.shot_points) >= 2:
                dy = bp.cy - self.shot_points[-2].cy
                dsign = 1 if dy > +0.5 else (-1 if dy < -0.5 else 0)
                if dsign != 0 and self._last_dy_sign != 0 and dsign != self._last_dy_sign:
                    self._bounce_count += 1; self._last_bounce_frame = frame_id
                    if self.entered_hoop_zone: # 每发生弹跳往死里顺延时间
                        self.hoop_zone_enter_frame = max(self.hoop_zone_enter_frame, frame_id - self.TRACKING_HOOP_ZONE_TIMEOUT + self.BOUNCE_EXTEND_FRAMES)
                if dsign != 0: self._last_dy_sign = dsign

            is_bouncing = self._bounce_count > 0 and (frame_id - self._last_bounce_frame < 50) and in_zone
            if is_bouncing: return None

            if self.entered_hoop_zone and (frame_id - self.hoop_zone_enter_frame > self.TRACKING_HOOP_ZONE_TIMEOUT):
                return self._finalize_shot(frame_id)

            if bp.cy > hy2 + hh * 0.8: return self._finalize_shot(frame_id)

        if len(self.shot_points) > self.TRACKING_MAX_FRAMES: return self._finalize_shot(frame_id)
        return None

    def get_display_info(self, frame_id: int) -> Dict:
        return {
            "phase": self.phase.value, "trail": list(self.trail),
            "shot_trajectory": [(p.cx, p.cy) for p in self.shot_points],
            "physics_release_point": (self.shot_points[0].cx, self.shot_points[0].cy) if self.shot_points else None,
            "hoop_bbox": self.hoop_bbox, "total_shots": self.total_shots, "total_scored": self.total_scored,
            "showing_result": self.last_event and frame_id <= self.last_event.result_display_until,
            "result_text": "SCORED!" if self.last_event and self.last_event.is_scored else "MISSED",
            "result_color": (0, 255, 0) if self.last_event and self.last_event.is_scored else (0, 0, 255)
        }