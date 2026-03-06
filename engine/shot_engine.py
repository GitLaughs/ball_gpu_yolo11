"""
2D 像素坐标驱动的投篮检测引擎 (v5 — 修复RISING误重置 & 弹地误判release)

核心改进 (v5):
1. RISING抗干扰：使用连续下降帧计数器，不再因单帧速度波动而误判
2. 篮筐附近保护：RISING阶段球在篮筐附近时强制转入TRACKING，不会被重置
3. 弹地过滤增强：检测上升前是否有快速下落（弹地特征），从源头阻止误检
4. Release point可靠性：仅从真正的持球/出手动作开始记录轨迹
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Dict

import numpy as np


# ========== 数据结构 ==========

class ShotPhase(Enum):
    IDLE     = "idle"
    RISING   = "rising"
    TRACKING = "tracking"
    RESULT   = "result"
    COOLDOWN = "cooldown"


@dataclass
class BallPoint2D:
    frame_id: int
    timestamp: float
    cx: int
    cy: int
    conf: float = 0.0
    x3d: float = 0.0
    y3d: float = 0.0
    z3d: float = 0.0


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


# ========== 主引擎 ==========

class ShotEngine:
    def __init__(self, img_w: int = 640, img_h: int = 480):
        self.img_w = img_w
        self.img_h = img_h

        # ─── 状态 ─── #
        self.phase = ShotPhase.IDLE

        # ─── 轨迹缓冲 ─── #
        self.ball_history: deque[BallPoint2D] = deque(maxlen=180)
        self.shot_points: List[BallPoint2D] = []

        # ─── 计数器 ─── #
        self.rising_count = 0
        self.apex_cy = 9999
        self.apex_idx = 0

        # ─── RISING阶段下降帧计数 (v5新增) ─── #
        self._descending_count = 0

        # ─── 上一帧信息 ─── #
        self.last_cx: Optional[int] = None
        self.last_cy: Optional[int] = None
        self.last_frame: int = 0
        self.last_ball_ts: float = 0.0

        # ─── 篮筐 ─── #
        self.hoop_raw: deque = deque(maxlen=90)
        self.hoop_center: Optional[Tuple[int, int]] = None
        self.hoop_bbox: Optional[Tuple[int, int, int, int]] = None

        # ─── 冷却 ─── #
        self.cooldown_until_frame = 0

        # ─── 跟踪阶段附加状态 ─── #
        self.entered_hoop_zone = False
        self.hoop_zone_enter_frame = 0
        self.ball_exited_below = False

        # ─── 实时穿越检测状态 (TRACKING 阶段使用) ─── #
        self._rt_crossing_active = False
        self._rt_crossing_frame = -1
        self._rt_consecutive_below = 0
        self._rt_crossing_invalidated = False
        self._rt_total_crossings = 0

        # ─── 历史 ─── #
        self.shot_history: List[ShotEvent] = []
        self.last_event: Optional[ShotEvent] = None

        # ─── 可视化轨迹 ─── #
        self.trail: deque = deque(maxlen=120)

        # ─── 得分统计 ─── #
        self.total_shots = 0
        self.total_scored = 0

        # ═══ 阈值 ═══ #
        self.MIN_RISE_FRAMES = 4
        self.MIN_RISE_SPEED = 1.5
        self.MIN_TOTAL_RISE = 40
        self.MAX_FRAME_GAP = 30
        self.MIN_TRAJ_FRAMES = 10
        self.COOLDOWN_FRAMES = 90
        self.MAX_JUMP_PX = 100
        self.RESULT_DISPLAY_FRAMES = 90

        # ── 出手起始位置限制 ── #
        self.MIN_START_CY_RATIO = 0.4
        self.MAX_START_CY_RATIO = 0.85

        # ── 弹地过滤 ── #
        self.GROUND_BOUNCE_MIN_CY_RATIO = 0.82

        # ── 篮筐附近抑制 ── #
        self.HOOP_PROXIMITY_X_FACTOR = 1.5
        self.HOOP_PROXIMITY_Y_ABOVE = 1.0
        self.HOOP_PROXIMITY_Y_BELOW = 1.0

        # ── 进球判定 ── #
        self.SCORE_X_TOL = 0.40
        self.SCORE_BOUNCE_BACK_TOL = 0.45
        self.SCORE_EXIT_BELOW_RATIO = 0.20

        # ── 实时快速进球确认 ── #
        self.QUICK_SCORE_FRAMES = 2
        self.RT_BOUNCE_BACK_MARGIN = 0.35

        # ── 跟踪阶段 ── #
        self.TRACKING_HOOP_ZONE_TIMEOUT = 80
        self.TRACKING_MAX_FRAMES = 240

        # ── RISING阶段下降容忍 (v5新增) ── #
        self.RISING_NEED_DESCENDING = 3       # 需要连续 N 帧下降才判定球开始下落
        self.RISING_NEAR_HOOP_MIN_RISE = 20   # 篮筐附近时的最低上升量

        # ── 弹地检测 - 回落速度阈值 (v5新增) ── #
        self.BOUNCE_FALL_FRAMES_MIN = 2       # 上升前至少连续下落帧数
        self.BOUNCE_FALL_SPEED_MIN = 4.0      # 上升前下落速度阈值(像素/帧)

    # ─────────── 篮筐稳定化 ───────────
    def _update_hoop(self, center, bbox):
        cx, cy = center
        x1, y1, x2, y2 = bbox
        self.hoop_raw.append((cx, cy, x1, y1, x2, y2))
        if len(self.hoop_raw) >= 3:
            data = list(self.hoop_raw)
            self.hoop_center = (
                int(np.median([d[0] for d in data])),
                int(np.median([d[1] for d in data])),
            )
            self.hoop_bbox = (
                int(np.median([d[2] for d in data])),
                int(np.median([d[3] for d in data])),
                int(np.median([d[4] for d in data])),
                int(np.median([d[5] for d in data])),
            )

    # ─────────── 判断球是否在篮筐附近 ───────────
    def _is_near_hoop(self, cx: int, cy: int) -> bool:
        if not self.hoop_bbox or not self.hoop_center:
            return False
        hx1, hy1, hx2, hy2 = self.hoop_bbox
        hw = max(hx2 - hx1, 1)
        hh = max(hy2 - hy1, 1)

        margin_x = hw * self.HOOP_PROXIMITY_X_FACTOR
        margin_y_up = hh * self.HOOP_PROXIMITY_Y_ABOVE
        margin_y_dn = hh * self.HOOP_PROXIMITY_Y_BELOW

        x_near = (hx1 - margin_x) < cx < (hx2 + margin_x)
        y_near = (hy1 - margin_y_up) < cy < (hy2 + margin_y_dn)
        return x_near and y_near

    # ─────────── 判断球是否在篮筐广域区域 (v5新增，用于RISING保护) ───────────
    def _is_in_hoop_region(self, cx: int, cy: int) -> bool:
        """比 _is_near_hoop 更宽松的区域检测，用于 RISING 阶段保护"""
        if not self.hoop_bbox or not self.hoop_center:
            return False
        hx1, hy1, hx2, hy2 = self.hoop_bbox
        hw = max(hx2 - hx1, 1)
        hh = max(hy2 - hy1, 1)

        margin_x = hw * 2.0
        margin_y_up = hh * 1.5
        margin_y_dn = hh * 2.0

        x_near = (hx1 - margin_x) < cx < (hx2 + margin_x)
        y_near = (hy1 - margin_y_up) < cy < (hy2 + margin_y_dn)
        return x_near and y_near

    # ─────────── 弹地/弹起前下落检测 (v5新增) ───────────
    def _was_falling_before_rise(self, start_idx: int) -> bool:
        """
        检查 ball_history 中 start_idx 之前的若干帧，判断球是否在快速下落。
        投篮起手通常是比较缓和的下沉，而拍球弹地或接掉落球前会有明显的高速下落过程。
        """
        if start_idx < 3:
            return False

        check_start = max(0, start_idx - 10)
        fall_frames = 0
        max_fall_speed = 0.0

        for i in range(check_start, start_idx):
            if i + 1 >= len(self.ball_history):
                break
            p_prev = self.ball_history[i]
            p_curr = self.ball_history[i + 1]
            frame_gap = p_curr.frame_id - p_prev.frame_id
            if frame_gap <= 0 or frame_gap > 5:
                continue
            
            # 每帧平均下落速度（正值代表下落）
            dy_per_frame = (p_curr.cy - p_prev.cy) / frame_gap
            if dy_per_frame > 2.0:
                fall_frames += 1
                max_fall_speed = max(max_fall_speed, dy_per_frame)

        if (fall_frames >= self.BOUNCE_FALL_FRAMES_MIN and
                max_fall_speed >= self.BOUNCE_FALL_SPEED_MIN):
            return True

        return False

    # ─────────── 进球判定 (全轨迹扫描) ───────────
    def _check_scoring(self, traj: List[BallPoint2D]) -> bool:
        """
        进球判定 — 全轨迹多穿越扫描, 从最后一次穿越开始检查:
        1. 收集所有 "从上到下穿越篮筐平面" 且 X 坐标对齐的事件
        2. 从最后一次穿越开始检查反弹和空心状况
        """
        if not self.hoop_center or not self.hoop_bbox:
            return False
        if len(traj) < 5:
            return False

        hcx, hcy = self.hoop_center
        hx1, hy1, hx2, hy2 = self.hoop_bbox
        hw = max(hx2 - hx1, 1)
        hh = max(hy2 - hy1, 1)

        rim_y = hy1 + hh * 0.45
        x_margin = hw * self.SCORE_X_TOL

        # ── Step 1: 收集所有有效穿越 ──
        crossings: List[int] = []
        for i in range(1, len(traj)):
            prev_p = traj[i - 1]
            curr_p = traj[i]
            if prev_p.cy <= rim_y < curr_p.cy:
                dy = curr_p.cy - prev_p.cy
                alpha = (rim_y - prev_p.cy) / dy if dy > 0 else 0.0
                cross_x = prev_p.cx + alpha * (curr_p.cx - prev_p.cx)
                if hx1 - x_margin < cross_x < hx2 + x_margin:
                    crossings.append(i)

        if not crossings:
            return False

        # ── Step 2 & 3: 从最后一次穿越开始检查 ──
        bounce_back_limit = rim_y - hh * self.SCORE_BOUNCE_BACK_TOL

        for cross_idx in reversed(crossings):
            post = traj[cross_idx:]

            if len(post) == 0:
                return True

            final_p = post[-1]
            if final_p.cy > rim_y + hh * 0.3:
                return True

            min_cy_after = min(p.cy for p in post)
            if min_cy_after < bounce_back_limit:
                continue

            if len(post) >= 2:
                downward_movement = True
                for i in range(1, len(post)):
                    if post[i].cy < post[i-1].cy - 1:
                        downward_movement = False
                        break
                if downward_movement:
                    return True

            if len(post) <= 6:
                return True

            tail_n = min(4, len(post))
            avg_cy_tail = sum(p.cy for p in post[-tail_n:]) / tail_n
            exit_line = rim_y + hh * self.SCORE_EXIT_BELOW_RATIO

            if avg_cy_tail >= exit_line:
                return True

            below_count = sum(1 for p in post if p.cy >= rim_y)
            if below_count >= len(post) * 0.45:
                return True

            continue

        return False

    # ─────────── 弹地检测 ───────────
    def _is_ground_bounce(self, traj: List[BallPoint2D]) -> bool:
        if not traj:
            return False

        start_cy = traj[0].cy
        max_cy = max(p.cy for p in traj)

        if start_cy > self.img_h * self.MAX_START_CY_RATIO:
            return True

        if max_cy > self.img_h * self.GROUND_BOUNCE_MIN_CY_RATIO:
            return True

        return False

    # ─────────── 有效出手验证 ───────────
    def _validate_shot(self, traj: List[BallPoint2D]) -> bool:
        if len(traj) < self.MIN_TRAJ_FRAMES:
            return False

        start_cy = traj[0].cy
        apex_cy = min(p.cy for p in traj)
        total_rise = start_cy - apex_cy

        if total_rise < self.MIN_TOTAL_RISE:
            return False

        if self._is_ground_bounce(traj):
            return False

        avg_cy = sum(p.cy for p in traj) / len(traj)
        if avg_cy > self.img_h * 0.85 and not self.entered_hoop_zone:
            return False

        big_jumps = 0
        for i in range(1, len(traj)):
            dx = abs(traj[i].cx - traj[i - 1].cx)
            dy = abs(traj[i].cy - traj[i - 1].cy)
            if dx + dy > self.MAX_JUMP_PX:
                big_jumps += 1
        if big_jumps > len(traj) * 0.2:
            return False

        if apex_cy > self.img_h * 0.65:
            return False

        if self.hoop_center:
            if start_cy < self.hoop_center[1] - 20:
                return False

        if len(traj) >= 2:
            duration = traj[-1].timestamp - traj[0].timestamp
            if 0 < duration < 0.25:
                return False

        return True

    # ─────────── 结束投篮并分析 ───────────
    def _finalize_shot(self, frame_id: int) -> Optional[ShotEvent]:
        traj = self.shot_points

        if not self._validate_shot(traj):
            self._reset()
            return None

        apex_idx = min(range(len(traj)), key=lambda i: traj[i].cy)
        apex = traj[apex_idx]
        start = traj[0]
        end = traj[-1]

        is_scored = self._check_scoring(traj)

        arc_height = start.cy - apex.cy
        duration = (end.timestamp - start.timestamp) if len(traj) > 1 else 0

        event = ShotEvent(
            is_scored=is_scored,
            trajectory=list(traj),
            release_point=(start.cx, start.cy),
            apex_point=(apex.cx, apex.cy),
            end_point=(end.cx, end.cy),
            hoop_center=self.hoop_center,
            hoop_bbox=self.hoop_bbox,
            arc_height_px=arc_height,
            duration_s=duration,
            frame_start=traj[0].frame_id,
            frame_end=frame_id,
            result_display_until=frame_id + self.RESULT_DISPLAY_FRAMES,
        )

        self.shot_history.append(event)
        self.last_event = event
        self.total_shots += 1
        if is_scored:
            self.total_scored += 1

        self.phase = ShotPhase.COOLDOWN
        self.cooldown_until_frame = frame_id + self.COOLDOWN_FRAMES
        self._clear_shot_state()

        return event

    def _clear_shot_state(self):
        self.shot_points = []
        self.rising_count = 0
        self.apex_cy = 9999
        self.apex_idx = 0
        self._descending_count = 0       # v5新增: 下降计数归零
        self.entered_hoop_zone = False
        self.hoop_zone_enter_frame = 0
        self.ball_exited_below = False
        self._rt_crossing_active = False
        self._rt_crossing_frame = -1
        self._rt_consecutive_below = 0
        self._rt_crossing_invalidated = False
        self._rt_total_crossings = 0

    def _reset(self):
        self.phase = ShotPhase.IDLE
        self._clear_shot_state()


    # ═══════════════ 主更新接口 ═══════════════
    def update(self, frame_id: int, timestamp: float,
               ball_det=None, hoop_det=None) -> Optional[ShotEvent]:
        result = None

        if hoop_det is not None:
            self._update_hoop(hoop_det.center, hoop_det.bbox)

        # ── 冷却阶段 ──
        if self.phase == ShotPhase.COOLDOWN:
            if frame_id >= self.cooldown_until_frame:
                if ball_det is not None and self._is_near_hoop(
                        ball_det.center[0], ball_det.center[1]):
                    self.cooldown_until_frame = frame_id + 10
                else:
                    self._reset()
            if ball_det is not None:
                self.trail.append((ball_det.center[0], ball_det.center[1], frame_id))
            return None

        # ── 处理篮球 ──
        if ball_det is not None:
            cx, cy = ball_det.center
            self.trail.append((cx, cy, frame_id))

            # 跳变过滤
            if self.last_cx is not None:
                dx = abs(cx - self.last_cx)
                dy = abs(cy - self.last_cy)
                frame_gap = frame_id - self.last_frame
                if frame_gap <= 2 and dx + dy > self.MAX_JUMP_PX:
                    self.last_frame = frame_id
                    return None

            bp = BallPoint2D(
                frame_id=frame_id,
                timestamp=timestamp,
                cx=cx,
                cy=cy,
                conf=getattr(ball_det, "conf", 0.0),
            )
            if hasattr(ball_det, "xyz"):
                bp.x3d, bp.y3d, bp.z3d = ball_det.xyz

            self.ball_history.append(bp)

            if self.phase == ShotPhase.IDLE:
                result = self._state_idle(bp)
            elif self.phase == ShotPhase.RISING:
                result = self._state_rising(bp)
            elif self.phase == ShotPhase.TRACKING:
                result = self._state_tracking(bp, frame_id)

            self.last_cx = cx
            self.last_cy = cy
            self.last_frame = frame_id
            self.last_ball_ts = timestamp

        else:
            # 没有检测到篮球，判断是否超时丢弃或结算
            if self.phase in (ShotPhase.RISING, ShotPhase.TRACKING):
                gap = frame_id - self.last_frame
                if gap > self.MAX_FRAME_GAP:
                    if self.phase == ShotPhase.TRACKING:
                        result = self._finalize_shot(frame_id)
                    else:
                        self._reset()

        return result


    # ─────────── IDLE 状态 ───────────
    def _state_idle(self, bp: BallPoint2D) -> Optional[ShotEvent]:
        if len(self.ball_history) < 2:
            return None

        if self._is_near_hoop(bp.cx, bp.cy):
            self.rising_count = 0
            return None

        prev = self.ball_history[-2]

        if self._is_near_hoop(prev.cx, prev.cy):
            self.rising_count = 0
            return None

        dy = bp.cy - prev.cy

        if dy < -self.MIN_RISE_SPEED:
            self.rising_count += 1
        else:
            self.rising_count = max(0, self.rising_count - 1)

        if self.rising_count >= self.MIN_RISE_FRAMES:
            start_idx = max(0, len(self.ball_history) - self.rising_count - 2)
            start_bp = self.ball_history[start_idx]

            # ★ v5: 弹地过滤 — 上升前球是否在快速下落
            # 如果是快速下落（如拍球地面反弹），则清空计数忽略它
            if self._was_falling_before_rise(start_idx):
                self.rising_count = 0
                return None

            below_hoop = True
            if self.hoop_center:
                below_hoop = start_bp.cy > self.hoop_center[1]
            else:
                below_hoop = start_bp.cy > self.img_h * self.MIN_START_CY_RATIO

            not_ground_bounce = start_bp.cy < self.img_h * self.MAX_START_CY_RATIO
            not_near_hoop = not self._is_near_hoop(start_bp.cx, start_bp.cy)

            if below_hoop and not_ground_bounce and not_near_hoop:
                self.phase = ShotPhase.RISING
                n = min(self.rising_count + 3, len(self.ball_history))
                self.shot_points = list(self.ball_history)[-n:]
                self.apex_cy = bp.cy
                self.apex_idx = len(self.shot_points) - 1
                self._descending_count = 0  # v5: 重置下降计数
            else:
                self.rising_count = 0

        return None


    # ─────────── RISING 状态 (v5: 抗篮筐干扰) ───────────
    def _state_rising(self, bp: BallPoint2D) -> Optional[ShotEvent]:
        self.shot_points.append(bp)

        if bp.cy < self.apex_cy:
            self.apex_cy = bp.cy
            self.apex_idx = len(self.shot_points) - 1

        if len(self.shot_points) >= 2:
            prev = self.shot_points[-2]
            dy = bp.cy - prev.cy

            # ★ v5: 使用连续下降帧计数，取代单帧 dy > 2 的即时判定
            if dy > 2:
                self._descending_count += 1
            else:
                # 只要有一帧不发生明显下降，就重置计数（要求"连续"下降验证物理下落）
                self._descending_count = 0

            # ★ v5: 检查球是否已经到达篮筐附近
            near_hoop = self._is_near_hoop(bp.cx, bp.cy)
            in_hoop_region = self._is_in_hoop_region(bp.cx, bp.cy)

            # 在篮筐附近时（含减速区）强制保护并进入下一个逻辑流程，决不半路夭折复位IDLE
            if near_hoop or in_hoop_region:
                total_rise = self.shot_points[0].cy - self.apex_cy
                if total_rise >= self.RISING_NEAR_HOOP_MIN_RISE:
                    self.phase = ShotPhase.TRACKING
                    self.entered_hoop_zone = True
                    self.hoop_zone_enter_frame = bp.frame_id
                    self.ball_exited_below = False
                    self._rt_crossing_active = False
                    self._rt_crossing_frame = -1
                    self._rt_consecutive_below = 0
                    self._rt_crossing_invalidated = False
                    self._rt_total_crossings = 0
                    return None
                
                # 如果没达到最低位移但也进了篮筐区域（可能是极端视角），也绝不重置，等待它离开或继续上升
                return None

            # ★ v5: 如果在野外空间，确认连续下落满足帧数，判定为已越过顶点
            if self._descending_count >= self.RISING_NEED_DESCENDING:
                total_rise = self.shot_points[0].cy - self.apex_cy
                if (total_rise >= self.MIN_TOTAL_RISE and
                        self.apex_cy < self.img_h * 0.65):
                    self.phase = ShotPhase.TRACKING
                    self.entered_hoop_zone = False
                    self.ball_exited_below = False
                    self._rt_crossing_active = False
                    self._rt_crossing_frame = -1
                    self._rt_consecutive_below = 0
                    self._rt_crossing_invalidated = False
                    self._rt_total_crossings = 0
                else:
                    self._reset()

            # 超长追踪保护
            elif len(self.shot_points) > 50:
                total_rise = self.shot_points[0].cy - self.apex_cy
                if (total_rise >= self.MIN_TOTAL_RISE and
                        self.apex_cy < self.img_h * 0.65):
                    self.phase = ShotPhase.TRACKING
                    self.entered_hoop_zone = False
                    self.ball_exited_below = False
                    self._rt_crossing_active = False
                    self._rt_crossing_frame = -1
                    self._rt_consecutive_below = 0
                    self._rt_crossing_invalidated = False
                    self._rt_total_crossings = 0
                else:
                    self._reset()

        return None


    # ─────────── TRACKING 状态 (含实时穿越检测) ───────────
    def _state_tracking(self, bp: BallPoint2D, frame_id: int) -> Optional[ShotEvent]:
        self.shot_points.append(bp)

        if bp.cy < self.apex_cy:
            self.apex_cy = bp.cy
            self.apex_idx = len(self.shot_points) - 1

        # ── 无篮筐信息: 简单规则 ──
        if not (self.hoop_bbox and self.hoop_center):
            if (bp.cy > self.img_h * 0.85 and
                    bp.cy > self.apex_cy + 60):
                return self._finalize_shot(frame_id)
            if len(self.shot_points) > self.TRACKING_MAX_FRAMES:
                return self._finalize_shot(frame_id)
            return None

        hx1, hy1, hx2, hy2 = self.hoop_bbox
        hcx, hcy = self.hoop_center
        hw = max(hx2 - hx1, 1)
        hh = max(hy2 - hy1, 1)
        rim_y = hy1 + hh * 0.45
        x_margin = hw * self.SCORE_X_TOL

        # ── 篮筐区域检测 ──
        zone_x1 = hx1 - hw * 0.8
        zone_x2 = hx2 + hw * 0.8
        zone_y1 = hy1 - hh * 1.2
        zone_y2 = hy2 + hh * 2.5

        in_zone = (zone_x1 < bp.cx < zone_x2 and
                   zone_y1 < bp.cy < zone_y2)

        if in_zone and not self.entered_hoop_zone:
            self.entered_hoop_zone = True
            self.hoop_zone_enter_frame = frame_id

        # ═══ 实时穿越检测 ═══
        if len(self.shot_points) >= 2:
            prev_p = self.shot_points[-2]
            curr_p = self.shot_points[-1]
            if prev_p.cy <= rim_y < curr_p.cy:
                dy_cross = curr_p.cy - prev_p.cy
                alpha = (rim_y - prev_p.cy) / dy_cross if dy_cross > 0 else 0.0
                cross_x = prev_p.cx + alpha * (curr_p.cx - prev_p.cx)
                if hx1 - x_margin < cross_x < hx2 + x_margin:
                    self._rt_crossing_active = True
                    self._rt_crossing_frame = frame_id
                    self._rt_consecutive_below = 1
                    self._rt_crossing_invalidated = False
                    self._rt_total_crossings += 1

        # ═══ 实时穿越后的跟踪 ═══
        if self._rt_crossing_active and not self._rt_crossing_invalidated:
            if frame_id > self._rt_crossing_frame:
                bounce_back_limit_rt = rim_y - hh * self.RT_BOUNCE_BACK_MARGIN

                if bp.cy < bounce_back_limit_rt:
                    self._rt_crossing_active = False
                    self._rt_crossing_invalidated = True

                elif bp.cy >= rim_y:
                    self._rt_consecutive_below += 1
                else:
                    self._rt_consecutive_below = 0

                if self._rt_consecutive_below >= self.QUICK_SCORE_FRAMES:
                    return self._finalize_shot(frame_id)

        # ── 球到达篮筐下方 ──
        if (bp.cy > hy2 + hh * 0.5
                and hx1 - hw * 0.6 < bp.cx < hx2 + hw * 0.6):
            self.ball_exited_below = True

        # ═══ 结束条件 ═══

        # A) 球已通过篮筐且在篮筐下方足够深
        if self.ball_exited_below and bp.cy > hy2 + hh * 0.8:
            return self._finalize_shot(frame_id)

        # B) 球在 hoop zone 后水平远离
        if self.entered_hoop_zone:
            if abs(bp.cx - hcx) > hw * 2.5 and bp.cy > hcy:
                return self._finalize_shot(frame_id)

        # C) 球大幅低于篮筐且从未进入 hoop zone → 打偏了
        if (not self.entered_hoop_zone
                and bp.cy > hcy + self.img_h * 0.25
                and bp.cy > self.apex_cy + 80):
            return self._finalize_shot(frame_id)

        # D) 在 hoop zone 停留超时
        if self.entered_hoop_zone:
            time_in_zone = frame_id - self.hoop_zone_enter_frame
            timeout = self.TRACKING_HOOP_ZONE_TIMEOUT
            if self._rt_crossing_active and not self._rt_crossing_invalidated:
                timeout += 15
            if time_in_zone > timeout:
                return self._finalize_shot(frame_id)

        # E) 总轨迹超长
        if len(self.shot_points) > self.TRACKING_MAX_FRAMES:
            return self._finalize_shot(frame_id)

        return None


    # ─────────── 可视化辅助 ───────────
    def get_display_info(self, frame_id: int) -> Dict:
        info = {
            "phase": self.phase.value,
            "trail": list(self.trail),
            "shot_trajectory": [(p.cx, p.cy) for p in self.shot_points],
            "hoop_center": self.hoop_center,
            "hoop_bbox": self.hoop_bbox,
            "total_shots": self.total_shots,
            "total_scored": self.total_scored,
            "showing_result": False,
            "result_text": "",
            "result_color": (255, 255, 255),
        }

        if self.last_event and frame_id <= self.last_event.result_display_until:
            info["showing_result"] = True
            if self.last_event.is_scored:
                info["result_text"] = "SCORED!"
                info["result_color"] = (0, 255, 0)
            else:
                info["result_text"] = "MISSED"
                info["result_color"] = (0, 0, 255)
            info["result_trajectory"] = [
                (p.cx, p.cy) for p in self.last_event.trajectory
            ]
            info["result_scored"] = self.last_event.is_scored

        return info