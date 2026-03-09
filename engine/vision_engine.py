# engine/vision_engine.py
"""
双目视觉引擎 (v2 — 流水线并行加速)

核心优化:
  ● stereo SGBM 计算移至后台线程 (OpenCV C++ 释放 GIL)
  ● 主线程用前一帧的深度结果, 实现流水线并行
  ● 无检测帧跳过 reprojectImageTo3D
  ● 保留 Phase 5 is_release_point 标记绘制
"""

from __future__ import annotations

import math
import time
import threading
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, Future

import cv2
import numpy as np

from engine.config import EngineConfig
from engine.models import Detection3D, FrameDetections, StableDetection2D


class StereoParams:
    def __init__(self, Q, left_map1, left_map2, right_map1, right_map2):
        self.Q          = Q
        self.left_map1  = left_map1
        self.left_map2  = left_map2
        self.right_map1 = right_map1
        self.right_map2 = right_map2


class StereoMatcher:
    def __init__(self):
        bs = 8
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=1,
            numDisparities=128,
            blockSize=bs,
            P1=8  * 3 * bs * bs,
            P2=32 * 3 * bs * bs,
            disp12MaxDiff=-1,
            preFilterCap=63,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=1,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    def compute(self, left_gray, right_gray):
        return self.stereo.compute(left_gray, right_gray)


class VisionEngine:
    def __init__(self, stereo: StereoParams, cfg: EngineConfig):
        self.stereo  = stereo
        self.cfg     = cfg
        self.matcher = StereoMatcher()

        # ─── 流水线并行: stereo 后台线程 ───
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="stereo"
        )
        self._pending_future: Optional[Future] = None

        # ─── 缓存上一帧结果 ───
        H, W = cfg.height, cfg.width
        self._cached_depth_color = np.zeros((H, W, 3), dtype=np.uint8)
        self._cached_threeD: Optional[np.ndarray] = None
        self._cached_disp_norm: Optional[np.ndarray] = None

    def shutdown(self):
        """关闭后台线程池"""
        self._executor.shutdown(wait=False)

    # ─── 后台 stereo 计算任务 ───
    def _stereo_task(
        self,
        gray_l: np.ndarray,
        gray_r: np.ndarray,
        need_3d: bool,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        在后台线程中执行 (OpenCV C++ 释放 GIL, 不受 GIL 限制):
          1. remap 矫正
          2. SGBM 立体匹配
          3. 深度着色
          4. (可选) reprojectImageTo3D
        """
        rect_l = cv2.remap(
            gray_l,
            self.stereo.left_map1,
            self.stereo.left_map2,
            cv2.INTER_LINEAR,
        )
        rect_r = cv2.remap(
            gray_r,
            self.stereo.right_map1,
            self.stereo.right_map2,
            cv2.INTER_LINEAR,
        )

        disp = self.matcher.compute(rect_l, rect_r)
        disp_norm = cv2.normalize(
            disp, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
        )
        depth_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)

        threeD = None
        if need_3d:
            threeD = (
                cv2.reprojectImageTo3D(
                    disp, self.stereo.Q, handleMissingValues=True
                )
                * 16
            )
            threeD[np.isinf(threeD)] = 0
            threeD[np.isnan(threeD)] = 0

        return depth_color, disp_norm, threeD

    # ─── 深度采样 ───
    def _median_xyz(self, threeD, cx, cy, win=7):
        r = win // 2
        h, w = threeD.shape[:2]
        x1c, x2c = max(0, cx - r), min(w - 1, cx + r)
        y1c, y2c = max(0, cy - r), min(h - 1, cy + r)
        patch = threeD[y1c : y2c + 1, x1c : x2c + 1, :].reshape(-1, 3)
        mask = np.isfinite(patch).all(axis=1)
        patch = patch[mask]
        if patch.size == 0:
            return 0.0, 0.0, 0.0
        nonzero = np.any(np.abs(patch) > 1e-6, axis=1)
        patch = patch[nonzero]
        if patch.size == 0:
            return 0.0, 0.0, 0.0
        med = np.median(patch, axis=0)
        return float(med[0]), float(med[1]), float(med[2])

    # ─── 主处理接口 ───
    def process_frame(
        self,
        frame: np.ndarray,
        frame_id: int,
        stable_dets_2d: List[StableDetection2D],
        shot_display_info: Dict = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, FrameDetections]:

        ts = time.time()
        W, H = self.cfg.width, self.cfg.height

        frame_left = frame[0:H, 0:W].copy()
        frame_right = frame[0:H, W : W * 2].copy()

        # ── 收取上一帧的 stereo 结果 ──
        if self._pending_future is not None:
            if self._pending_future.done():
                try:
                    depth_color, disp_norm, threeD = (
                        self._pending_future.result()
                    )
                    self._cached_depth_color = depth_color
                    self._cached_disp_norm = disp_norm
                    if threeD is not None:
                        self._cached_threeD = threeD
                except Exception:
                    pass
                self._pending_future = None

        # ── 提交当前帧的 stereo 任务到后台 ──
        need_3d = len(stable_dets_2d) > 0
        if self._pending_future is None:
            gray_l = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
            self._pending_future = self._executor.submit(
                self._stereo_task,
                gray_l.copy(),
                gray_r.copy(),
                need_3d,
            )

        # ── 使用缓存的深度结果 ──
        depth_color = self._cached_depth_color
        threeD = self._cached_threeD

        # ── 2D → 3D 检测 ──
        dets_3d: List[Detection3D] = []
        for d2 in stable_dets_2d:
            cx, cy = d2.center
            if not (0 <= cx < W and 0 <= cy < H):
                continue
            if threeD is not None:
                Xmm, Ymm, Zmm = self._median_xyz(
                    threeD, cx, cy, self.cfg.sample_window
                )
            else:
                Xmm, Ymm, Zmm = 0.0, 0.0, 0.0
            X, Y, Z = Xmm / 1000.0, Ymm / 1000.0, Zmm / 1000.0
            if not (abs(X) < 50 and abs(Y) < 50 and abs(Z) < 50):
                X, Y, Z = 0.0, 0.0, 0.0
            dist = math.sqrt(X * X + Y * Y + Z * Z)
            dets_3d.append(
                Detection3D(
                    track_id=d2.track_id,
                    cls=d2.cls,
                    conf=d2.conf,
                    bbox=d2.bbox,
                    center=d2.center,
                    xyz=(X, Y, Z),
                    distance=dist,
                    is_interpolated=d2.is_interpolated,
                    is_release_point=d2.is_release_point,
                )
            )

        # ── 绘制 ──
        self._draw_detections(frame_left, dets_3d)
        if shot_display_info:
            self._draw_shot_overlay(frame_left, shot_display_info, frame_id)

        return (
            frame_left,
            frame_right,
            depth_color,
            FrameDetections(
                frame_id=frame_id, timestamp=ts, detections=dets_3d
            ),
        )

    # ─── 目标框绘制 ───
    def _draw_detections(self, img: np.ndarray, dets: List[Detection3D]):
        for det in dets:
            color = (
                (0, 255, 0)
                if det.cls == "basketball"
                else (0, 165, 255)
                if det.cls == "hoop"
                else (255, 255, 0)
            )
            x1, y1, x2, y2 = det.bbox
            thick = 1 if det.is_interpolated else 2
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
            label = f"{det.cls} {det.conf:.2f}"
            cv2.putText(
                img,
                label,
                (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
            )

            # Phase 5 出手帧标记
            if det.is_release_point and det.cls == "basketball":
                cx, cy = det.center
                cv2.drawMarker(
                    img, (cx, cy), (0, 0, 0), cv2.MARKER_STAR, 28, 4
                )
                cv2.drawMarker(
                    img,
                    (cx, cy),
                    (0, 215, 255),
                    cv2.MARKER_STAR,
                    24,
                    2,
                )
                cv2.circle(img, (cx, cy), 16, (0, 0, 0), 3)
                cv2.circle(img, (cx, cy), 16, (0, 215, 255), 2)

    # ─── 投篮覆盖层绘制 ───
    def _draw_shot_overlay(
        self, img: np.ndarray, info: Dict, frame_id: int
    ):
        phase = info.get("phase", "idle")
        H_img, W_img = img.shape[:2]

        # 1) 运动轨迹尾巴
        trail = info.get("trail", [])
        if len(trail) >= 2:
            recent = trail[-60:]
            n = len(recent)
            for i in range(1, n):
                alpha = i / n
                r = max(1, int(3 * alpha))
                ci = int(130 + 125 * alpha)
                color = (ci, 255, ci)
                pt1 = (int(recent[i - 1][0]), int(recent[i - 1][1]))
                pt2 = (int(recent[i][0]), int(recent[i][1]))
                cv2.line(
                    img, pt1, pt2, color, max(1, int(2 * alpha))
                )
                cv2.circle(img, pt2, r, color, -1)

        # 2) 出手弧线
        if phase in ("rising", "tracking"):
            traj_pts = info.get("shot_trajectory", [])
            if len(traj_pts) >= 2:
                for i in range(1, len(traj_pts)):
                    pt1 = (
                        int(traj_pts[i - 1][0]),
                        int(traj_pts[i - 1][1]),
                    )
                    pt2 = (int(traj_pts[i][0]), int(traj_pts[i][1]))
                    cv2.line(img, pt1, pt2, (0, 255, 255), 3)
                    cv2.circle(img, pt2, 4, (0, 255, 255), -1)

            rp = info.get("physics_release_point")
            if rp is not None:
                rpx, rpy = int(rp[0]), int(rp[1])
                cv2.circle(img, (rpx, rpy), 18, (0, 0, 0), 4)
                cv2.circle(img, (rpx, rpy), 14, (0, 215, 255), -1)
                cv2.drawMarker(
                    img,
                    (rpx, rpy),
                    (0, 0, 0),
                    cv2.MARKER_STAR,
                    20,
                    2,
                )
                txt = "RELEASE"
                tx, ty = rpx + 20, rpy + 6
                cv2.putText(
                    img,
                    txt,
                    (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 0),
                    3,
                )
                cv2.putText(
                    img,
                    txt,
                    (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 215, 255),
                    1,
                )

        # 3) 进球/未中结果
        if info.get("showing_result"):
            result_text = info["result_text"]
            result_color = info["result_color"]

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.8
            thickness = 4
            (tw, th), _ = cv2.getTextSize(
                result_text, font, font_scale, thickness
            )
            tx = (W_img - tw) // 2
            ty = 60

            overlay_img = img.copy()
            cv2.rectangle(
                overlay_img,
                (tx - 15, ty - th - 10),
                (tx + tw + 15, ty + 15),
                (0, 0, 0),
                -1,
            )
            cv2.addWeighted(overlay_img, 0.6, img, 0.4, 0, img)

            cv2.putText(
                img,
                result_text,
                (tx, ty),
                font,
                font_scale,
                (0, 0, 0),
                thickness + 2,
            )
            cv2.putText(
                img,
                result_text,
                (tx, ty),
                font,
                font_scale,
                result_color,
                thickness,
            )

            result_traj = info.get("result_trajectory", [])
            if len(result_traj) >= 2:
                scored = info.get("result_scored", False)
                traj_color = (0, 255, 0) if scored else (0, 0, 255)
                pts_arr = np.array(result_traj, dtype=np.int32)
                cv2.polylines(
                    img, [pts_arr], False, traj_color, 2, cv2.LINE_AA
                )
                cv2.circle(
                    img, tuple(pts_arr[0]), 8, (255, 255, 0), -1
                )
                apex_idx = int(np.argmin(pts_arr[:, 1]))
                cv2.circle(
                    img, tuple(pts_arr[apex_idx]), 8, (255, 0, 255), -1
                )

            rp = info.get("result_release_point")
            if rp is not None:
                rpx, rpy = int(rp[0]), int(rp[1])
                cv2.circle(img, (rpx, rpy), 18, (0, 0, 0), 4)
                cv2.circle(img, (rpx, rpy), 14, (0, 215, 255), -1)
                cv2.drawMarker(
                    img,
                    (rpx, rpy),
                    (0, 0, 0),
                    cv2.MARKER_STAR,
                    20,
                    2,
                )
                cv2.putText(
                    img,
                    "RELEASE",
                    (rpx + 20, rpy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 0),
                    3,
                )
                cv2.putText(
                    img,
                    "RELEASE",
                    (rpx + 20, rpy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 215, 255),
                    1,
                )

        # 4) 篮筐框
        hoop_bbox = info.get("hoop_bbox")
        if hoop_bbox:
            hx1, hy1, hx2, hy2 = hoop_bbox
            cv2.rectangle(
                img, (hx1, hy1), (hx2, hy2), (0, 165, 255), 2
            )

        # 5) 右上角标签 + 得分
        phase_colors = {
            "idle": (128, 128, 128),
            "rising": (0, 255, 255),
            "tracking": (0, 200, 255),
            "cooldown": (200, 200, 0),
        }
        phase_labels = {
            "idle": "IDLE",
            "rising": "RISING",
            "tracking": "TRACKING",
            "cooldown": "COOLDOWN",
        }
        pc = phase_colors.get(phase, (255, 255, 255))
        pd = phase_labels.get(phase, phase.upper())

        x_right = W_img - 180
        cv2.putText(
            img,
            pd,
            (x_right, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            3,
        )
        cv2.putText(
            img,
            pd,
            (x_right, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            pc,
            2,
        )

        scored = info.get("total_scored", 0)
        total = info.get("total_shots", 0)
        score_txt = f"Score: {scored}/{total}"
        cv2.putText(
            img,
            score_txt,
            (x_right, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            3,
        )
        cv2.putText(
            img,
            score_txt,
            (x_right, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )