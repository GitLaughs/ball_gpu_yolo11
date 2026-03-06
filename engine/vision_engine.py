# engine/vision_engine.py

from __future__ import annotations

import math
import time
from typing import List, Tuple, Dict, Optional

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

    # ─── 深度采样 ───
    def _median_xyz(self, threeD, cx, cy, win=7):
        r  = win // 2
        h, w = threeD.shape[:2]
        x1c, x2c = max(0, cx - r), min(w - 1, cx + r)
        y1c, y2c = max(0, cy - r), min(h - 1, cy + r)
        patch = threeD[y1c:y2c+1, x1c:x2c+1, :].reshape(-1, 3)
        mask  = np.isfinite(patch).all(axis=1)
        patch = patch[mask]
        if patch.size == 0:
            return 0., 0., 0.
        nonzero = np.any(np.abs(patch) > 1e-6, axis=1)
        patch   = patch[nonzero]
        if patch.size == 0:
            return 0., 0., 0.
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

        ts  = time.time()
        W, H = self.cfg.width, self.cfg.height

        frame_left  = frame[0:H,  0:W    ].copy()
        frame_right = frame[0:H,  W:W*2  ].copy()

        gray_l = cv2.cvtColor(frame_left,  cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
        rect_l = cv2.remap(gray_l, self.stereo.left_map1,
                           self.stereo.left_map2, cv2.INTER_LINEAR)
        rect_r = cv2.remap(gray_r, self.stereo.right_map1,
                           self.stereo.right_map2, cv2.INTER_LINEAR)

        disp       = self.matcher.compute(rect_l, rect_r)
        disp_norm  = cv2.normalize(disp, None, 0, 255,
                                   cv2.NORM_MINMAX, cv2.CV_8U)
        depth_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)

        threeD = cv2.reprojectImageTo3D(
            disp, self.stereo.Q, handleMissingValues=True) * 16
        threeD[np.isinf(threeD)] = 0
        threeD[np.isnan(threeD)] = 0

        # ── 2D → 3D 检测 ──
        dets_3d: List[Detection3D] = []
        for d2 in stable_dets_2d:
            cx, cy = d2.center
            if not (0 <= cx < W and 0 <= cy < H):
                continue
            Xmm, Ymm, Zmm = self._median_xyz(
                threeD, cx, cy, self.cfg.sample_window)
            X, Y, Z = Xmm / 1000., Ymm / 1000., Zmm / 1000.
            if not (abs(X) < 50 and abs(Y) < 50 and abs(Z) < 50):
                X, Y, Z = 0., 0., 0.
            dist = math.sqrt(X*X + Y*Y + Z*Z)
            dets_3d.append(Detection3D(
                track_id=d2.track_id, cls=d2.cls, conf=d2.conf,
                bbox=d2.bbox, center=d2.center, xyz=(X, Y, Z),
                distance=dist, is_interpolated=d2.is_interpolated))

        # ── 绘制 ──
        self._draw_detections(frame_left, dets_3d)
        if shot_display_info:
            self._draw_shot_overlay(frame_left, shot_display_info, frame_id)

        return frame_left, frame_right, depth_color, FrameDetections(
            frame_id=frame_id, timestamp=ts, detections=dets_3d)

    # ─── 目标框绘制 ───
    def _draw_detections(self, img, dets):
        for det in dets:
            color = (0, 255, 0) if det.cls == "basketball" else \
                    (0, 165, 255) if det.cls == "hoop" else \
                    (255, 255, 0)
            x1, y1, x2, y2 = det.bbox
            thick = 1 if det.is_interpolated else 2
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
            label = f"{det.cls} {det.conf:.2f}"
            cv2.putText(img, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # ─── 投篮覆盖层绘制 ───
    def _draw_shot_overlay(self, img, info: Dict, frame_id: int):
        phase = info.get("phase", "idle")
        H_img, W_img = img.shape[:2]

        # 1) 运动轨迹尾巴
        trail = info.get("trail", [])
        if len(trail) >= 2:
            recent = trail[-60:]
            n = len(recent)
            for i in range(1, n):
                alpha = i / n
                r     = max(1, int(3 * alpha))
                ci    = int(130 + 125 * alpha)
                color = (ci, 255, ci)
                pt1 = (int(recent[i-1][0]), int(recent[i-1][1]))
                pt2 = (int(recent[i  ][0]), int(recent[i  ][1]))
                cv2.line(img, pt1, pt2, color, max(1, int(2 * alpha)))
                cv2.circle(img, pt2, r, color, -1)

        # 2) 出手弧线 (rising / tracking 阶段)
        if phase in ("rising", "tracking"):
            traj_pts = info.get("shot_trajectory", [])
            if len(traj_pts) >= 2:
                for i in range(1, len(traj_pts)):
                    pt1 = (int(traj_pts[i-1][0]), int(traj_pts[i-1][1]))
                    pt2 = (int(traj_pts[i  ][0]), int(traj_pts[i  ][1]))
                    cv2.line(img, pt1, pt2, (0, 255, 255), 3)
                    cv2.circle(img, pt2, 4, (0, 255, 255), -1)
                sp = traj_pts[0]
                cv2.circle(img, (int(sp[0]), int(sp[1])), 8,
                           (255, 200, 0), 2)
                cv2.putText(img, "RELEASE",
                            (int(sp[0]) + 10, int(sp[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 200, 0), 1)

        # 3) 进球/未中结果
        if info.get("showing_result"):
            result_text  = info["result_text"]
            result_color = info["result_color"]

            font       = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.8
            thickness  = 4
            (tw, th), _ = cv2.getTextSize(result_text, font,
                                          font_scale, thickness)
            tx = (W_img - tw) // 2
            ty = 60

            overlay_img = img.copy()
            cv2.rectangle(overlay_img,
                          (tx - 15, ty - th - 10),
                          (tx + tw + 15, ty + 15),
                          (0, 0, 0), -1)
            cv2.addWeighted(overlay_img, 0.6, img, 0.4, 0, img)

            # 黑色描边 + 彩色文字
            cv2.putText(img, result_text, (tx, ty), font,
                        font_scale, (0, 0, 0), thickness + 2)
            cv2.putText(img, result_text, (tx, ty), font,
                        font_scale, result_color, thickness)

            # 绘制历史弹道
            result_traj = info.get("result_trajectory", [])
            if len(result_traj) >= 2:
                scored     = info.get("result_scored", False)
                traj_color = (0, 255, 0) if scored else (0, 0, 255)
                pts_arr    = np.array(result_traj, dtype=np.int32)
                cv2.polylines(img, [pts_arr], False,
                              traj_color, 2, cv2.LINE_AA)
                cv2.circle(img, tuple(pts_arr[0]), 8,
                           (255, 255, 0), -1)
                apex_idx = int(np.argmin(pts_arr[:, 1]))
                cv2.circle(img, tuple(pts_arr[apex_idx]), 8,
                           (255, 0, 255), -1)

        # 4) 篮筐框
        hoop_bbox = info.get("hoop_bbox")
        if hoop_bbox:
            hx1, hy1, hx2, hy2 = hoop_bbox
            cv2.rectangle(img, (hx1, hy1), (hx2, hy2),
                          (0, 165, 255), 2)

        # 5) 右上角阶段标签 + 得分
        phase_colors = {
            "idle":     (128, 128, 128),
            "rising":   (0,   255, 255),
            "tracking": (0,   200, 255),
            "cooldown": (200, 200,   0),
        }
        phase_labels = {
            "idle":     "IDLE",
            "rising":   "RISING",
            "tracking": "TRACKING",
            "cooldown": "COOLDOWN",
        }
        pc = phase_colors.get(phase, (255, 255, 255))
        pd = phase_labels.get(phase, phase.upper())

        x_right = W_img - 180
        cv2.putText(img, pd, (x_right, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(img, pd, (x_right, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, pc, 2)

        scored     = info.get("total_scored", 0)
        total      = info.get("total_shots",  0)
        score_txt  = f"Score: {scored}/{total}"
        cv2.putText(img, score_txt, (x_right, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(img, score_txt, (x_right, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)