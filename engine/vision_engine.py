from __future__ import annotations
import time
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from engine.config import EngineConfig
from engine.models import Detection3D, FrameDetections

CLASS_NAMES = {0: "hoop", 1: "basketball"}

@dataclass
class StereoParams:
    Q: np.ndarray
    left_map1: np.ndarray
    left_map2: np.ndarray
    right_map1: np.ndarray
    right_map2: np.ndarray

class StereoMatcher:
    def __init__(self):
        blockSize = 8
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=1,
            numDisparities=128,
            blockSize=blockSize,
            P1=8 * 3 * blockSize * blockSize,
            P2=32 * 3 * blockSize * blockSize,
            disp12MaxDiff=-1,
            preFilterCap=63,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=1,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    def compute(self, left_img, right_img):
        return self.stereo.compute(left_img, right_img)

def load_yolo_model(model_path: str, device: str, cfg: EngineConfig) -> YOLO:
    model = YOLO(model_path)
    model.conf = cfg.conf_threshold
    model.iou = cfg.iou_threshold
    model.max_det = cfg.max_det
    if device.startswith("cuda") and torch.cuda.is_available():
        model.to(device)
    return model

class VisionEngine:
    def __init__(self, model: YOLO, stereo: StereoParams, cfg: EngineConfig, device: str):
        self.model = model
        self.stereo = stereo
        self.cfg = cfg
        self.device = device
        self.matcher = StereoMatcher()

    def _median_xyz(self, threeD: np.ndarray, cx: int, cy: int, win: int) -> Tuple[float, float, float]:
        """在 bbox center 周围取 win*win 的中位数（比单点更稳）"""
        r = win // 2
        x1, x2 = max(0, cx - r), min(self.cfg.width - 1, cx + r)
        y1, y2 = max(0, cy - r), min(self.cfg.height - 1, cy + r)
        patch = threeD[y1:y2+1, x1:x2+1, :]  # (h,w,3)

        patch = patch.reshape(-1, 3)
        # 过滤 inf/nan/0
        mask = np.isfinite(patch).all(axis=1)
        patch = patch[mask]
        if patch.size == 0:
            return 0.0, 0.0, 0.0

        # 把全0点去掉
        nonzero = np.any(np.abs(patch) > 1e-6, axis=1)
        patch = patch[nonzero]
        if patch.size == 0:
            return 0.0, 0.0, 0.0

        med = np.median(patch, axis=0)
        return float(med[0]), float(med[1]), float(med[2])

    def process_frame(self, frame: np.ndarray, frame_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, FrameDetections]:
        """
        输入：原始拼接帧 (左640x480 | 右640x480)
        输出：frame_left_draw, frame_right_draw, depth_color, FrameDetections
        """
        ts = time.time()

        frame_left = frame[0:self.cfg.height, 0:self.cfg.width].copy()
        frame_right = frame[0:self.cfg.height, self.cfg.width:self.cfg.width*2].copy()

        gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

        rect_left = cv2.remap(gray_left, self.stereo.left_map1, self.stereo.left_map2, cv2.INTER_LINEAR)
        rect_right = cv2.remap(gray_right, self.stereo.right_map1, self.stereo.right_map2, cv2.INTER_LINEAR)

        disparity = self.matcher.compute(rect_left, rect_right)
        disp_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_color = cv2.applyColorMap(disp_normalized, cv2.COLORMAP_JET)

        threeD = cv2.reprojectImageTo3D(disparity, self.stereo.Q, handleMissingValues=True)
        threeD = threeD * 16  # 你原来就这么做的 :contentReference[oaicite:3]{index=3}

        threeD[np.isinf(threeD)] = 0
        threeD[np.isnan(threeD)] = 0

        # YOLO
        results = self.model.predict(frame_left, device=self.device, verbose=False, conf=self.cfg.conf_threshold)

        dets: List[Detection3D] = []
        if hasattr(results, "__iter__") and len(results) > 0 and hasattr(results[0], "boxes"):
            boxes = results[0].boxes
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())

                x1, y1, x2, y2 = map(int, box)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                if not (0 <= cx < self.cfg.width and 0 <= cy < self.cfg.height):
                    continue

                Xmm, Ymm, Zmm = self._median_xyz(threeD, cx, cy, self.cfg.sample_window)
                X = Xmm / 1000.0
                Y = Ymm / 1000.0
                Z = Zmm / 1000.0

                # 过滤明显异常（你原来也在做合理性判断 :contentReference[oaicite:4]{index=4}，这里只保留简化版本）
                if not (abs(X) < 100 and abs(Y) < 100 and abs(Z) < 100):
                    continue

                dist = math.sqrt(X*X + Y*Y + Z*Z)
                cls_name = CLASS_NAMES.get(cls_id, str(cls_id))

                dets.append(
                    Detection3D(
                        cls=cls_name,
                        conf=conf,
                        bbox=(x1, y1, x2, y2),
                        center=(cx, cy),
                        xyz=(X, Y, Z),
                        distance=dist,
                    )
                )

                # draw
                color = (0, 255, 0) if cls_name == "basketball" else (0, 0, 255)
                label = f"{cls_name} {conf:.2f}"
                for f in (frame_left, frame_right):
                    cv2.rectangle(f, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(f, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.putText(f, f"X:{X:.2f} Y:{Y:.2f} Z:{Z:.2f}m", (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        return frame_left, frame_right, depth_color, FrameDetections(frame_id=frame_id, timestamp=ts, detections=dets)