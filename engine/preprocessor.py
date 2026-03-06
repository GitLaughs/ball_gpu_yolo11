### engine/preprocessor.py
"""
离线视频预处理器：
  Phase 1: 视频 → 逐帧拆图 → YOLO逐张检测 → 原始检测结果
  Phase 2: 跨帧IoU关联 → 构建轨迹
  Phase 3: 中位数滤波 → 滑动窗口平滑 → 线性插值补缺帧
  Phase 4: 导出为 {frame_id: [StableDetection2D, ...]}
"""
from __future__ import annotations
import math
import os
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable

import cv2
import numpy as np

from engine.config import EngineConfig
from engine.models import RawDetection2D, StableDetection2D

CLASS_NAMES = {0: "hoop", 1: "basketball"}


# ====================================================================
# 数据结构
# ====================================================================

@dataclass
class TrackPoint:
    """轨迹中的一个点"""
    frame_id: int
    bbox: np.ndarray          # float64 [x1,y1,x2,y2]
    center: np.ndarray        # float64 [cx,cy]
    conf: float
    is_interpolated: bool = False


@dataclass
class Track:
    """一条完整轨迹"""
    track_id: int
    cls: str
    points: Dict[int, TrackPoint] = field(default_factory=dict)  # frame_id -> TrackPoint
    last_frame: int = 0
    last_bbox: Optional[np.ndarray] = None
    last_center: Optional[np.ndarray] = None
    lost_count: int = 0


# ====================================================================
# 工具函数
# ====================================================================

def _box_iou(a, b) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / (area_a + area_b - inter + 1e-6)


def _center_dist(a, b) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    """一维滑动平均，边界用对称填充"""
    if len(arr) <= 1:
        return arr.copy()
    pad = window // 2
    padded = np.pad(arr, pad, mode='reflect')
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode='valid')[:len(arr)]


def _median_filter_1d(arr: np.ndarray, window: int) -> np.ndarray:
    """一维中位数滤波"""
    if len(arr) <= 1:
        return arr.copy()
    pad = window // 2
    out = arr.copy()
    for i in range(len(arr)):
        lo = max(0, i - pad)
        hi = min(len(arr), i + pad + 1)
        out[i] = np.median(arr[lo:hi])
    return out


# ====================================================================
# 主类
# ====================================================================

class VideoPreprocessor:
    """离线视频预处理器"""

    def __init__(self, model, cfg: EngineConfig, device: str):
        self.model = model
        self.cfg = cfg
        self.device = device
        self._next_track_id = 0

    def _new_track_id(self) -> int:
        self._next_track_id += 1
        return self._next_track_id

    # ================================================================
    # Phase 1: 逐帧YOLO检测
    # ================================================================

    def _detect_single_frame(self, frame: np.ndarray) -> List[RawDetection2D]:
        """
        把一帧当作独立图片来检测
        关键：先编码成JPG再解码，模拟"从文件读取图片"的效果
        """
        detect_img = frame

        if self.cfg.detect_save_jpg:
            # 编码→解码：去除原始噪声，模拟图片文件输入
            ok, buf = cv2.imencode('.jpg', frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, self.cfg.detect_jpg_quality])
            if ok:
                detect_img = cv2.imdecode(buf, cv2.IMREAD_COLOR)

        # 单张图片检测（不是流式）
        results = self.model.predict(
            detect_img,
            device=self.device,
            verbose=False,
            conf=self.cfg.conf_threshold,
            iou=self.cfg.iou_threshold,
            max_det=self.cfg.max_det,
            augment=False,       # 可设True做TTA增强检测率
        )

        dets = []
        if not results or len(results) == 0:
            return dets

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return dets

        for i in range(len(boxes)):
            box = boxes.xyxy[i].cpu().numpy()
            conf = float(boxes.conf[i].cpu().numpy())
            cls_id = int(boxes.cls[i].cpu().numpy())

            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cls_name = CLASS_NAMES.get(cls_id, str(cls_id))

            dets.append(RawDetection2D(
                cls=cls_name, conf=conf,
                bbox=(x1, y1, x2, y2), center=(cx, cy),
            ))

        return dets

    def _phase1_detect_all(
        self, video_path: str,
        progress_cb: Optional[Callable[[int, int, str], None]] = None
    ) -> Tuple[Dict[int, List[RawDetection2D]], int, float]:
        """
        第一遍：读取视频每一帧，逐帧YOLO检测
        返回：{frame_id: [RawDetection2D, ...]}, total_frames, fps
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        raw: Dict[int, List[RawDetection2D]] = {}
        frame_id = 0
        t0 = time.time()

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_id += 1

            # 只取左半图检测（双目拼接帧）
            left = frame[0:self.cfg.height, 0:self.cfg.width]
            dets = self._detect_single_frame(left)
            raw[frame_id] = dets

            if progress_cb and frame_id % 10 == 0:
                elapsed = time.time() - t0
                eta = (elapsed / frame_id) * (total - frame_id) if frame_id > 0 else 0
                progress_cb(frame_id, total,
                            f"检测中 {frame_id}/{total} | "
                            f"检出 {len(dets)} 个目标 | "
                            f"ETA {eta:.0f}s")

        cap.release()

        if progress_cb:
            progress_cb(total, total, f"检测完成，共 {frame_id} 帧")

        print(f"[Preprocessor] Phase1 完成: {frame_id} 帧, "
              f"耗时 {time.time() - t0:.1f}s")

        return raw, frame_id, fps

    # ================================================================
    # Phase 2: 跨帧跟踪关联
    # ================================================================

    def _phase2_build_tracks(
        self, raw: Dict[int, List[RawDetection2D]], total_frames: int
    ) -> List[Track]:
        """
        按时间顺序遍历每帧检测，用IoU+距离做贪心匹配，构建轨迹
        """
        active_tracks: List[Track] = []
        finished_tracks: List[Track] = []

        for fid in range(1, total_frames + 1):
            dets = raw.get(fid, [])

            # 标记是否已匹配
            matched_track = set()
            matched_det = set()

            # ---- 贪心匹配：对每个 (track, det) 对算兼容分数 ----
            if active_tracks and dets:
                scores = []
                for ti, tk in enumerate(active_tracks):
                    for di, d in enumerate(dets):
                        if tk.cls != d.cls:
                            continue
                        iou = _box_iou(
                            tuple(int(v) for v in tk.last_bbox), d.bbox
                        )
                        cdist = _center_dist(tk.last_center, d.center)
                        if cdist > self.cfg.track_max_center_dist:
                            continue
                        if iou < self.cfg.track_iou_thresh * 0.5:
                            # IoU太低但距离近也给个小分（补偿遮挡后重现）
                            if cdist < self.cfg.track_max_center_dist * 0.5:
                                score = 0.01
                            else:
                                continue
                        else:
                            score = iou
                        scores.append((score, ti, di))

                # 按分数从高到低贪心匹配
                scores.sort(key=lambda x: -x[0])
                for score, ti, di in scores:
                    if ti in matched_track or di in matched_det:
                        continue
                    tk = active_tracks[ti]
                    d = dets[di]
                    tk.points[fid] = TrackPoint(
                        frame_id=fid,
                        bbox=np.array(d.bbox, dtype=float),
                        center=np.array(d.center, dtype=float),
                        conf=d.conf,
                    )
                    tk.last_bbox = np.array(d.bbox, dtype=float)
                    tk.last_center = np.array(d.center, dtype=float)
                    tk.last_frame = fid
                    tk.lost_count = 0
                    matched_track.add(ti)
                    matched_det.add(di)

            # ---- 未匹配的检测 → 新建轨迹 ----
            for di, d in enumerate(dets):
                if di not in matched_det:
                    tk = Track(
                        track_id=self._new_track_id(),
                        cls=d.cls,
                        last_frame=fid,
                        last_bbox=np.array(d.bbox, dtype=float),
                        last_center=np.array(d.center, dtype=float),
                    )
                    tk.points[fid] = TrackPoint(
                        frame_id=fid,
                        bbox=np.array(d.bbox, dtype=float),
                        center=np.array(d.center, dtype=float),
                        conf=d.conf,
                    )
                    active_tracks.append(tk)

            # ---- 更新丢失计数，关闭过期轨迹 ----
            still_active = []
            for ti, tk in enumerate(active_tracks):
                if ti not in matched_track:
                    tk.lost_count += 1
                if tk.lost_count > self.cfg.track_max_gap:
                    finished_tracks.append(tk)
                else:
                    still_active.append(tk)
            active_tracks = still_active

        # 把仍存活的也加入
        finished_tracks.extend(active_tracks)

        # 过滤掉太短的轨迹（闪烁噪声）
        valid = [t for t in finished_tracks if len(t.points) >= self.cfg.track_min_len]

        print(f"[Preprocessor] Phase2 完成: {len(finished_tracks)} 条轨迹, "
              f"过滤后保留 {len(valid)} 条")

        return valid

    # ================================================================
    # Phase 3: 平滑 + 插值
    # ================================================================

    def _smooth_single_track(self, track: Track) -> Track:
        """对单条轨迹做：中位数去噪 → 滑动平均平滑 → 缺失帧插值"""
        if len(track.points) < 2:
            return track

        # 获取所有 frame_id 排序
        fids = sorted(track.points.keys())
        n = len(fids)

        # 提取坐标数组
        x1s = np.array([track.points[f].bbox[0] for f in fids])
        y1s = np.array([track.points[f].bbox[1] for f in fids])
        x2s = np.array([track.points[f].bbox[2] for f in fids])
        y2s = np.array([track.points[f].bbox[3] for f in fids])
        confs = np.array([track.points[f].conf for f in fids])

        # ---- Step 1: 中位数滤波（去除离群点）----
        med_win = self.cfg.median_filter_size
        x1s = _median_filter_1d(x1s, med_win)
        y1s = _median_filter_1d(y1s, med_win)
        x2s = _median_filter_1d(x2s, med_win)
        y2s = _median_filter_1d(y2s, med_win)

        # ---- Step 2: 尺寸平滑（宽高独立平滑，保持大小稳定）----
        ws = x2s - x1s
        hs = y2s - y1s
        cxs = (x1s + x2s) / 2.0
        cys = (y1s + y2s) / 2.0

        size_win = min(self.cfg.size_smooth_window, n)
        ws = _moving_average(ws, size_win)
        hs = _moving_average(hs, size_win)

        # ---- Step 3: 中心点平滑 ----
        sm_win = min(self.cfg.smooth_window, n)
        cxs = _moving_average(cxs, sm_win)
        cys = _moving_average(cys, sm_win)

        # ---- 重建bbox ----
        x1s = cxs - ws / 2.0
        y1s = cys - hs / 2.0
        x2s = cxs + ws / 2.0
        y2s = cys + hs / 2.0

        # 写回
        for i, fid in enumerate(fids):
            track.points[fid].bbox = np.array([x1s[i], y1s[i], x2s[i], y2s[i]])
            track.points[fid].center = np.array([cxs[i], cys[i]])

        # ---- Step 4: 缺失帧线性插值 ----
        first_fid = fids[0]
        last_fid = fids[-1]

        # 找出所有缺失的帧段
        all_fids_set = set(fids)
        gaps = []
        i = 0
        while i < n - 1:
            gap_start = fids[i]
            gap_end = fids[i + 1]
            gap_len = gap_end - gap_start - 1
            if 0 < gap_len <= self.cfg.interpolate_max_gap:
                gaps.append((gap_start, gap_end, gap_len))
            i += 1

        for gap_start, gap_end, gap_len in gaps:
            p0 = track.points[gap_start]
            p1 = track.points[gap_end]
            for k in range(1, gap_len + 1):
                alpha = k / (gap_len + 1)
                fid = gap_start + k
                bbox = (1 - alpha) * p0.bbox + alpha * p1.bbox
                center = (1 - alpha) * p0.center + alpha * p1.center
                conf = (1 - alpha) * p0.conf + alpha * p1.conf
                track.points[fid] = TrackPoint(
                    frame_id=fid,
                    bbox=bbox,
                    center=center,
                    conf=conf * 0.8,     # 插值点给低一点置信度
                    is_interpolated=True,
                )

        return track

    def _phase3_smooth(self, tracks: List[Track]) -> List[Track]:
        """对所有轨迹做平滑+插值"""
        smoothed = []
        for tk in tracks:
            smoothed.append(self._smooth_single_track(tk))
        print(f"[Preprocessor] Phase3 完成: {len(smoothed)} 条轨迹已平滑+插值")
        return smoothed

    # ================================================================
    # Phase 4: 导出为逐帧字典
    # ================================================================

    def _phase4_export(
        self, tracks: List[Track], total_frames: int
    ) -> Dict[int, List[StableDetection2D]]:
        """
        将轨迹转回 {frame_id: [StableDetection2D, ...]} 格式
        同一帧同一类别保留置信度最高的（去重）
        """
        result: Dict[int, List[StableDetection2D]] = {}

        for fid in range(1, total_frames + 1):
            result[fid] = []

        for tk in tracks:
            for fid, pt in tk.points.items():
                if fid < 1 or fid > total_frames:
                    continue
                bbox = tuple(max(0, int(round(v))) for v in pt.bbox)
                center = tuple(int(round(v)) for v in pt.center)
                result[fid].append(StableDetection2D(
                    track_id=tk.track_id,
                    cls=tk.cls,
                    conf=pt.conf,
                    bbox=bbox,
                    center=center,
                    is_interpolated=pt.is_interpolated,
                ))

        # 同一帧同一类别去重：只保留conf最高的那个
        for fid in result:
            by_cls = {}
            for d in result[fid]:
                key = d.cls
                if key not in by_cls or d.conf > by_cls[key].conf:
                    by_cls[key] = d
            result[fid] = list(by_cls.values())

        n_total = sum(len(v) for v in result.values())
        n_interp = sum(1 for v in result.values()
                       for d in v if d.is_interpolated)
        print(f"[Preprocessor] Phase4 完成: {n_total} 个检测 "
              f"(其中 {n_interp} 个为插值补充)")

        return result

    # ================================================================
    # 公开接口
    # ================================================================

    def process(
        self, video_path: str,
        progress_cb: Optional[Callable[[int, int, str], None]] = None
    ) -> Tuple[Dict[int, List[StableDetection2D]], int, float]:
        """
        完整预处理管线
        
        返回:
            stable_dets: {frame_id: [StableDetection2D, ...]}
            total_frames: int
            fps: float
        """
        print(f"[Preprocessor] ═══ 开始预处理: {video_path} ═══")
        t_start = time.time()

        # Phase 1
        raw, total, fps = self._phase1_detect_all(video_path, progress_cb)

        # Phase 2
        tracks = self._phase2_build_tracks(raw, total)

        # Phase 3
        tracks = self._phase3_smooth(tracks)

        # Phase 4
        stable = self._phase4_export(tracks, total)

        elapsed = time.time() - t_start
        print(f"[Preprocessor] ═══ 预处理完成，总耗时 {elapsed:.1f}s ═══")

        return stable, total, fps