### engine/preprocessor.py
"""
离线视频预处理器（性能优化版 + 出手点 40% 约束）：
  Phase 1: 视频 → 隔帧 YOLO 批量推理 → 原始检测结果
  Phase 2: 跨帧 IoU 关联 → 构建轨迹
  Phase 3: 向量化中位数滤波 → 滑动窗口平滑 → 线性插值补齐漏帧
  Phase 4: 导出为 {frame_id: [StableDetection2D, ...]}
  Phase 5: 批量矩阵乘法物理分析 → 像素重力估算 + 自由飞行段检测 + 出手帧确定

优化点：
  ● Phase 1: YOLO 批量推理
  ● Phase 3: numpy 向量化中值滤波
  ● Phase 5: 预计算 Vandermonde 伪逆，矩阵乘法批量拟合 + R² 计算
  ● Phase 5: 滑动窗口步进采样 (step ≈ win//3)
  ● Phase 5: 向量化篮筐区域过滤
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from engine.config import EngineConfig
from engine.models import PhysicsMetadata, RawDetection2D, StableDetection2D

CLASS_NAMES = {0: "hoop", 1: "basketball"}


# ====================================================================
# 内部数据结构
# ====================================================================

@dataclass
class TrackPoint:
    frame_id: int
    bbox: np.ndarray
    center: np.ndarray
    conf: float
    is_interpolated: bool = False


@dataclass
class Track:
    track_id: int
    cls: str
    points: Dict[int, TrackPoint] = field(default_factory=dict)
    last_frame: int = 0
    last_bbox: Optional[np.ndarray] = None
    last_center: Optional[np.ndarray] = None
    lost_count: int = 0


# ====================================================================
# 工具函数
# ====================================================================

def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / (area_a + area_b - inter + 1e-6)


def _center_dist(a: np.ndarray, b: np.ndarray) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy)


def _moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    if len(arr) <= 1 or window <= 1:
        return arr.copy()
    pad = window // 2
    padded = np.pad(arr, pad, mode="reflect")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")[: len(arr)]


def _median_filter_1d(arr: np.ndarray, window: int) -> np.ndarray:
    n = len(arr)
    if n <= 1 or window <= 1:
        return arr.copy()
    pad = window // 2
    padded = np.pad(arr, pad, mode="reflect")
    windows = sliding_window_view(padded, window)
    return np.median(windows, axis=1)[:n].copy()


def _precompute_polyfit2(
    win_size: int, dt: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    t = np.arange(win_size, dtype=np.float64) * dt
    V = np.column_stack([t * t, t, np.ones(win_size)])
    pinv_V = np.linalg.pinv(V)
    return pinv_V, V


# ====================================================================
# 主类
# ====================================================================

class VideoPreprocessor:
    def __init__(self, model, cfg: EngineConfig, device: str):
        self.model = model
        self.cfg = cfg
        self.device = device
        self._next_track_id = 0

    def _new_track_id(self) -> int:
        self._next_track_id += 1
        return self._next_track_id

    # ================================================================
    # Phase 1: 批量 YOLO 检测
    # ================================================================

    def _parse_yolo_result(self, result) -> List[RawDetection2D]:
        dets: List[RawDetection2D] = []
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return dets
        xyxy_all = boxes.xyxy.cpu().numpy()
        conf_all = boxes.conf.cpu().numpy()
        cls_all = boxes.cls.cpu().numpy().astype(int)
        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, xyxy_all[i])
            dets.append(
                RawDetection2D(
                    cls=CLASS_NAMES.get(int(cls_all[i]), str(cls_all[i])),
                    conf=float(conf_all[i]),
                    bbox=(x1, y1, x2, y2),
                    center=((x1 + x2) // 2, (y1 + y2) // 2),
                )
            )
        return dets

    def _phase1_detect_all(
        self,
        video_path: str,
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
    ) -> Tuple[Dict[int, List[RawDetection2D]], int, float]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        raw: Dict[int, List[RawDetection2D]] = {}
        frame_id = 0
        t0 = time.time()
        H, W = self.cfg.height, self.cfg.width
        batch_size = self.cfg.detect_batch_size

        batch_imgs: List[np.ndarray] = []
        batch_ids: List[int] = []

        def _flush_batch():
            if not batch_imgs:
                return
            results = self.model.predict(
                batch_imgs,
                device=self.device,
                verbose=False,
                conf=self.cfg.conf_threshold,
                iou=self.cfg.iou_threshold,
                max_det=self.cfg.max_det,
                augment=False,
            )
            for i, res in enumerate(results):
                raw[batch_ids[i]] = self._parse_yolo_result(res)
            batch_imgs.clear()
            batch_ids.clear()

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_id += 1

            if frame_id % 2 != 0:
                left = frame[0:H, 0:W]
                detect_img = left
                if self.cfg.detect_save_jpg:
                    ok_enc, buf = cv2.imencode(
                        ".jpg", left,
                        [cv2.IMWRITE_JPEG_QUALITY, self.cfg.detect_jpg_quality],
                    )
                    if ok_enc:
                        detect_img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                batch_imgs.append(detect_img)
                batch_ids.append(frame_id)
            else:
                raw[frame_id] = []

            if len(batch_imgs) >= batch_size:
                _flush_batch()

            if progress_cb and frame_id % 20 == 0:
                elapsed = time.time() - t0
                eta = (
                    (elapsed / frame_id) * (total - frame_id)
                    if frame_id > 0
                    else 0
                )
                progress_cb(
                    frame_id,
                    total,
                    f"检测中 {frame_id}/{total} | 批量×{batch_size} | ETA {eta:.0f}s",
                )

        _flush_batch()
        cap.release()

        if progress_cb:
            progress_cb(total, total, f"检测完成，共 {frame_id} 帧")
        print(
            f"[Preprocessor] Phase1 完成: {frame_id} 帧, "
            f"耗时 {time.time()-t0:.1f}s"
        )
        return raw, frame_id, fps

    # ================================================================
    # Phase 2: 跨帧跟踪关联
    # ================================================================

    def _phase2_build_tracks(
        self, raw: Dict[int, List[RawDetection2D]], total_frames: int
    ) -> List[Track]:
        active_tracks: List[Track] = []
        finished_tracks: List[Track] = []

        for fid in range(1, total_frames + 1):
            dets = raw.get(fid, [])
            matched_track: set = set()
            matched_det: set = set()

            if active_tracks and dets:
                det_bboxes = [np.array(d.bbox, dtype=float) for d in dets]
                det_centers = [np.array(d.center, dtype=float) for d in dets]

                scores = []
                for ti, tk in enumerate(active_tracks):
                    for di, d in enumerate(dets):
                        if tk.cls != d.cls:
                            continue
                        iou = _box_iou(tk.last_bbox, det_bboxes[di])
                        dist = _center_dist(tk.last_center, det_centers[di])
                        if (
                            iou >= self.cfg.track_iou_thresh
                            or dist <= self.cfg.track_max_center_dist
                        ):
                            scores.append((iou * 10 - dist * 0.01, ti, di))

                scores.sort(key=lambda x: -x[0])
                for score, ti, di in scores:
                    if ti in matched_track or di in matched_det:
                        continue
                    tk = active_tracks[ti]
                    d = dets[di]
                    bb = det_bboxes[di]
                    cc = det_centers[di]
                    tk.points[fid] = TrackPoint(
                        frame_id=fid, bbox=bb, center=cc, conf=d.conf,
                    )
                    tk.last_bbox = bb
                    tk.last_center = cc
                    tk.last_frame = fid
                    tk.lost_count = 0
                    matched_track.add(ti)
                    matched_det.add(di)

            for di, d in enumerate(dets):
                if di not in matched_det:
                    bb = np.array(d.bbox, dtype=float)
                    cc = np.array(d.center, dtype=float)
                    tk = Track(
                        track_id=self._new_track_id(),
                        cls=d.cls,
                        last_frame=fid,
                        last_bbox=bb,
                        last_center=cc,
                    )
                    tk.points[fid] = TrackPoint(
                        frame_id=fid, bbox=bb, center=cc, conf=d.conf,
                    )
                    active_tracks.append(tk)

            still_active = []
            for ti, tk in enumerate(active_tracks):
                if ti not in matched_track:
                    tk.lost_count += 1
                if tk.lost_count > self.cfg.track_max_gap:
                    finished_tracks.append(tk)
                else:
                    still_active.append(tk)
            active_tracks = still_active

        finished_tracks.extend(active_tracks)
        valid = [
            t for t in finished_tracks if len(t.points) >= self.cfg.track_min_len
        ]
        print(
            f"[Preprocessor] Phase2 完成: {len(finished_tracks)} 条轨迹, "
            f"过滤后保留 {len(valid)} 条"
        )
        return valid

    # ================================================================
    # Phase 3: 平滑 + 插值
    # ================================================================

    def _smooth_single_track(self, track: Track) -> Track:
        if len(track.points) < 2:
            return track

        fids = sorted(track.points.keys())
        n = len(fids)

        x1s = np.array([track.points[f].bbox[0] for f in fids])
        y1s = np.array([track.points[f].bbox[1] for f in fids])
        x2s = np.array([track.points[f].bbox[2] for f in fids])
        y2s = np.array([track.points[f].bbox[3] for f in fids])

        med_win = self.cfg.median_filter_size
        x1s = _median_filter_1d(x1s, med_win)
        y1s = _median_filter_1d(y1s, med_win)
        x2s = _median_filter_1d(x2s, med_win)
        y2s = _median_filter_1d(y2s, med_win)

        ws = x2s - x1s
        hs = y2s - y1s
        cxs = (x1s + x2s) / 2.0
        cys = (y1s + y2s) / 2.0

        size_win = min(self.cfg.size_smooth_window, n)
        ws = _moving_average(ws, size_win)
        hs = _moving_average(hs, size_win)

        sm_win = min(self.cfg.smooth_window, n)
        cxs = _moving_average(cxs, sm_win)
        cys = _moving_average(cys, sm_win)

        x1s = cxs - ws / 2.0
        y1s = cys - hs / 2.0
        x2s = cxs + ws / 2.0
        y2s = cys + hs / 2.0

        for i, fid in enumerate(fids):
            track.points[fid].bbox = np.array(
                [x1s[i], y1s[i], x2s[i], y2s[i]]
            )
            track.points[fid].center = np.array([cxs[i], cys[i]])

        i = 0
        while i < n - 1:
            gap_start = fids[i]
            gap_end = fids[i + 1]
            gap_len = gap_end - gap_start - 1
            if 0 < gap_len <= self.cfg.interpolate_max_gap:
                p0 = track.points[gap_start]
                p1 = track.points[gap_end]
                for k in range(1, gap_len + 1):
                    alpha = k / (gap_len + 1)
                    fid_k = gap_start + k
                    track.points[fid_k] = TrackPoint(
                        frame_id=fid_k,
                        bbox=(1 - alpha) * p0.bbox + alpha * p1.bbox,
                        center=(1 - alpha) * p0.center + alpha * p1.center,
                        conf=((1 - alpha) * p0.conf + alpha * p1.conf) * 0.8,
                        is_interpolated=True,
                    )
            i += 1

        return track

    def _phase3_smooth(self, tracks: List[Track]) -> List[Track]:
        smoothed = [self._smooth_single_track(tk) for tk in tracks]
        print(
            f"[Preprocessor] Phase3 完成: {len(smoothed)} 条轨迹已平滑+插值"
        )
        return smoothed

    # ================================================================
    # Phase 4: 导出
    # ================================================================

    def _phase4_export(
        self,
        tracks: List[Track],
        total_frames: int,
        release_frames: Optional[Dict[int, Optional[int]]] = None,
    ) -> Dict[int, List[StableDetection2D]]:
        release_frames = release_frames or {}
        result: Dict[int, List[StableDetection2D]] = {
            fid: [] for fid in range(1, total_frames + 1)
        }

        for tk in tracks:
            release_fid = release_frames.get(tk.track_id)
            for fid, pt in tk.points.items():
                if fid < 1 or fid > total_frames:
                    continue
                bbox = tuple(max(0, int(round(v))) for v in pt.bbox)
                center = tuple(int(round(v)) for v in pt.center)
                result[fid].append(
                    StableDetection2D(
                        track_id=tk.track_id,
                        cls=tk.cls,
                        conf=pt.conf,
                        bbox=bbox,
                        center=center,
                        is_interpolated=pt.is_interpolated,
                        is_release_point=(
                            release_fid is not None and fid == release_fid
                        ),
                    )
                )

        for fid in result:
            by_cls: Dict[str, StableDetection2D] = {}
            for d in result[fid]:
                key = d.cls
                if key not in by_cls:
                    by_cls[key] = d
                else:
                    existing = by_cls[key]
                    if d.is_release_point and not existing.is_release_point:
                        by_cls[key] = d
                    elif (
                        not d.is_release_point and existing.is_release_point
                    ):
                        pass
                    elif d.conf > existing.conf:
                        by_cls[key] = d
            result[fid] = list(by_cls.values())

        n_total = sum(len(v) for v in result.values())
        n_interp = sum(
            1 for v in result.values() for d in v if d.is_interpolated
        )
        n_release = sum(
            1 for v in result.values() for d in v if d.is_release_point
        )
        print(
            f"[Preprocessor] Phase4 完成: {n_total} 个检测 "
            f"(插值: {n_interp}, 出手帧: {n_release})"
        )
        return result

    # ================================================================
    # Phase 5: 物理分析
    # ================================================================

    def _estimate_pixel_gravity(
        self, track: Track, fps: float
    ) -> Optional[float]:
        fids = sorted(track.points.keys())
        n = len(fids)
        if n < 10:
            return None

        cys = np.array(
            [track.points[f].center[1] for f in fids], dtype=np.float64
        )
        frame_arr = np.array(fids, dtype=np.float64)

        win = max(8, int(fps * self.cfg.gravity_fit_window_sec))
        win = min(win, n)
        if n < win:
            return None

        num_windows = n - win + 1
        step = max(1, win // 3)

        diffs = np.diff(frame_arr)
        is_regular = diffs.size > 0 and np.all(diffs == diffs[0])

        if is_regular and num_windows > 0:
            dt = diffs[0]
            pinv_V, V = _precompute_polyfit2(win, dt)

            all_y = sliding_window_view(cys, win)
            sampled_idx = np.arange(0, num_windows, step)
            y_win = all_y[sampled_idx]

            coeffs = pinv_V @ y_win.T
            g_all = 2.0 * coeffs[0]

            y_fit = (V @ coeffs).T
            residuals = y_win - y_fit
            ss_res = np.sum(residuals * residuals, axis=1)
            y_mean = np.mean(y_win, axis=1, keepdims=True)
            ss_tot = np.sum((y_win - y_mean) ** 2, axis=1)
            r2 = 1.0 - ss_res / (ss_tot + 1e-9)

            mask = (
                (r2 >= self.cfg.gravity_fit_r2_min)
                & (g_all >= self.cfg.gravity_min_px_per_frame2)
                & (g_all <= self.cfg.gravity_max_px_per_frame2)
            )
            g_candidates = g_all[mask]
        else:
            g_list: List[float] = []
            for i in range(0, num_windows, step):
                t = frame_arr[i : i + win] - frame_arr[i]
                y = cys[i : i + win]
                try:
                    c = np.polyfit(t, y, 2)
                except (np.linalg.LinAlgError, ValueError):
                    continue
                g_px = 2.0 * c[0]
                y_fit = np.polyval(c, t)
                ss_res = np.sum((y - y_fit) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2 = 1.0 - ss_res / (ss_tot + 1e-9)
                if (
                    r2 >= self.cfg.gravity_fit_r2_min
                    and self.cfg.gravity_min_px_per_frame2
                    <= g_px
                    <= self.cfg.gravity_max_px_per_frame2
                ):
                    g_list.append(g_px)
            g_candidates = np.array(g_list) if g_list else np.array([])

        if g_candidates.size == 0:
            return None

        g_median = float(np.median(g_candidates))
        if g_candidates.size > 1:
            g_std = float(np.std(g_candidates))
            filtered = g_candidates[
                np.abs(g_candidates - g_median) <= 2.0 * g_std + 1e-6
            ]
            if filtered.size > 0:
                g_median = float(np.median(filtered))
        return g_median

    def _find_free_flight_segments(
        self,
        track: Track,
        gravity_px: float,
        fps: float,
        hoop_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> List[Tuple[int, int]]:
        fids = sorted(track.points.keys())
        n = len(fids)
        if n < 5:
            return []

        frame_arr = np.array(fids, dtype=np.float64)
        cxs = np.array(
            [track.points[f].center[0] for f in fids], dtype=np.float64
        )
        cys = np.array(
            [track.points[f].center[1] for f in fids], dtype=np.float64
        )

        ax_threshold = gravity_px * self.cfg.horiz_accel_ratio
        win = max(4, int(fps * self.cfg.free_flight_window_sec))
        win = min(win, n)
        if n < win:
            return []

        num_windows = n - win + 1
        step = max(1, win // 3)
        sampled_starts = np.arange(0, num_windows, step)
        n_sampled = len(sampled_starts)

        free_mask = np.zeros(n, dtype=bool)

        diffs = np.diff(frame_arr)
        is_regular = diffs.size > 0 and np.all(diffs == diffs[0])

        if is_regular and n_sampled > 0:
            dt = diffs[0]
            pinv_V, V = _precompute_polyfit2(win, dt)

            all_x = sliding_window_view(cxs, win)
            all_y = sliding_window_view(cys, win)

            x_win = all_x[sampled_starts]
            y_win = all_y[sampled_starts]

            layer1_mask = np.ones(n_sampled, dtype=bool)
            if hoop_bbox is not None:
                hx1, hy1, hx2, hy2 = hoop_bbox
                hw = max(hx2 - hx1, 1)
                hh = max(hy2 - hy1, 1)
                cx_m = np.mean(x_win, axis=1)
                cy_m = np.mean(y_win, axis=1)
                mx = hw * self.cfg.release_hoop_margin_x
                myu = hh * self.cfg.release_hoop_margin_y_up
                myd = hh * self.cfg.release_hoop_margin_y_dn
                near = (
                    (cx_m > hx1 - mx)
                    & (cx_m < hx2 + mx)
                    & (cy_m > hy1 - myu)
                    & (cy_m < hy2 + myd)
                )
                layer1_mask &= ~near

            l1_idx = np.where(layer1_mask)[0]
            if l1_idx.size == 0:
                return []

            x_l1 = x_win[l1_idx]
            coeffs_x = pinv_V @ x_l1.T
            ax_abs = np.abs(2.0 * coeffs_x[0])
            horiz_ok = ax_abs <= ax_threshold

            l2_idx = l1_idx[horiz_ok]
            if l2_idx.size == 0:
                return []

            y_l2 = y_win[l2_idx]
            coeffs_y = pinv_V @ y_l2.T
            y_fit = (V @ coeffs_y).T
            residuals = y_l2 - y_fit
            ss_res = np.sum(residuals * residuals, axis=1)
            y_mean = np.mean(y_l2, axis=1, keepdims=True)
            ss_tot = np.sum((y_l2 - y_mean) ** 2, axis=1)
            r2 = 1.0 - ss_res / (ss_tot + 1e-9)
            r2_ok = r2 >= self.cfg.free_flight_r2_min

            passing_sampled = l2_idx[r2_ok]
            for si in passing_sampled:
                orig_start = sampled_starts[si]
                free_mask[orig_start : orig_start + win] = True

        else:
            for i in range(0, num_windows, step):
                t = frame_arr[i : i + win] - frame_arr[i]
                x = cxs[i : i + win]
                y = cys[i : i + win]
                if len(t) < 3:
                    continue
                if hoop_bbox is not None:
                    hx1, hy1, hx2, hy2 = hoop_bbox
                    hw = max(hx2 - hx1, 1)
                    hh = max(hy2 - hy1, 1)
                    cx_mid, cy_mid = np.mean(x), np.mean(y)
                    mx = hw * self.cfg.release_hoop_margin_x
                    myu = hh * self.cfg.release_hoop_margin_y_up
                    myd = hh * self.cfg.release_hoop_margin_y_dn
                    if (hx1 - mx < cx_mid < hx2 + mx) and (
                        hy1 - myu < cy_mid < hy2 + myd
                    ):
                        continue
                try:
                    cx_coeffs = np.polyfit(t, x, 2)
                except (np.linalg.LinAlgError, ValueError):
                    continue
                if abs(2.0 * cx_coeffs[0]) > ax_threshold:
                    continue
                try:
                    cy_coeffs = np.polyfit(t, y, 2)
                except (np.linalg.LinAlgError, ValueError):
                    continue
                y_fit = np.polyval(cy_coeffs, t)
                ss_res = np.sum((y - y_fit) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2_y = 1.0 - ss_res / (ss_tot + 1e-9)
                if r2_y >= self.cfg.free_flight_r2_min:
                    free_mask[i : i + win] = True

        segments: List[Tuple[int, int]] = []
        in_seg = False
        seg_start_idx = 0

        for i, flag in enumerate(free_mask):
            if flag and not in_seg:
                in_seg = True
                seg_start_idx = i
            elif not flag and in_seg:
                in_seg = False
                if i - seg_start_idx >= win:
                    segments.append((fids[seg_start_idx], fids[i - 1]))
        if in_seg and n - seg_start_idx >= win:
            segments.append((fids[seg_start_idx], fids[-1]))

        return segments

    def _find_release_frame(
        self,
        track: Track,
        free_segs: List[Tuple[int, int]],
        hoop_center: Optional[Tuple[int, int]],
        hoop_bbox: Optional[Tuple[int, int, int, int]],
    ) -> Optional[int]:
        fids_set = set(track.points.keys())
        img_h = self.cfg.height
        min_cy_for_release = int(img_h * self.cfg.release_min_cy_ratio)

        for seg_start, seg_end in free_segs:
            seg_fids = sorted(
                f for f in fids_set if seg_start <= f <= seg_end
            )
            if len(seg_fids) < 4:
                continue

            cys_seg = [track.points[f].center[1] for f in seg_fids]

            hx1_h = hy1_h = hx2_h = hy2_h = 0
            hw_h = hh_h = 1
            has_hoop = hoop_bbox is not None and hoop_center is not None
            if has_hoop:
                hx1_h, hy1_h, hx2_h, hy2_h = hoop_bbox
                hw_h = max(hx2_h - hx1_h, 1)
                hh_h = max(hy2_h - hy1_h, 1)
                margin_x = hw_h * self.cfg.release_hoop_margin_x
                margin_y_up = hh_h * self.cfg.release_hoop_margin_y_up
                margin_y_dn = hh_h * self.cfg.release_hoop_margin_y_dn
                # ★ 出手 cy 下界 = max(画面40%, 篮筐 cy)
                min_cy_for_release = max(
                    min_cy_for_release, hoop_center[1]
                )

            for i, fid in enumerate(seg_fids[:-1]):
                cy_curr = cys_seg[i]
                cy_next = cys_seg[i + 1]

                # ★ 画面下方 40% 不检测出手
                if cy_curr > img_h * (1 - 0.40):
                    continue

                # 篮筐附近排除
                if has_hoop:
                    cx_r = track.points[fid].center[0]
                    cy_r = track.points[fid].center[1]
                    near_x = hx1_h - margin_x < cx_r < hx2_h + margin_x
                    near_y = (
                        hy1_h - margin_y_up < cy_r < hy2_h + margin_y_dn
                    )
                    if near_x and near_y:
                        continue

                # 弹地判定排除
                if i >= self.cfg.release_bounce_consec_frames:
                    prev_fids = seg_fids[max(0, i - 8) : i]
                    if len(prev_fids) >= 2:
                        consec_fall = 0
                        max_consec = 0
                        for j in range(len(prev_fids) - 1):
                            dy = (
                                track.points[prev_fids[j + 1]].center[1]
                                - track.points[prev_fids[j]].center[1]
                            )
                            if dy > self.cfg.release_bounce_fall_speed:
                                consec_fall += 1
                                max_consec = max(max_consec, consec_fall)
                            else:
                                consec_fall = 0
                        if (
                            max_consec
                            >= self.cfg.release_bounce_consec_frames
                        ):
                            continue

                # 上升动作检测
                if cy_curr <= cy_next:
                    continue

                cy_apex = min(cys_seg[i:])
                if cy_curr - cy_apex < self.cfg.release_min_rise_px:
                    continue

                return fid

        return None

    def _phase5_physics_analysis(
        self, tracks: List[Track], fps: float
    ) -> Tuple[Optional[float], Dict[int, Optional[int]]]:
        t5_start = time.time()

        hcx_list: List[float] = []
        hcy_list: List[float] = []
        hx1_list: List[float] = []
        hy1_list: List[float] = []
        hx2_list: List[float] = []
        hy2_list: List[float] = []

        for tk in tracks:
            if tk.cls != "hoop":
                continue
            for pt in tk.points.values():
                hcx_list.append(pt.center[0])
                hcy_list.append(pt.center[1])
                hx1_list.append(pt.bbox[0])
                hy1_list.append(pt.bbox[1])
                hx2_list.append(pt.bbox[2])
                hy2_list.append(pt.bbox[3])

        hoop_center_stable: Optional[Tuple[int, int]] = None
        hoop_bbox_stable: Optional[Tuple[int, int, int, int]] = None
        if hcx_list:
            hoop_center_stable = (
                int(np.median(hcx_list)),
                int(np.median(hcy_list)),
            )
            hoop_bbox_stable = (
                int(np.median(hx1_list)),
                int(np.median(hy1_list)),
                int(np.median(hx2_list)),
                int(np.median(hy2_list)),
            )

        ball_tracks = [tk for tk in tracks if tk.cls == "basketball"]

        g_estimates: List[float] = []
        for tk in ball_tracks:
            g = self._estimate_pixel_gravity(tk, fps)
            if g is not None:
                g_estimates.append(g)
                print(
                    f"[Preprocessor] Phase5   轨迹#{tk.track_id} "
                    f"重力估算: {g:.4f} px/frame²"
                )

        global_gravity_px: Optional[float] = None
        if g_estimates:
            global_gravity_px = float(np.median(g_estimates))
            print(
                f"[Preprocessor] Phase5 全局像素重力: "
                f"{global_gravity_px:.4f} px/frame² "
                f"({len(g_estimates)} 条轨迹参与)"
            )
        else:
            print(
                "[Preprocessor] Phase5 警告: 无法估算像素重力，"
                "篮球自由飞行轨迹不足"
            )

        g_for_analysis = (
            global_gravity_px if global_gravity_px is not None else 0.3
        )

        release_frames: Dict[int, Optional[int]] = {}
        for tk in ball_tracks:
            free_segs = self._find_free_flight_segments(
                tk, g_for_analysis, fps, hoop_bbox_stable
            )
            rel_frame = self._find_release_frame(
                tk, free_segs, hoop_center_stable, hoop_bbox_stable
            )
            release_frames[tk.track_id] = rel_frame
            print(
                f"[Preprocessor] Phase5   轨迹#{tk.track_id} "
                f"{len(free_segs)} 段自由飞行, "
                + (
                    f"出手帧=frame{rel_frame}"
                    if rel_frame is not None
                    else "未检测到出手"
                )
            )

        n_found = sum(1 for v in release_frames.values() if v is not None)
        print(
            f"[Preprocessor] Phase5 完成: 共检测到 {n_found} 个出手事件, "
            f"耗时 {time.time()-t5_start:.2f}s"
        )
        return global_gravity_px, release_frames

    # ================================================================
    # 公开接口
    # ================================================================

    def process(
        self,
        video_path: str,
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
    ) -> Tuple[
        Dict[int, List[StableDetection2D]], int, float, PhysicsMetadata
    ]:
        print(f"[Preprocessor] ═══ 开始预处理: {video_path} ═══")
        t_start = time.time()

        raw, total, fps = self._phase1_detect_all(video_path, progress_cb)
        tracks = self._phase2_build_tracks(raw, total)
        tracks = self._phase3_smooth(tracks)

        global_gravity_px, release_frames = self._phase5_physics_analysis(
            tracks, fps
        )

        stable = self._phase4_export(tracks, total, release_frames)

        physics = PhysicsMetadata(
            estimated_gravity_px=global_gravity_px,
            release_frames=release_frames,
        )

        print(
            f"[Preprocessor] ═══ 预处理完成，总耗时 "
            f"{time.time()-t_start:.1f}s ═══"
        )
        return stable, total, fps, physics