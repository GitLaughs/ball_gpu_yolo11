# trajectory_analyzer.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class AnalyzerConfig:
    # 数据清洗
    min_points: int = 6
    dedup_eps: float = 1e-6          # 去重复点阈值
    outlier_mad_z: float = 4.5       # z方向离群点阈值(MAD倍数)
    outlier_mad_r: float = 5.0       # (x,z)半径离群点阈值(MAD倍数)

    # 平滑（Savitzky–Golay）
    smooth_enable: bool = True
    smooth_window: int = 7           # 必须是奇数，且 <= 点数
    smooth_poly: int = 2             # 2 或 3 比较常用

    # 拟合（鲁棒加权二次拟合）
    robust_iters: int = 8
    huber_k: float = 1.5             # Huber阈值（越小越“硬”）
    ridge: float = 1e-8              # 防止矩阵病态

    # 预测轨迹采样
    pred_n: int = 40
    pred_extend: float = 0.10        # 预测时在[minX,maxX]两侧各扩展比例

    # 速度/角度估计
    default_fps: float = 30.0        # 没有时间戳时用这个估计速度
    angle_deg_clip: float = 89.0     # 防止数值爆炸

    # 可选：篮筐高度（如果你的坐标系中 z 表示深度而非高度，这里仅做“到篮筐处的z预测”，不当做物理高度）
    hoop_z_tol: float = 0.25         # 允许接近篮筐z的容差（单位同z）


class TrajectoryAnalyzer:

    def __init__(self, cfg: Optional[AnalyzerConfig] = None):
        self.cfg = cfg or AnalyzerConfig()

    # -------------------- Public API --------------------

    def analyze_shot(
        self,
        position_data: Sequence[Sequence[float]],
        hoop_position: Optional[Sequence[float]] = None,
        timestamps: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:

        pts = self._parse_points(position_data, timestamps=timestamps)

        out: Dict[str, Any] = {
            "predicted_trajectory": [],
            "actual": {"velocity": 0.0, "angle": 0.0},
            "fit": {},
            "quality": {},
            "hoop": {},
            "debug": {},
        }

        if pts is None or len(pts) < self.cfg.min_points:
            out["quality"] = {"ok": False, "reason": "not_enough_points", "n": 0 if pts is None else len(pts)}
            return out

        # 清洗 + 去离群
        pts0 = pts.copy()
        pts = self._dedup(pts)
        pts = self._remove_outliers_mad(pts)

        if len(pts) < self.cfg.min_points:
            out["quality"] = {"ok": False, "reason": "too_many_outliers_or_duplicates", "n": len(pts)}
            out["debug"]["n_raw"] = len(pts0)
            return out

        # 准备序列：按时间排序（若无时间戳，按输入顺序）
        pts = self._sort_by_time(pts)

        # 拆分
        t = pts[:, 0]
        x = pts[:, 1]
        z = pts[:, 2]

        # 平滑（只对z做，x基本可认为单调变化）
        if self.cfg.smooth_enable and len(z) >= max(5, self.cfg.smooth_window):
            z_s = self._savgol(z, window=self._odd_clamp(self.cfg.smooth_window, len(z)), poly=self.cfg.smooth_poly)
        else:
            z_s = z

        # 鲁棒拟合 z = a x^2 + b x + c
        coef, fit_stats = self._robust_quad_fit(x, z_s)
        a, b, c = coef

        # 预测轨迹
        x_pred = self._linspace_extended(x.min(), x.max(), self.cfg.pred_n, self.cfg.pred_extend)
        z_pred = a * x_pred**2 + b * x_pred + c
        pred_traj = [[float(xx), float(zz)] for xx, zz in zip(x_pred, z_pred)]
        out["predicted_trajectory"] = pred_traj

        # 速度/角度估计
        vel, ang = self._estimate_velocity_angle(t, x, z_s)
        out["actual"]["velocity"] = float(vel)
        out["actual"]["angle"] = float(ang)

        # 拟合信息
        out["fit"] = {
            "model": "z = a*x^2 + b*x + c",
            "coef": {"a": float(a), "b": float(b), "c": float(c)},
            "rmse": float(fit_stats["rmse"]),
            "mad_resid": float(fit_stats["mad"]),
            "iters": int(fit_stats["iters"]),
        }

        # 与篮筐的关系（你传的是 hoop_position=[x,z]）
        if hoop_position is not None and len(hoop_position) >= 2:
            hx = float(hoop_position[0])
            hz = float(hoop_position[1])
            z_at_hx = float(a * hx * hx + b * hx + c)
            dz = z_at_hx - hz
            out["hoop"] = {
                "x": hx,
                "z": hz,
                "predicted_z_at_hoop_x": z_at_hx,
                "dz": float(dz),
                "close_to_hoop_z": bool(abs(dz) <= self.cfg.hoop_z_tol),
            }

        # 质量评估（简单可用）
        mono = self._monotonic_ratio(x)
        out["quality"] = {
            "ok": True,
            "n_raw": int(len(pts0)),
            "n_used": int(len(pts)),
            "x_monotonic_ratio": float(mono),  # 越接近1越好
            "has_time": bool(np.any(np.diff(t) > 0)),
        }

        # debug
        out["debug"] = {
            "x_range": [float(x.min()), float(x.max())],
            "z_range": [float(z.min()), float(z.max())],
        }
        return out

    # -------------------- Parsing --------------------

    def _parse_points(
        self,
        position_data: Sequence[Sequence[float]],
        timestamps: Optional[Sequence[float]] = None,
    ) -> Optional[np.ndarray]:
        if position_data is None:
            return None
        if len(position_data) == 0:
            return None

        arr = np.array(position_data, dtype=float)
        if arr.ndim != 2 or arr.shape[0] < 2:
            return None

        # 支持几种形态：
        # [x,z]
        # [t,x,z]
        # [t,x,y,z] -> 只取t,x,z
        if arr.shape[1] == 2:
            x = arr[:, 0]
            z = arr[:, 1]
            if timestamps is not None and len(timestamps) == len(x):
                t = np.array(timestamps, dtype=float)
            else:
                # 没有时间戳：用等间隔“伪时间”
                t = np.arange(len(x), dtype=float) / float(self.cfg.default_fps)
            pts = np.stack([t, x, z], axis=1)

        elif arr.shape[1] == 3:
            # 认为是 [t,x,z]
            pts = arr[:, [0, 1, 2]]

        elif arr.shape[1] >= 4:
            # 认为是 [t,x,y,z,...] 取 t,x,z
            pts = arr[:, [0, 1, 3]]

        else:
            return None

        # 过滤 NaN/Inf
        m = np.isfinite(pts).all(axis=1)
        pts = pts[m]
        if len(pts) == 0:
            return None
        return pts

    # -------------------- Cleaning --------------------

    def _dedup(self, pts: np.ndarray) -> np.ndarray:
        """去重复 (x,z) 点"""
        if len(pts) < 2:
            return pts
        keep = [0]
        for i in range(1, len(pts)):
            dx = abs(pts[i, 1] - pts[keep[-1], 1])
            dz = abs(pts[i, 2] - pts[keep[-1], 2])
            if dx + dz > self.cfg.dedup_eps:
                keep.append(i)
        return pts[np.array(keep, dtype=int)]

    def _remove_outliers_mad(self, pts: np.ndarray) -> np.ndarray:
        """用MAD去离群点（z方向 + 半径方向）"""
        if len(pts) < 6:
            return pts

        x = pts[:, 1]
        z = pts[:, 2]

        # z方向MAD
        z_med = np.median(z)
        z_mad = np.median(np.abs(z - z_med)) + 1e-12
        z_score = np.abs(z - z_med) / z_mad

        # (x,z)半径MAD
        cx, cz = np.median(x), np.median(z)
        r = np.sqrt((x - cx) ** 2 + (z - cz) ** 2)
        r_med = np.median(r)
        r_mad = np.median(np.abs(r - r_med)) + 1e-12
        r_score = np.abs(r - r_med) / r_mad

        m = (z_score <= self.cfg.outlier_mad_z) & (r_score <= self.cfg.outlier_mad_r)
        kept = pts[m]
        return kept if len(kept) >= 2 else pts

    def _sort_by_time(self, pts: np.ndarray) -> np.ndarray:
        # 若 t 全相等（伪时间也会单调），按 t 排序
        idx = np.argsort(pts[:, 0])
        return pts[idx]

    # -------------------- Smoothing --------------------

    def _odd_clamp(self, w: int, n: int) -> int:
        w = int(w)
        if w < 3:
            w = 3
        if w % 2 == 0:
            w += 1
        if w > n:
            w = n if n % 2 == 1 else n - 1
        return max(w, 3)

    def _savgol(self, y: np.ndarray, window: int, poly: int) -> np.ndarray:
        """
        Savitzky–Golay 滤波（不依赖 SciPy）
        对每个点用局部多项式最小二乘拟合，取中心值。
        """
        n = len(y)
        if window < 3 or window > n:
            return y.copy()
        poly = int(poly)
        poly = max(1, min(poly, window - 1))
        half = window // 2

        yy = y.astype(float)
        out = np.empty_like(yy)

        # 预计算设计矩阵 (k=-half..half)
        k = np.arange(-half, half + 1, dtype=float)
        A = np.vander(k, N=poly + 1, increasing=True)  # [1, k, k^2,...]
        # 取中心点的滤波系数：e0 = [1,0,0..] 对应常数项，求最小二乘解投影
        # 系数 = row of pseudo-inverse corresponding to constant term
        ATA = A.T @ A
        ATA += np.eye(ATA.shape[0]) * 1e-12
        pinv = np.linalg.solve(ATA, A.T)  # (poly+1, window)
        coeff = pinv[0, :]               # constant term weights

        # 边界：用镜像padding
        ypad = np.pad(yy, (half, half), mode="reflect")

        for i in range(n):
            seg = ypad[i : i + window]
            out[i] = float(np.dot(coeff, seg))
        return out

    # -------------------- Robust fitting --------------------

    def _robust_quad_fit(self, x: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        IRLS + Huber 权重
        拟合：z = a x^2 + b x + c
        """
        x = x.astype(float)
        z = z.astype(float)

        # 设计矩阵
        X = np.stack([x**2, x, np.ones_like(x)], axis=1)  # (n,3)

        # 初始：普通最小二乘
        w = np.ones(len(x), dtype=float)
        coef = self._wls_solve(X, z, w)

        iters = 0
        for it in range(self.cfg.robust_iters):
            iters = it + 1
            r = z - X @ coef
            mad = np.median(np.abs(r - np.median(r))) + 1e-12
            s = 1.4826 * mad + 1e-12  # robust scale

            # Huber weights
            u = r / (self.cfg.huber_k * s)
            w_new = np.ones_like(w)
            mask = np.abs(u) > 1.0
            w_new[mask] = 1.0 / (np.abs(u[mask]) + 1e-12)

            # 如果权重变化很小就停
            if np.linalg.norm(w_new - w) / (np.linalg.norm(w) + 1e-12) < 1e-3:
                w = w_new
                break
            w = w_new
            coef = self._wls_solve(X, z, w)

        resid = z - X @ coef
        rmse = float(np.sqrt(np.mean(resid**2)))
        mad_res = float(np.median(np.abs(resid - np.median(resid))) + 1e-12)

        return coef, {"rmse": rmse, "mad": mad_res, "iters": float(iters)}

    def _wls_solve(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        W = w.reshape(-1, 1)
        Xw = X * W
        yw = y * w
        A = Xw.T @ X + np.eye(X.shape[1]) * self.cfg.ridge
        b = Xw.T @ yw
        return np.linalg.solve(A, b)

    # -------------------- Velocity / Angle --------------------

    def _estimate_velocity_angle(self, t: np.ndarray, x: np.ndarray, z: np.ndarray) -> Tuple[float, float]:
        """
        速度：用末端若干点做线性回归估计 dx/dt, dz/dt
        角度：atan2(dz/dt, dx/dt)（在x-z平面）
        """
        n = len(x)
        if n < 2:
            return 0.0, 0.0

        # 取末尾K点更稳定
        K = int(min(8, n))
        tt = t[-K:]
        xx = x[-K:]
        zz = z[-K:]

        # 如果没有有效时间跨度（伪时间通常OK）
        dt = tt[-1] - tt[0]
        if abs(dt) < 1e-9:
            # fallback：用差分并假设fps
            dx = xx[-1] - xx[-2]
            dz = zz[-1] - zz[-2]
            speed = math.sqrt(dx * dx + dz * dz) * float(self.cfg.default_fps)
            angle = math.degrees(math.atan2(dz, dx)) if abs(dx) + abs(dz) > 0 else 0.0
            return float(speed), float(np.clip(angle, -self.cfg.angle_deg_clip, self.cfg.angle_deg_clip))

        # 线性回归：v = cov(t,p)/var(t)
        vt = np.var(tt)
        if vt < 1e-12:
            return 0.0, 0.0

        vx = float(np.cov(tt, xx, bias=True)[0, 1] / vt)
        vz = float(np.cov(tt, zz, bias=True)[0, 1] / vt)

        speed = float(math.sqrt(vx * vx + vz * vz))
        angle = float(math.degrees(math.atan2(vz, vx))) if abs(vx) + abs(vz) > 0 else 0.0
        angle = float(np.clip(angle, -self.cfg.angle_deg_clip, self.cfg.angle_deg_clip))
        return speed, angle

    # -------------------- Helpers --------------------

    def _linspace_extended(self, x_min: float, x_max: float, n: int, extend_ratio: float) -> np.ndarray:
        if x_max < x_min:
            x_min, x_max = x_max, x_min
        span = max(1e-9, x_max - x_min)
        ext = span * float(extend_ratio)
        return np.linspace(x_min - ext, x_max + ext, int(max(10, n)), dtype=float)

    def _monotonic_ratio(self, x: np.ndarray) -> float:
        """x 单调性的比例（用于粗评估轨迹是否“像投篮”）"""
        if len(x) < 3:
            return 1.0
        dx = np.diff(x)
        # 允许递增或递减，只要大多数同号即可
        pos = np.sum(dx > 0)
        neg = np.sum(dx < 0)
        return float(max(pos, neg) / max(1, len(dx)))


# 允许快速本地测试
if __name__ == "__main__":
    ta = TrajectoryAnalyzer()
    # 模拟一些抛物线 + 噪声
    xs = np.linspace(0, 4, 25)
    zs = -0.6 * xs**2 + 2.2 * xs + 0.2 + np.random.normal(0, 0.05, size=len(xs))
    data = [[float(x), float(z)] for x, z in zip(xs, zs)]
    res = ta.analyze_shot(data, hoop_position=[3.2, 2.0])
    print("velocity:", res["actual"]["velocity"], "angle:", res["actual"]["angle"])
    print("coef:", res["fit"]["coef"])
    print("hoop:", res["hoop"])