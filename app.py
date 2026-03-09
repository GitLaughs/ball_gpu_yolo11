# app.py

from __future__ import annotations

import os
import time
import threading
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request, Response, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO

from engine.config import EngineConfig
from engine.preprocessor import VideoPreprocessor
from engine.vision_engine import VisionEngine, StereoParams
from engine.shot_engine import ShotEngine

app = Flask(__name__, static_folder="./static", static_url_path="/static")
CORS(app)

ALLOWED_EXT = {"avi", "mp4", "mov", "mkv", "flv", "wmv"}
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def allowed_file(fn: str) -> bool:
    return "." in fn and fn.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def build_stereo(w: int = 640, h: int = 480) -> StereoParams:
    lm = np.array(
        [
            [834.45170, 0, 341.14720],
            [0, 834.98848, 239.35073],
            [0.0, 0.0, 1.0],
        ]
    )
    rm = np.array(
        [
            [838.79197, 0, 326.20140],
            [0, 839.51716, 240.13509],
            [0.0, 0.0, 1.0],
        ]
    )
    ld = np.array([[-0.08620, 0.18821, -0.00029, -0.00201, 0.00000]])
    rd = np.array([[-0.08728, 0.08733, -0.00064, 0.00157, 0.00000]])
    R = np.array(
        [
            [0.999953933, -0.005218517, 0.008055977],
            [0.005212219, 0.999986094, 0.000802640],
            [-0.008060054, -0.000760614, 0.999967227],
        ]
    )
    T = np.array([-404.16546, -3.11006, 3.71327])
    sz = (w, h)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(lm, ld, rm, rd, sz, R, T)
    lm1, lm2 = cv2.initUndistortRectifyMap(
        lm, ld, R1, P1, sz, cv2.CV_16SC2
    )
    rm1, rm2 = cv2.initUndistortRectifyMap(
        rm, rd, R2, P2, sz, cv2.CV_16SC2
    )
    return StereoParams(Q, lm1, lm2, rm1, rm2)


# ══════════════════════════════════════════════
#  全局应用状态
# ══════════════════════════════════════════════


class AppState:
    def __init__(self):
        self.lock = threading.RLock()
        self.is_running = False
        self.is_paused = False
        self.is_preprocessing = False
        self.preprocess_progress = ""
        self.thread: Optional[threading.Thread] = None

        self.cfg = EngineConfig()
        self.video_path = "./ball.avi"
        self.model_path = (
            "./best_yolo11.pt"
            if os.path.exists("./best_yolo11.pt")
            else "./best.pt"
        )

        self.stats = {
            "fps": 0.0,
            "current_frame": 0,
            "total_frames": 0,
            "process_time": 0.0,
            "gpu_memory": 0.0,
            "basketball_count": 0,
            "hoops_count": 0,
            "current_video": "ball.avi",
            "phase": "idle",
        }

        self.latest_frame_left = None
        self.latest_frame_right = None
        self.latest_frame_depth = None

        self.shot_info = {
            "phase": "idle",
            "total_shots": 0,
            "total_scored": 0,
            "accuracy": "0%",
            "last_shot": None,
            "shot_history": [],
        }


state = AppState()


# ══════════════════════════════════════════════
#  检测主线程
# ══════════════════════════════════════════════


def detection_loop():
    vision = None
    try:
        cfg = state.cfg

        print(f"[INFO] 加载模型: {state.model_path}")
        if not os.path.exists(state.model_path):
            print(f"[ERROR] 模型文件不存在: {state.model_path}")
            with state.lock:
                state.is_running = False
            return

        model = YOLO(state.model_path)
        model.conf = cfg.conf_threshold
        model.iou = cfg.iou_threshold
        if DEVICE.startswith("cuda") and torch.cuda.is_available():
            model.to(DEVICE)

        stereo = build_stereo(cfg.width, cfg.height)

        if not os.path.exists(state.video_path):
            print(f"[ERROR] 视频文件不存在: {state.video_path}")
            with state.lock:
                state.is_running = False
            return

        # ════ 第一阶段: 离线预处理 ════
        with state.lock:
            state.is_preprocessing = True
            state.stats["phase"] = "preprocessing"

        def progress_cb(cur, total, msg):
            with state.lock:
                state.preprocess_progress = msg
                state.stats["current_frame"] = cur
                state.stats["total_frames"] = total

        preprocessor = VideoPreprocessor(model, cfg, DEVICE)
        stable_dets, total_frames, vid_fps, physics_meta = (
            preprocessor.process(state.video_path, progress_cb)
        )

        with state.lock:
            state.is_preprocessing = False
            state.stats["phase"] = "playback"
            state.stats["total_frames"] = total_frames

        det_count = sum(len(v) for v in stable_dets.values())
        ball_frames = sum(
            1
            for v in stable_dets.values()
            if any(d.cls == "basketball" for d in v)
        )
        hoop_frames = sum(
            1
            for v in stable_dets.values()
            if any(d.cls == "hoop" for d in v)
        )
        print(
            f"[INFO] 预处理完成: 共 {det_count} 个检测 | "
            f"篮球帧 {ball_frames}/{total_frames} | "
            f"篮筐帧 {hoop_frames}/{total_frames}"
        )
        if physics_meta.estimated_gravity_px is not None:
            print(
                f"[INFO] 估算像素重力: "
                f"{physics_meta.estimated_gravity_px:.4f} px/frame²"
            )
        n_release = sum(
            1
            for v in physics_meta.release_frames.values()
            if v is not None
        )
        print(f"[INFO] 检测到 {n_release} 个出手事件")

        # ════ 第二阶段: 回放 + 投篮分析 ════
        vision = VisionEngine(stereo=stereo, cfg=cfg)
        shot_engine = ShotEngine(img_w=cfg.width, img_h=cfg.height)

        cap = cv2.VideoCapture(state.video_path)
        if not cap.isOpened():
            with state.lock:
                state.is_running = False
            return

        frame_interval = 1.0 / (vid_fps if vid_fps > 0 else 30.0)
        frame_id = 0
        fps_smooth = 0.0

        print(
            f"[INFO] 开始回放: {total_frames} 帧 @ {vid_fps:.1f} fps "
            f"(流水线 stereo 已启用)"
        )

        while True:
            with state.lock:
                if not state.is_running:
                    break
                paused = state.is_paused

            if paused:
                time.sleep(0.05)
                continue

            ok, frame = cap.read()
            if not ok:
                with state.lock:
                    state.is_running = False
                break

            frame_id += 1
            t0 = time.time()

            dets_2d = stable_dets.get(frame_id, [])
            display_info = shot_engine.get_display_info(frame_id)

            try:
                left_draw, right_draw, depth_color, fd = (
                    vision.process_frame(
                        frame,
                        frame_id,
                        dets_2d,
                        shot_display_info=display_info,
                    )
                )
            except Exception:
                import traceback

                traceback.print_exc()
                continue

            # ── 提取最高置信度的篮球/篮筐 ──
            ball_det = None
            hoop_det = None
            for d in fd.detections:
                if d.cls == "basketball" and (
                    ball_det is None or d.conf > ball_det.conf
                ):
                    ball_det = d
                if d.cls == "hoop" and (
                    hoop_det is None or d.conf > hoop_det.conf
                ):
                    hoop_det = d

            # ── 判断当前帧是否为 Phase 5 确认的出手帧 ──
            is_physics_release = any(
                d.is_release_point and d.cls == "basketball"
                for d in fd.detections
            )

            ts_now = time.time()
            shot_event = shot_engine.update(
                frame_id,
                ts_now,
                ball_det,
                hoop_det,
                is_physics_release=is_physics_release,
            )

            bc = sum(1 for d in fd.detections if d.cls == "basketball")
            hc = sum(1 for d in fd.detections if d.cls == "hoop")
            proc = time.time() - t0
            fps_smooth = 0.9 * fps_smooth + 0.1 * (
                1.0 / max(proc, 1e-4)
            )

            with state.lock:
                state.stats["current_frame"] = frame_id
                state.stats["fps"] = round(fps_smooth, 1)
                state.stats["process_time"] = round(proc * 1000, 1)
                state.stats["basketball_count"] = bc
                state.stats["hoops_count"] = hc
                if torch.cuda.is_available():
                    state.stats["gpu_memory"] = round(
                        torch.cuda.memory_allocated() / 1024**3, 2
                    )

                state.latest_frame_left = left_draw.copy()
                state.latest_frame_right = right_draw.copy()
                state.latest_frame_depth = depth_color.copy()

                # ── 同步投篮信息 ──
                state.shot_info["phase"] = shot_engine.phase.value
                state.shot_info["total_shots"] = shot_engine.total_shots
                state.shot_info["total_scored"] = (
                    shot_engine.total_scored
                )
                if shot_engine.total_shots > 0:
                    pct = (
                        shot_engine.total_scored
                        / shot_engine.total_shots
                        * 100
                    )
                    state.shot_info["accuracy"] = f"{pct:.0f}%"

                if shot_event is not None:
                    shot_data = {
                        "is_scored": shot_event.is_scored,
                        "arc_height": round(shot_event.arc_height_px),
                        "duration": round(shot_event.duration_s, 2),
                        "release": list(shot_event.release_point),
                        "apex": list(shot_event.apex_point),
                        "frame_start": shot_event.frame_start,
                        "frame_end": shot_event.frame_end,
                        "time": datetime.now().strftime("%H:%M:%S"),
                    }
                    state.shot_info["last_shot"] = shot_data
                    state.shot_info["shot_history"].append(shot_data)
                    if len(state.shot_info["shot_history"]) > 20:
                        state.shot_info["shot_history"] = state.shot_info[
                            "shot_history"
                        ][-20:]

                    tag = (
                        "✓ 命中"
                        if shot_event.is_scored
                        else "✗ 未中"
                    )
                    print(
                        f"[投篮] #{shot_engine.total_shots:02d} {tag} | "
                        f"弧高={shot_event.arc_height_px:.0f}px | "
                        f"时长={shot_event.duration_s:.2f}s | "
                        f"帧段 {shot_event.frame_start}-"
                        f"{shot_event.frame_end}"
                    )

            # 控帧速率
            elapsed = time.time() - t0
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

        cap.release()
        print("[INFO] 回放结束")

    except Exception as e:
        import traceback

        print(f"[FATAL] {e}")
        traceback.print_exc()
    finally:
        if vision is not None:
            vision.shutdown()
        with state.lock:
            state.is_running = False
            state.is_preprocessing = False
            state.stats["phase"] = "idle"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ══════════════════════════════════════════════
#  Flask 路由
# ══════════════════════════════════════════════


@app.route("/api/start", methods=["POST"])
def start_detection():
    with state.lock:
        if state.is_running:
            return jsonify({"success": False, "message": "已在运行中"})
        state.is_running = True
        state.is_paused = False
        state.shot_info = {
            "phase": "idle",
            "total_shots": 0,
            "total_scored": 0,
            "accuracy": "0%",
            "last_shot": None,
            "shot_history": [],
        }
    state.thread = threading.Thread(target=detection_loop, daemon=True)
    state.thread.start()
    return jsonify({"success": True})


@app.route("/api/pause", methods=["POST"])
def pause():
    with state.lock:
        if not state.is_running:
            return jsonify({"success": False, "message": "未在运行"})
        state.is_paused = not state.is_paused
        return jsonify({"success": True, "paused": state.is_paused})


@app.route("/api/stop", methods=["POST"])
def stop():
    with state.lock:
        state.is_running = False
        state.is_paused = False
    return jsonify({"success": True})


@app.route("/api/status")
def api_status():
    with state.lock:
        return jsonify(
            {
                "success": True,
                "is_running": state.is_running,
                "is_paused": state.is_paused,
                "is_preprocessing": state.is_preprocessing,
                "preprocess_progress": state.preprocess_progress,
                "stats": state.stats,
                "shot_info": state.shot_info,
                "device": (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else "CPU"
                ),
            }
        )


@app.route("/api/shot_analysis")
@app.route("/api/shot-analysis")
def shot_analysis():
    with state.lock:
        return jsonify(state.shot_info)


@app.route("/api/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify({"success": False, "message": "请求中无文件字段"})
    f = request.files["video"]
    if not f.filename or not allowed_file(f.filename):
        return jsonify({"success": False, "message": "不支持的文件格式"})
    os.makedirs("./uploads", exist_ok=True)
    save_path = os.path.join("./uploads", f.filename)
    f.save(save_path)
    with state.lock:
        state.video_path = save_path
        state.stats["current_video"] = f.filename
    return jsonify({"success": True, "filename": f.filename})


def gen_frames(ftype: str = "left"):
    """MJPEG 流生成器"""
    while True:
        with state.lock:
            frame_map = {
                "left": state.latest_frame_left,
                "right": state.latest_frame_right,
                "depth": state.latest_frame_depth,
            }
            f = frame_map.get(ftype)

        if f is None:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            with state.lock:
                msg = (
                    (state.preprocess_progress or "预处理中...")
                    if state.is_preprocessing
                    else "等待视频..."
                )
            cv2.putText(
                blank,
                msg,
                (30, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 200, 200),
                1,
            )
            f = blank

        ok, buf = cv2.imencode(
            ".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 85]
        )
        if ok:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buf.tobytes()
                + b"\r\n"
            )
        time.sleep(0.033)


@app.route("/api/video_feed/<stream_type>")
def video_feed(stream_type: str):
    return Response(
        gen_frames(stream_type),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/")
def index():
    fp = os.path.join(app.static_folder, "index.html")
    if os.path.exists(fp):
        return send_from_directory(app.static_folder, "index.html")
    return "index.html not found", 404


if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    os.makedirs("uploads", exist_ok=True)
    print(f"[INFO] 运行设备: {DEVICE}")
    print(f"[INFO] 访问地址: http://127.0.0.1:5002")
    app.run(host="0.0.0.0", port=5002, debug=False, threaded=True)