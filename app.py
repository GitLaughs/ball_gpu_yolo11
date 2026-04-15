# /app.py
from __future__ import annotations

import os
import sys
import time
import threading
import webbrowser
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request, Response, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO

from engine.config import EngineConfig
from engine.preprocessor import PreprocessPipeline
from engine.vision_engine import VisionEngine, StereoParams
from engine.shot_engine import ShotEngine


def resource_path(relative_path: str) -> str:
    # Support both source run and PyInstaller bundle mode.
    base_path = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base_path, relative_path)


app = Flask(__name__, static_folder=resource_path("static"), static_url_path="/static")
CORS(app)

ALLOWED_EXT = {"avi", "mp4", "mov", "mkv", "flv", "wmv"}
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def allowed_file(fn: str) -> bool:
    return "." in fn and fn.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def build_stereo(w: int = 640, h: int = 480) -> StereoParams:
    lm = np.array([[834.45170, 0, 341.14720], [0, 834.98848, 239.35073], [0.0, 0.0, 1.0]])
    rm = np.array([[838.79197, 0, 326.20140], [0, 839.51716, 240.13509], [0.0, 0.0, 1.0]])
    ld = np.array([[-0.08620, 0.18821, -0.00029, -0.00201, 0.00000]])
    rd = np.array([[-0.08728, 0.08733, -0.00064, 0.00157, 0.00000]])
    R = np.array([[0.999953933, -0.005218517, 0.008055977], [0.005212219, 0.999986094, 0.000802640], [-0.008060054, -0.000760614, 0.999967227]])
    T = np.array([-404.16546, -3.11006, 3.71327])
    sz = (w, h)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(lm, ld, rm, rd, sz, R, T)
    lm1, lm2 = cv2.initUndistortRectifyMap(lm, ld, R1, P1, sz, cv2.CV_16SC2)
    rm1, rm2 = cv2.initUndistortRectifyMap(rm, rd, R2, P2, sz, cv2.CV_16SC2)
    return StereoParams(Q, lm1, lm2, rm1, rm2)

class AppState:
    def __init__(self):
        self.lock = threading.RLock()
        self.is_running = False
        self.is_paused = False
        self.is_preprocessing = False
        self.preprocess_progress = ""
        self.thread: Optional[threading.Thread] = None

        self.cfg = EngineConfig()
        self.video_path = next(
            (
                path for path in (
                    "./ball.avi",
                    "./ball.mp4",
                    "./basketball1.mp4",
                    "./basketball2.mp4",
                )
                if os.path.exists(path)
            ),
            "./ball.mp4",
        )
        self.model_path = next(
            (
                path for path in (
                    "./best_yolo11.pt",
                    "./best.pt",
                    resource_path("best_yolo11.pt"),
                    resource_path("best.pt"),
                )
                if os.path.exists(path)
            ),
            "./best.pt",
        )

        self.stats = {
            "fps": 0.0, "current_frame": 0, "total_frames": 0,
            "process_time": 0.0, "gpu_memory": 0.0, "phase": "idle",
            "basketball_count": 0, "hoops_count": 0,
        }

        self.latest_frame_left = None
        self.latest_frame_right = None
        self.latest_frame_depth = None

        self.shot_info = {
            "phase": "idle", "total_shots": 0, "total_scored": 0,
            "accuracy": "0%", "last_shot": None
        }

state = AppState()

def detection_loop():
    vision = None
    try:
        cfg = state.cfg
        if not os.path.exists(state.model_path):
            with state.lock: state.is_running = False
            return

        model = YOLO(state.model_path)
        model.conf = cfg.conf_threshold
        model.iou = cfg.iou_threshold
        if DEVICE.startswith("cuda") and torch.cuda.is_available():
            model.to(DEVICE)

        stereo = build_stereo(cfg.width, cfg.height)
        if not os.path.exists(state.video_path):
            with state.lock: state.is_running = False
            return

        with state.lock:
            state.is_preprocessing = True
            state.stats["phase"] = "preprocessing"

        pipeline = PreprocessPipeline(model, cfg, DEVICE)
        pipeline.start(state.video_path, None)

        pipeline.wait_for_info(timeout=30.0)
        vid_fps = pipeline.fps
        total_frames = pipeline.total_frames
        frame_interval = 1.0 / (vid_fps if vid_fps > 0 else 30.0)

        with state.lock:
            state.stats["total_frames"] = total_frames

        # ★ 等待【全量预处理和物理跟踪】完全结束以换换取完全平滑流畅的体验
        print("[INFO] 等待预处理和轨迹物理排查全量运行完毕...")
        while True:
            with state.lock:
                if not state.is_running: return
                state.preprocess_progress = pipeline.progress_msg
            if pipeline.is_all_done(): break
            if pipeline.error is not None: raise pipeline.error
            time.sleep(0.05)

        with state.lock:
            state.is_preprocessing = False
            state.stats["phase"] = "playback"

        vision = VisionEngine(stereo=stereo, cfg=cfg)
        shot_engine = ShotEngine(img_w=cfg.width, img_h=cfg.height)

        cap = cv2.VideoCapture(state.video_path)
        frame_id = 0
        fps_smooth = 0.0
        next_frame_time = time.time()

        while True:
            with state.lock:
                if not state.is_running: break
                paused = state.is_paused

            if paused:
                time.sleep(0.05)
                next_frame_time = time.time()
                continue

            now = time.time()
            sleep_time = next_frame_time - now
            if sleep_time > 0.001: time.sleep(sleep_time)
            elif sleep_time < -frame_interval * 5: next_frame_time = time.time()

            t_frame_start = time.time()
            ok, frame = cap.read()
            if not ok: break

            frame_id += 1
            dets = pipeline.get_dets(frame_id)

            W, H = cfg.width, cfg.height
            left = frame[0:H, 0:W]
            W_frame = frame.shape[1]

            # 调用立体相机处理
            if W_frame >= W * 2:
                _, frame_right_v, depth_color_v, dets_3d = vision.process_frame(frame, frame_id, dets, None)
            else:
                frame_right_v = left.copy()
                depth_color_v = np.zeros_like(left)

            ball_det = None
            hoop_det = None
            for d in dets:
                if d.cls == "basketball":
                    if ball_det is None or d.conf > ball_det.conf: ball_det = d
                elif d.cls == "hoop":
                    if hoop_det is None or d.conf > hoop_det.conf: hoop_det = d

            basketball_count = sum(1 for d in dets if d.cls == "basketball")
            hoops_count = sum(1 for d in dets if d.cls == "hoop")

            is_physics_release = hasattr(ball_det, "is_release_point") and ball_det.is_release_point

            shot_event = shot_engine.update(
                frame_id=frame_id, timestamp=frame_id / vid_fps,
                ball_det=ball_det, hoop_det=hoop_det, is_physics_release=is_physics_release,
            )

            display_info = shot_engine.get_display_info(frame_id)
            
            # 手动清爽作画
            vis_left = left.copy()
            vis_left = _draw_clean_visualization(vis_left, dets, display_info, frame_id, fps_smooth)

            t_elapsed = time.time() - t_frame_start
            fps_current = 1.0 / max(t_elapsed, 0.001)
            fps_smooth = fps_smooth * 0.9 + fps_current * 0.1

            with state.lock:
                state.latest_frame_left = vis_left
                state.latest_frame_right = frame_right_v
                state.latest_frame_depth = depth_color_v
                state.stats.update({
                    "fps": round(fps_smooth, 1),
                    "current_frame": frame_id,
                    "process_time": round(t_elapsed * 1000, 1),
                    "phase": display_info["phase"],
                    "basketball_count": basketball_count,
                    "hoops_count": hoops_count,
                })

                if DEVICE.startswith("cuda") and torch.cuda.is_available():
                    state.stats["gpu_memory"] = round(torch.cuda.memory_allocated() / 1024**2, 1)

                acc = "0%" if shot_engine.total_shots == 0 else f"{shot_engine.total_scored / shot_engine.total_shots * 100:.0f}%"
                state.shot_info.update({
                    "phase": display_info["phase"],
                    "total_shots": shot_engine.total_shots,
                    "total_scored": shot_engine.total_scored,
                    "accuracy": acc,
                    "shot_trajectory": display_info.get("shot_trajectory", []),
                    "physics_release_point": display_info.get("physics_release_point"),
                })

                if shot_event is not None:
                    state.shot_info["last_shot"] = {
                        "scored": shot_event.is_scored,
                        "arc_height": shot_event.arc_height_px,
                        "duration": shot_event.duration_s,
                        "final_trajectory": [(p.cx, p.cy) for p in shot_event.trajectory],
                        "final_release": list(shot_event.release_point),
                        "release": list(shot_event.release_point),   # 新增，供前端直接使用
                        "apex": list(shot_event.apex_point),         # 新增，弧顶位置
                    }

            next_frame_time += frame_interval

        cap.release()

    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        if vision is not None: vision.shutdown()
        with state.lock:
            state.is_running = False
            state.is_preprocessing = False


def _draw_clean_visualization(img, dets, display_info, frame_id, fps):
    H, W = img.shape[:2]

    # 清爽描边，消灭闪烁粗框
    for d in dets:
        x1, y1, x2, y2 = d.bbox
        if d.cls == "basketball":
            color = (0, 215, 255)
            cx, cy = d.center
            cv2.circle(img, (cx, cy), 12, color, 2)
            if hasattr(d, "is_release_point") and d.is_release_point:
                cv2.drawMarker(img, (cx, cy), (0, 255, 0), cv2.MARKER_STAR, 20, 2)
        elif d.cls == "hoop":
            color = (50, 150, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)

    # 画尾迹
    trail = display_info.get("trail", [])
    for i in range(1, len(trail)):
        cx0, cy0, fid0 = trail[i - 1]
        cx1, cy1, fid1 = trail[i]
        age = frame_id - fid1
        if age > 120: continue
        alpha = max(0.2, 1.0 - age / 120.0)
        cv2.line(img, (cx0, cy0), (cx1, cy1), (int(100*alpha), int(220*alpha), int(255*alpha)), 2)

    # 绘制抛物线
    shot_traj = display_info.get("shot_trajectory", [])
    if shot_traj and len(shot_traj) > 1:
        traj_color = (0, 255, 255) if display_info.get("phase") == "rising" else (255, 200, 0)
        cv2.polylines(img, [np.array(shot_traj)], False, traj_color, 2)

    # 展现结果
    if display_info.get("showing_result"):
        text = display_info.get("result_text", "")
        icolor = display_info.get("result_color", (255, 255, 255))
        tsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4)[0]
        cv2.rectangle(img, (W//2 - tsize[0]//2 - 10, 30), (W//2 + tsize[0]//2 + 10, 80), (0,0,0), -1)
        cv2.putText(img, text, (W//2 - tsize[0]//2, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, icolor, 4)

    return img


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/api/start", methods=["POST"])
def api_start():
    with state.lock:
        if state.is_running: return jsonify({"status": "already_running"})
        state.is_running = True; state.is_paused = False
    threading.Thread(target=detection_loop, daemon=True).start()
    return jsonify({"status": "started"})

@app.route("/api/stop", methods=["POST"])
def api_stop():
    with state.lock: state.is_running = False
    return jsonify({"status": "stopped"})

@app.route("/api/pause", methods=["POST"])
def api_pause():
    with state.lock: state.is_paused = not state.is_paused
    return jsonify({"status": "paused" if state.is_paused else "resumed"})

@app.route("/api/status")
def api_status():
    with state.lock:
        return jsonify({
            "is_running": state.is_running, "is_paused": state.is_paused,
            "is_preprocessing": state.is_preprocessing, "preprocess_progress": state.preprocess_progress,
            "stats": state.stats, "shot_info": state.shot_info,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        })

def make_stream(source_var_name):
    def gen():
        while True:
            with state.lock:
                frame = getattr(state, source_var_name)
                running = state.is_running
            if frame is not None:
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok: yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
            if not running: break
            time.sleep(0.03)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/video_feed")
def video_feed(): return make_stream("latest_frame_left")

@app.route("/api/video_feed/right")
def video_feed_right(): return make_stream("latest_frame_right")

@app.route("/api/video_feed/depth")
def video_feed_depth(): return make_stream("latest_frame_depth")

@app.route("/api/upload_video", methods=["POST"])
def upload_video():
    f = request.files.get("file")
    if not f or not allowed_file(f.filename): return jsonify({"error": "非法文件"}), 400
    os.makedirs("uploads", exist_ok=True)
    save_path = os.path.join("uploads", f"vid_{datetime.now().strftime('%M%S')}.mp4")
    f.save(save_path)
    with state.lock: state.video_path = save_path
    return jsonify({"status": "uploaded", "filename": os.path.basename(save_path)})


def open_frontend_in_browser() -> None:
    # Delay browser open slightly to allow Flask server socket to bind.
    def _open() -> None:
        try:
            opened = webbrowser.open("http://127.0.0.1:5000", new=2)
            if opened:
                print("[INFO] Frontend opened in browser: http://127.0.0.1:5000")
            else:
                print("[WARN] Browser did not open automatically. Please visit http://127.0.0.1:5000")
        except Exception:
            print("[WARN] Failed to open browser automatically. Please visit http://127.0.0.1:5000")

    threading.Timer(1.2, _open).start()

if __name__ == "__main__":
    open_frontend_in_browser()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
