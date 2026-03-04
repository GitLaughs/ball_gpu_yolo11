from __future__ import annotations
import os, sys, time, threading
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request, Response, send_from_directory, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

from engine.config import EngineConfig
from engine.vision_engine import VisionEngine, load_yolo_model, StereoParams
from engine.shot_engine import ShotEngine

# ---------- Flask ----------
app = Flask(__name__, static_folder="../static", static_url_path="/static", template_folder=None)
CORS(app)

ALLOWED_EXTENSIONS = {"avi","mp4","mov","mkv","flv","wmv"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------- Device ----------
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# ---------- Stereo param builder (从你原 setup_stereo_params 拿过来) ----------
def build_stereo_params(width=640, height=480) -> StereoParams:
    left_camera_matrix = np.array([[834.45170, 0, 341.14720],
                                   [0, 834.98848, 239.35073],
                                   [0., 0., 1.]])
    right_camera_matrix = np.array([[838.79197, 0, 326.20140],
                                    [0, 839.51716, 240.13509],
                                    [0., 0., 1.]])
    left_distortion = np.array([[-0.08620, 0.18821, -0.00029, -0.00201, 0.00000]])
    right_distortion = np.array([[-0.08728, 0.08733, -0.00064, 0.00157, 0.00000]])
    R = np.array([[0.999953933085443, -0.00521851778828608, 0.008055977845488866],
                  [0.00521221927703499, 0.999986094172510, 0.000802640526958538],
                  [-0.00806005441431820, -0.000760614028764388, 0.999967227957565]])
    T = np.array([-404.16546, -3.11006, 3.71327])

    size = (width, height)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        left_camera_matrix, left_distortion,
        right_camera_matrix, right_distortion,
        size, R, T
    )
    left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
    right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)
    return StereoParams(Q=Q, left_map1=left_map1, left_map2=left_map2, right_map1=right_map1, right_map2=right_map2)

# ---------- Shared State ----------
class AppState:
    def __init__(self):
        self.lock = threading.RLock()
        self.is_running = False
        self.is_paused = False
        self.thread: Optional[threading.Thread] = None

        self.cfg = EngineConfig()
        self.video_path = "./ball.avi"
        self.model_path = "./best_yolo11.pt" if os.path.exists("./best_yolo11.pt") else "./best.pt"

        self.stats = {
            "fps": 0.0,
            "current_frame": 0,
            "total_frames": 0,
            "process_time": 0.0,
            "gpu_memory": 0.0,
            "basketball_count": 0,
            "hoops_count": 0,
            "current_video": "ball.avi",
            "is_shot_active": False,
            "current_shot_speed": 0.0,
        }

        self.latest_frame_left = None
        self.latest_frame_right = None
        self.latest_frame_depth = None
        self.latest_frame_disp = None  # 可选：如果你要保留disp流

        self.last_shot = {
            "is_scored": False,
            "shot_speed": 0.0,
            "shot_angle": 0.0,
            "shot_position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "hoop_position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "analysis_time": None,
            "is_active_shot": False,
            "current_shot_sequence": [],
            "predicted_trajectory": [],
            "trajectory_analysis": {},
        }

state = AppState()

def detection_loop():
    cfg = state.cfg
    stereo = build_stereo_params(cfg.width, cfg.height)
    model = load_yolo_model(state.model_path, DEVICE, cfg)
    vision = VisionEngine(model=model, stereo=stereo, cfg=cfg, device=DEVICE)
    shot = ShotEngine(cfg=cfg)

    cap = cv2.VideoCapture(state.video_path)
    if not cap.isOpened():
        with state.lock:
            state.is_running = False
        return

    with state.lock:
        state.stats["total_frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        state.stats["current_video"] = os.path.basename(state.video_path)

    frame_id = 0
    fps_smooth = 0.0

    while True:
        with state.lock:
            if not state.is_running:
                break
            paused = state.is_paused

        if paused:
            time.sleep(0.1)
            continue

        ok, frame = cap.read()
        if not ok:
            with state.lock:
                state.is_running = False
            break

        t0 = time.time()
        frame_id += 1

        left_draw, right_draw, depth_color, fd = vision.process_frame(frame, frame_id)

        # counts
        basketball_count = sum(1 for d in fd.detections if d.cls == "basketball")
        hoops_count = sum(1 for d in fd.detections if d.cls == "hoop")

        # shot update (只有在“丢球超时结束”时才返回结果)
        shot_res = shot.update(fd)

        # 简易实时速度（可选）：用最近两点算导数；为了不耦合Vision，这里放 shot_engine 更合适
        # 这里先不做复杂滤波，你测试稳定后再加

        process_time = time.time() - t0
        fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / max(process_time, 1e-3))

        with state.lock:
            state.stats["current_frame"] = frame_id
            state.stats["fps"] = round(fps_smooth, 2)
            state.stats["process_time"] = round(process_time * 1000, 1)
            state.stats["basketball_count"] = basketball_count
            state.stats["hoops_count"] = hoops_count
            if torch.cuda.is_available():
                state.stats["gpu_memory"] = round(torch.cuda.memory_allocated() / 1024**3, 2)

            state.latest_frame_left = left_draw.copy()
            state.latest_frame_right = right_draw.copy()
            state.latest_frame_depth = depth_color.copy()

            # 是否处于投篮跟踪中
            state.last_shot["is_active_shot"] = shot.state.is_active
            state.last_shot["current_shot_sequence"] = []  # 你如果要前端展示轨迹点，可在这里塞简化后的 points

            if shot_res is not None:
                state.last_shot.update({
                    "is_scored": shot_res.is_scored,
                    "shot_speed": shot_res.shot_speed,
                    "shot_angle": shot_res.shot_angle,
                    "shot_position": {"x": shot_res.shot_position[0], "y": shot_res.shot_position[1], "z": shot_res.shot_position[2]},
                    "hoop_position": {"x": shot_res.hoop_position[0], "y": shot_res.hoop_position[1], "z": shot_res.hoop_position[2]},
                    "analysis_time": datetime.fromtimestamp(shot_res.analysis_time).isoformat(),
                    "sequence_len": shot_res.sequence_len,
                    "predicted_trajectory": shot_res.predicted_trajectory,
                    "actual_trajectory": shot_res.actual_trajectory,
                    "trajectory_analysis": shot_res.trajectory_analysis,
                })

    cap.release()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ---------- API ----------
@app.route("/api/start", methods=["POST"])
def start_detection():
    with state.lock:
        if state.is_running:
            return jsonify({"success": False, "message": "Detection already running"})
        state.is_running = True
        state.is_paused = False
        state.thread = threading.Thread(target=detection_loop, daemon=True)
        state.thread.start()
    return jsonify({"success": True, "message": "Detection started"})

@app.route("/api/pause", methods=["POST"])
def pause_detection():
    with state.lock:
        if not state.is_running:
            return jsonify({"success": False, "message": "Detection not running"})
        state.is_paused = not state.is_paused
        return jsonify({"success": True, "paused": state.is_paused})

@app.route("/api/stop", methods=["POST"])
def stop_detection():
    with state.lock:
        state.is_running = False
        state.is_paused = False
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return jsonify({"success": True, "message": "Detection stopped"})

@app.route("/api/status")
def get_status():
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    cuda_version = torch.version.cuda if torch.cuda.is_available() else "N/A"
    with state.lock:
        return jsonify({
            "success": True,
            "is_running": state.is_running,
            "is_paused": state.is_paused,
            "stats": state.stats,
            "device": device_name,
            "cuda_version": cuda_version,
            "has_results": True
        })

@app.route("/api/shot_analysis")
def get_shot_analysis():
    with state.lock:
        sd = dict(state.last_shot)
    return jsonify({
        "is_scored": sd.get("is_scored", False),
        "shot_speed": sd.get("shot_speed", 0.0),
        "shot_angle": sd.get("shot_angle", 0.0),
        "shot_position": sd.get("shot_position", {"x":0,"y":0,"z":0}),
        "hoop_position": sd.get("hoop_position", {"x":0,"y":0,"z":0}),
        "analysis_time": sd.get("analysis_time"),
        "is_active_shot": sd.get("is_active_shot", False),
        "sequence_length": sd.get("sequence_len", 0),
        "predicted_trajectory": sd.get("predicted_trajectory", []),
        "actual_trajectory": sd.get("actual_trajectory", []),
        "trajectory_analysis": sd.get("trajectory_analysis", {}),
    })

@app.route("/api/shot-analysis")
def get_shot_analysis_dash():
    return get_shot_analysis()

@app.route("/api/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"success": False, "message": "请求中没有视频文件"})
    f = request.files["video"]
    if f.filename == "":
        return jsonify({"success": False, "message": "未选择文件"})
    if not allowed_file(f.filename):
        return jsonify({"success": False, "message": "不支持的文件格式"})
    os.makedirs("./uploads", exist_ok=True)
    fn = secure_filename(f.filename)
    path = os.path.join("./uploads", fn)
    f.save(path)
    with state.lock:
        state.video_path = path
        state.stats["current_video"] = fn
    return jsonify({"success": True, "message": "视频上传成功", "filename": fn, "filepath": path})

def generate_frame(frame_type="left"):
    while True:
        with state.lock:
            if frame_type == "left":
                frame = state.latest_frame_left
            elif frame_type == "right":
                frame = state.latest_frame_right
            elif frame_type == "depth":
                frame = state.latest_frame_depth
            else:
                frame = state.latest_frame_left

        if frame is None:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Waiting for video data...", (120, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            frame = blank

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        time.sleep(0.033)

@app.route("/api/video_feed/<stream_type>")
def video_feed(stream_type):
    return Response(generate_frame(stream_type),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/")
def index():
    file_path = os.path.join(app.static_folder, "index.html")
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return Response(f.read(), mimetype="text/html")
    return jsonify({"error": "Static index.html not found"}), 404

@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("exports", exist_ok=True)
    print("Server: http://127.0.0.1:5002")
    app.run(host="0.0.0.0", port=5002, debug=False, threaded=True)