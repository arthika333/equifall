"""
EquiFall+ Backend — ONNX Runtime (no PyTorch, ~120MB RAM)
Requires: yolov8n-pose.onnx in the same directory.
Generate it once on your laptop:
    pip install ultralytics
    python -c "from ultralytics import YOLO; YOLO('yolov8n-pose.pt').export(format='onnx', imgsz=640)"
Then upload yolov8n-pose.onnx to your Render repo.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import onnxruntime as ort
import logging
from datetime import datetime
import uvicorn
import os
from dotenv import load_dotenv
import tempfile
import shutil
from pathlib import Path
import requests
import threading
import time
import uuid
import json

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("equifall")

app = FastAPI(title="EquiFall+ API (ONNX)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Directories ───────────────────────────────────────────────
UPLOAD_DIR = Path(tempfile.mkdtemp(prefix="equifall_"))
PROCESSED_DIR = UPLOAD_DIR / "processed"
PROCESSED_DIR.mkdir(exist_ok=True)

# ─── ONNX Model ───────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "yolov8n-pose.onnx")
INPUT_SIZE = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# COCO pose keypoint indices
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 13, 14, 15, 16

# Skeleton pairs for drawing
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
    (5, 11), (6, 12), (11, 12),  # torso
    (11, 13), (13, 15), (12, 14), (14, 16)  # legs
]

session = None


def load_model():
    global session
    if not Path(MODEL_PATH).exists():
        logger.error(f"Model file not found: {MODEL_PATH}")
        logger.info("Generate it: python -c \"from ultralytics import YOLO; YOLO('yolov8n-pose.pt').export(format='onnx', imgsz=640)\"")
        return
    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(MODEL_PATH, providers=providers)
    logger.info(f"ONNX model loaded: {MODEL_PATH} (providers: {providers})")


load_model()

# ─── ONNX inference helpers ───────────────────────────────────


def preprocess(frame):
    """Resize + normalize frame for YOLOv8 ONNX input."""
    h, w = frame.shape[:2]
    scale = INPUT_SIZE / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))

    padded = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    padded[:new_h, :new_w] = resized

    blob = padded.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis]  # NCHW
    return blob, scale, (0, 0)  # pad offsets are 0,0 since we pad bottom-right


def nms(boxes, scores, iou_thresh):
    """Simple NMS."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou < iou_thresh]
    return keep


def run_inference(frame):
    """Run ONNX pose model, return list of (bbox, keypoints, score)."""
    if session is None:
        return []

    blob, scale, (pad_x, pad_y) = preprocess(frame)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: blob})

    # Output shape: [1, 56, 8400] → transpose to [8400, 56]
    preds = outputs[0][0].T

    results = []
    for det in preds:
        # det: [cx, cy, w, h, conf, kp0_x, kp0_y, kp0_conf, ...]
        cx, cy, w, h, conf = det[:5]
        if conf < CONF_THRESHOLD:
            continue

        # Convert to xyxy in padded coords
        x1 = (cx - w / 2)
        y1 = (cy - h / 2)
        x2 = (cx + w / 2)
        y2 = (cy + h / 2)

        # Scale back to original image coords
        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale

        # Parse 17 keypoints
        kps_raw = det[5:]  # 51 values: 17 * (x, y, conf)
        keypoints = []
        for k in range(17):
            kx = (kps_raw[k * 3] - pad_x) / scale
            ky = (kps_raw[k * 3 + 1] - pad_y) / scale
            kc = kps_raw[k * 3 + 2]
            keypoints.append((kx, ky, kc))

        results.append({
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "keypoints": keypoints,
            "score": float(conf),
        })

    # NMS
    if results:
        boxes = np.array([r["bbox"] for r in results])
        scores = np.array([r["score"] for r in results])
        keep = nms(boxes, scores, IOU_THRESHOLD)
        results = [results[i] for i in keep]

    return results


# ─── Fall detection logic ─────────────────────────────────────

def is_fall(keypoints):
    """Detect fall from keypoints using body angle heuristics."""

    def kp_valid(idx):
        return keypoints[idx][2] > 0.3

    # Check if shoulders and hips are visible
    if not (kp_valid(L_SHOULDER) or kp_valid(R_SHOULDER)):
        return False
    if not (kp_valid(L_HIP) or kp_valid(R_HIP)):
        return False

    # Average shoulder and hip positions
    shoulders_y = []
    hips_y = []
    shoulders_x = []
    hips_x = []

    for idx in [L_SHOULDER, R_SHOULDER]:
        if kp_valid(idx):
            shoulders_y.append(keypoints[idx][1])
            shoulders_x.append(keypoints[idx][0])
    for idx in [L_HIP, R_HIP]:
        if kp_valid(idx):
            hips_y.append(keypoints[idx][1])
            hips_x.append(keypoints[idx][0])

    avg_shoulder_y = np.mean(shoulders_y)
    avg_hip_y = np.mean(hips_y)
    avg_shoulder_x = np.mean(shoulders_x)
    avg_hip_x = np.mean(hips_x)

    # Torso vector
    dy = avg_hip_y - avg_shoulder_y
    dx = avg_hip_x - avg_shoulder_x
    torso_length = np.sqrt(dx ** 2 + dy ** 2)

    if torso_length < 10:
        return False

    # Angle from vertical (0° = upright, 90° = horizontal)
    angle = np.degrees(np.arctan2(abs(dx), abs(dy)))

    # Fall if torso is nearly horizontal (angle > 50°)
    if angle > 50:
        return True

    # Also check: hips above shoulders (inverted body)
    if avg_hip_y < avg_shoulder_y - 20:
        return True

    return False


# ─── Drawing helpers ──────────────────────────────────────────

def draw_detections(frame, detections, falling):
    """Draw bboxes, skeleton, and fall status on frame."""
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        kps = det["keypoints"]
        is_falling = is_fall(kps)

        color = (0, 0, 255) if is_falling else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = "FALL DETECTED" if is_falling else "OK"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw keypoints
        for kx, ky, kc in kps:
            if kc > 0.3:
                cv2.circle(frame, (int(kx), int(ky)), 3, (255, 255, 0), -1)

        # Draw skeleton
        for i, j in SKELETON:
            if kps[i][2] > 0.3 and kps[j][2] > 0.3:
                pt1 = (int(kps[i][0]), int(kps[i][1]))
                pt2 = (int(kps[j][0]), int(kps[j][1]))
                cv2.line(frame, pt1, pt2, (255, 255, 0), 1)

    return frame


# ─── Telegram alerts ─────────────────────────────────────────

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


def send_telegram(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.info(f"Telegram not configured. Message: {message}")
        return False, "Not configured"
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        resp = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}, timeout=10)
        if resp.ok:
            logger.info("Telegram alert sent")
            return True, None
        return False, resp.text
    except Exception as e:
        return False, str(e)


# ─── Alert state ──────────────────────────────────────────────
alert_cancelled = False

# ─── Job processing ──────────────────────────────────────────
jobs = {}


def process_video_job(job_id, video_path, facility_id):
    """Process video in background thread."""
    global alert_cancelled
    alert_cancelled = False

    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            jobs[job_id] = {"status": "error", "error": "Cannot open video"}
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Output processed video
        out_path = PROCESSED_DIR / f"{job_id}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

        fall_events = []
        in_fall = False
        fall_start_frame = 0
        frame_idx = 0
        process_every = max(1, int(fps / 5))  # ~5 fps analysis

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % process_every == 0:
                detections = run_inference(frame)
                any_fall = any(is_fall(d["keypoints"]) for d in detections)

                if any_fall and not in_fall:
                    in_fall = True
                    fall_start_frame = frame_idx
                elif not any_fall and in_fall:
                    in_fall = False
                    fall_events.append({
                        "event_id": len(fall_events) + 1,
                        "start_frame": fall_start_frame,
                        "end_frame": frame_idx,
                        "start_time": f"{fall_start_frame / fps:.1f}s",
                        "end_time": f"{frame_idx / fps:.1f}s",
                        "duration": round((frame_idx - fall_start_frame) / fps, 2),
                    })

                frame = draw_detections(frame, detections, any_fall)

            out.write(frame)
            frame_idx += 1

            # Update progress
            progress = int(frame_idx / max(total_frames, 1) * 100)
            jobs[job_id]["progress"] = min(progress, 99)

        # Close fall if still in one
        if in_fall:
            fall_events.append({
                "event_id": len(fall_events) + 1,
                "start_frame": fall_start_frame,
                "end_frame": frame_idx,
                "start_time": f"{fall_start_frame / fps:.1f}s",
                "end_time": f"{frame_idx / fps:.1f}s",
                "duration": round((frame_idx - fall_start_frame) / fps, 2),
            })

        cap.release()
        out.release()

        # Build result
        video_id = job_id
        alert_result = None
        if fall_events and not alert_cancelled:
            msg = (
                f"🚨 <b>EquiFall+ Alert</b>\n"
                f"📍 Location: {facility_id or 'Unknown'}\n"
                f"⚠️ Falls detected: {len(fall_events)}\n"
                f"🕐 Time: {datetime.now().strftime('%H:%M:%S')}"
            )
            tg_ok, _ = send_telegram(msg)
            alert_result = {
                "status": "sent",
                "total_alerts": 1 if tg_ok else 0,
                "contacted": ["Telegram"] if tg_ok else [],
                "fall_events": len(fall_events),
            }

        result = {
            "video_filename": jobs[job_id].get("filename", "video.mp4"),
            "video_id": video_id,
            "stream_url": f"/api/video/stream/{video_id}",
            "analysis": {
                "total_falls": len(fall_events),
                "falls_detected": fall_events,
                "status": "falls_detected" if fall_events else "no_falls",
            },
            "alert_result": alert_result,
            "message": f"Analysis complete. {len(fall_events)} fall(s) detected.",
        }

        jobs[job_id]["status"] = "complete"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["result"] = result
        logger.info(f"Job {job_id} complete: {len(fall_events)} falls")

    except Exception as e:
        logger.exception(f"Job {job_id} failed")
        jobs[job_id] = {"status": "error", "error": str(e)}


# ─── Incident history (in-memory) ────────────────────────────
incidents = []

# ─── Notification log (in-memory) ────────────────────────────
notifications_log = []

# ─── Routes ───────────────────────────────────────────────────


@app.get("/")
def root():
    return {"status": "EquiFall+ ONNX backend running", "model": MODEL_PATH}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": session is not None}


@app.post("/api/cctv/analyze-video")
async def analyze_video(file: UploadFile = File(...), facility_id: str = None):
    if session is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Upload yolov8n-pose.onnx.")

    job_id = str(uuid.uuid4())[:8]
    save_path = UPLOAD_DIR / f"{job_id}_{file.filename}"

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    jobs[job_id] = {"status": "processing", "progress": 0, "filename": file.filename}

    thread = threading.Thread(target=process_video_job, args=(job_id, save_path, facility_id), daemon=True)
    thread.start()

    return {"job_id": job_id}


@app.get("/api/job/{job_id}")
def get_job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/video/stream/{video_id}")
def stream_video(video_id: str):
    path = PROCESSED_DIR / f"{video_id}.mp4"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Video not found")

    def iterfile():
        with open(path, "rb") as f:
            while chunk := f.read(1024 * 64):
                yield chunk

    return StreamingResponse(iterfile(), media_type="video/mp4")


@app.get("/api/notifications/sent")
def get_notifications():
    return {
        "total_notifications": len(notifications_log),
        "notifications": notifications_log,
        "summary": {
            "sms": sum(1 for n in notifications_log if n["type"] == "SMS"),
            "email": sum(1 for n in notifications_log if n["type"] == "EMAIL"),
            "telegram": sum(1 for n in notifications_log if n["type"] == "TELEGRAM"),
        },
    }


@app.get("/api/incidents")
def get_incidents():
    return incidents


@app.post("/api/telegram/test")
def test_telegram():
    ok, error = send_telegram("✅ EquiFall+ test message — Telegram alerts are working!")
    return {"success": ok, "error": error}


@app.post("/api/alerts/cancel")
def cancel_alert():
    global alert_cancelled
    alert_cancelled = True
    logger.info("Alert cancellation requested")
    return {"success": True, "message": "Alert cancelled"}


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print(f"\n{'='*50}")
    print(f"  EquiFall+ ONNX Backend")
    print(f"  Port: {port}")
    print(f"  Model: {MODEL_PATH}")
    print(f"  RAM usage: ~120MB (vs ~500MB with PyTorch)")
    print(f"{'='*50}\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
#end of file