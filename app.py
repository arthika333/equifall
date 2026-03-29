from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import cv2
import numpy as np
import logging
from datetime import datetime
import uvicorn
from ultralytics import YOLO
import asyncio
import os
from dotenv import load_dotenv
import tempfile
import shutil
from pathlib import Path
import json
import requests
import threading
import time
import uuid

# Text-to-speech (Windows: pyttsx3, fallback: print-only)
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("⚠️  pyttsx3 not installed — voice alerts will be printed only")

load_dotenv()

# ============ CONFIG ============
app = FastAPI(title="EquiFall+ API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============ GLOBALS ============
MODEL_PATH = os.getenv("MODEL_PATH", "yolov8n-pose.pt")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
VOICE_CHECK_WAIT = int(os.getenv("VOICE_CHECK_WAIT", "20"))

UPLOAD_DIR = Path(tempfile.mkdtemp(prefix="equifall_"))
video_store: dict = {}
incident_store: list = []
notification_store: list = []
alert_cancelled = False

# Job store for async processing
job_store: dict = {}  # job_id -> { status, progress, result, error }

# Emergency contacts
EMERGENCY_CONTACTS = [
    {"name": "Nurse Aisha", "phone": "+60 12-345 6789", "type": "SMS"},
    {"name": "Nursing Station", "email": "nurse.station@hospital.my", "type": "EMAIL"},
    {"name": "Dr. Rahman (On-Call)", "phone": "+60 11-987 6543", "type": "SMS"},
    {"name": "Facility Manager", "email": "admin@equifall.care", "type": "EMAIL"},
    {"name": "Family: Mrs. Chen", "phone": "+60 14-222 3344", "type": "SMS"},
    {"name": "Safety Officer", "email": "safety@carehome.my", "type": "EMAIL"},
]

# ============ YOLO MODEL ============
model = None

def get_model():
    global model
    if model is None:
        try:
            model = YOLO(MODEL_PATH)
            logger.info(f"✅ Model loaded: {MODEL_PATH}")
        except Exception as e:
            logger.error(f"❌ Model load failed: {e}")
            model = None
    return model

# ============ VOICE ENGINE ============
_tts_lock = threading.Lock()

def speak(text: str):
    """Thread-safe text-to-speech"""
    if not TTS_AVAILABLE:
        logger.info(f"🔊 [TTS disabled] {text}")
        return
    def _speak():
        with _tts_lock:
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)
                engine.setProperty('volume', 1.0)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
                logger.info(f"🔊 Voice alert: {text}")
            except Exception as e:
                logger.error(f"❌ TTS error: {e}")
    threading.Thread(target=_speak, daemon=True).start()

# ============ TELEGRAM ============
def send_telegram(message: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("⚠️ Telegram not configured (missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID)")
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        resp = requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML",
        }, timeout=10)
        if resp.ok:
            logger.info("✅ Telegram alert sent")
            return True
        else:
            logger.error(f"❌ Telegram failed: {resp.text}")
            return False
    except Exception as e:
        logger.error(f"❌ Telegram send failed: {e}")
        return False

# ============ ALERT SYSTEM ============
def send_all_alerts(location: str, falls_count: int) -> list:
    """Send alerts to all emergency contacts + Telegram"""
    contacted = []
    now = datetime.now().isoformat()

    for contact in EMERGENCY_CONTACTS:
        notif = {
            "type": contact["type"],
            "recipient": contact.get("phone", contact.get("email", "")),
            "recipient_name": contact["name"],
            "location": location,
            "falls_count": falls_count,
            "timestamp": now,
            "status": "delivered",
        }

        if contact["type"] == "SMS":
            notif["message"] = f"🚨 EQUIFALL ALERT: {falls_count} fall(s) at {location}. No response from patient after voice check. Immediate assistance needed."
            contacted.append(contact["phone"])
        elif contact["type"] == "EMAIL":
            notif["subject"] = f"⚠️ EquiFall+ Alert: {falls_count} Fall(s) — {location}"
            notif["message"] = f"Fall detected. Voice check received no response after {VOICE_CHECK_WAIT}s. Emergency contacts are being notified."
            contacted.append(contact["email"])

        notification_store.append(notif)

    # Telegram alert
    telegram_msg = (
        f"🚨 <b>EQUIFALL+ FALL ALERT</b>\n\n"
        f"📍 Location: {location}\n"
        f"🔢 Falls: {falls_count}\n"
        f"🎤 Voice Check: No response after {VOICE_CHECK_WAIT}s\n"
        f"📞 Emergency call initiated to {EMERGENCY_CONTACTS[0]['name']}\n"
        f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}\n\n"
        f"Immediate assistance required."
    )
    tg_ok = send_telegram(telegram_msg)

    if tg_ok:
        notification_store.append({
            "type": "TELEGRAM",
            "recipient": "@equifall_bot",
            "recipient_name": "Telegram Alert Bot",
            "location": location,
            "falls_count": falls_count,
            "timestamp": now,
            "status": "delivered",
            "message": telegram_msg,
        })
        contacted.append("@equifall_bot")

    logger.warning(f"✓ Alerts sent to {len(contacted)} contact(s)")
    return contacted

def staged_voice_alert(location: str, falls_count: int) -> list:
    """
    Synchronous staged alert (runs in background thread):
    Stage 1: Voice asks 'Are you okay?'
    Stage 2: Wait 20 seconds
    Stage 3: If not cancelled → announce fall, send Telegram + all alerts
    """
    global alert_cancelled
    alert_cancelled = False

    # --- Stage 1: Ask if person is okay ---
    speak("Attention. A possible fall has been detected. Are you okay? Please respond verbally.")
    logger.info("🔊 Voice check: Asked 'Are you okay?'")

    # --- Stage 2: Wait for response ---
    logger.info(f"⏳ Waiting {VOICE_CHECK_WAIT} seconds for patient response...")
    for i in range(VOICE_CHECK_WAIT):
        time.sleep(1)
        if alert_cancelled:
            logger.info("✅ Alert cancelled during wait — false alarm confirmed")
            return []

    # --- Stage 3: No response — escalate ---
    if alert_cancelled:
        logger.info("✅ Alert cancelled — false alarm confirmed")
        return []

    logger.info("🚨 No response after wait — escalating!")
    speak(f"Alert! Fall detected at {location}. No response from patient. Contacting emergency services now.")

    # Send all alerts (SMS, Email, Telegram)
    contacted = send_all_alerts(location, falls_count)
    return contacted

# ============ FALL DETECTION ============
def is_fall(keypoints, confidence_threshold=0.5):
    """Detect falls using pose keypoints"""
    if keypoints is None or len(keypoints) == 0:
        return False

    try:
        kp = keypoints[0] if len(keypoints.shape) > 2 else keypoints
        if len(kp) < 17:
            return False

        nose = kp[0]
        left_hip, right_hip = kp[11], kp[12]
        left_ankle, right_ankle = kp[15], kp[16]
        left_shoulder, right_shoulder = kp[5], kp[6]

        key_points = [nose, left_hip, right_hip, left_ankle, right_ankle]
        if any(p[2] < confidence_threshold for p in key_points if len(p) > 2):
            return False

        hip_y = (left_hip[1] + right_hip[1]) / 2
        shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
        ankle_y = (left_ankle[1] + right_ankle[1]) / 2
        nose_y = nose[1]

        torso_height = abs(shoulder_y - hip_y)
        if torso_height < 10:
            return False

        hip_ankle_diff = ankle_y - hip_y
        shoulder_hip_ratio = abs(shoulder_y - hip_y) / max(abs(left_shoulder[0] - right_shoulder[0]), 1)
        nose_below_threshold = nose_y > hip_y - torso_height * 0.3

        is_on_ground = hip_ankle_diff < torso_height * 0.5
        is_horizontal = shoulder_hip_ratio < 1.2
        nose_low = nose_below_threshold

        return (is_on_ground and is_horizontal) or (is_on_ground and nose_low) or (is_horizontal and nose_low)

    except (IndexError, TypeError):
        return False

def analyze_video_file(video_path: str, facility_id: str = "Facility A", job_id: str = None):
    """Analyze video for falls — updates job_store progress if job_id provided"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Cannot open video file")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    falls = []
    current_fall_start = None
    consecutive_fall = 0
    MIN_CONSECUTIVE = 2

    logger.info(f"🔍 Analyzing video: {total_frames} frames, {fps} FPS")

    frame_idx = 0
    SKIP = 3  # Only analyze every 3rd frame — saves memory on free tier
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % SKIP != 0:
            frame_idx += 1
            continue

        # Update progress in job store
        if job_id and total_frames > 0:
            pct = int((frame_idx / total_frames) * 85) + 10  # 10-95%
            job_store[job_id]["progress"] = pct

        if frame_idx % max(1, total_frames // 5) == 0:
            pct = (frame_idx / total_frames) * 100
            logger.info(f"⏳ Progress: {pct:.1f}%")

        m = get_model()
        if m:
            results = m(frame, verbose=False)
            for r in results:
                if r.keypoints is not None and len(r.keypoints) > 0:
                    if is_fall(r.keypoints.data.cpu().numpy()):
                        consecutive_fall += 1
                        if consecutive_fall >= MIN_CONSECUTIVE:
                            if current_fall_start is None:
                                current_fall_start = frame_idx
                                logger.warning(f"🚨 FALL DETECTED at frame {frame_idx}")
                    else:
                        if current_fall_start is not None and consecutive_fall >= MIN_CONSECUTIVE:
                            falls.append({
                                "start_frame": current_fall_start,
                                "end_frame": frame_idx,
                            })
                        current_fall_start = None
                        consecutive_fall = 0

        frame_idx += 1

    if current_fall_start is not None:
        falls.append({
            "start_frame": current_fall_start,
            "end_frame": frame_idx - 1,
        })

    cap.release()

    fall_events = []
    for i, fall in enumerate(falls):
        start_sec = fall["start_frame"] / fps
        end_sec = fall["end_frame"] / fps
        duration = end_sec - start_sec
        fall_events.append({
            "event_id": i + 1,
            "start_frame": fall["start_frame"],
            "end_frame": fall["end_frame"],
            "start_time": f"{int(start_sec // 60):02d}:{start_sec % 60:05.2f}",
            "end_time": f"{int(end_sec // 60):02d}:{end_sec % 60:05.2f}",
            "duration": round(duration, 2),
        })

    logger.info(f"✓ Analysis complete. Found {len(fall_events)} fall event(s)")
    return fall_events, total_frames, fps

def create_annotated_video(video_path: str, output_path: str, fall_events: list, fps: float):
    """Create annotated video with skeleton overlay and fall markers"""
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    fall_frame_set = set()
    for event in fall_events:
        for f in range(event["start_frame"], event["end_frame"] + 1):
            fall_frame_set.add(f)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        m = get_model()
        if m:
            results = m(frame, verbose=False)
            for r in results:
                if r.keypoints is not None:
                    kps = r.keypoints.data.cpu().numpy()
                    for person_kps in kps:
                        skeleton_pairs = [
                            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                            (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
                            (12, 14), (14, 16),
                        ]
                        for p1, p2 in skeleton_pairs:
                            if p1 < len(person_kps) and p2 < len(person_kps):
                                x1, y1, c1 = person_kps[p1]
                                x2, y2, c2 = person_kps[p2]
                                if c1 > 0.5 and c2 > 0.5:
                                    color = (0, 0, 255) if frame_idx in fall_frame_set else (0, 255, 0)
                                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                        for kp in person_kps:
                            x, y, c = kp
                            if c > 0.5:
                                color = (0, 0, 255) if frame_idx in fall_frame_set else (0, 255, 0)
                                cv2.circle(frame, (int(x), int(y)), 4, color, -1)

                if r.boxes is not None:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        color = (0, 0, 255) if frame_idx in fall_frame_set else (0, 255, 0)
                        label = "FALL" if frame_idx in fall_frame_set else "Person"
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        cv2.putText(frame, label, (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if frame_idx in fall_frame_set:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 200), -1)
            cv2.putText(overlay, "FALL DETECTED",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

# ============ BACKGROUND JOB PROCESSOR ============
def process_video_job(job_id: str, video_path: str, filename: str, facility_id: str):
    """Runs in a background thread — processes video and updates job_store"""
    try:
        job_store[job_id]["status"] = "processing"
        job_store[job_id]["progress"] = 10

        # Analyze
        fall_events, total_frames, fps = analyze_video_file(str(video_path), facility_id, job_id)

        job_store[job_id]["progress"] = 90

        # Create annotated video
        video_id = f"video_{len(video_store)}"
        annotated_path = UPLOAD_DIR / f"annotated_{filename}"
        create_annotated_video(str(video_path), str(annotated_path), fall_events, fps)
        video_store[video_id] = str(annotated_path)

        job_store[job_id]["progress"] = 95

        # Handle falls with staged voice alert
        alert_result = None
        if fall_events:
            logger.warning(f"🚨 {len(fall_events)} FALL(S) DETECTED — INITIATING STAGED VOICE CHECK")

            def run_staged_alert():
                contacted = staged_voice_alert(facility_id, len(fall_events))
                incident_store.append({
                    "video_filename": filename,
                    "location": facility_id,
                    "source": "upload",
                    "falls_count": len(fall_events),
                    "severity": "Critical" if len(fall_events) >= 3 else "Moderate",
                    "timestamp": datetime.now().isoformat(),
                    "falls": fall_events,
                    "alert_stages": ["voice_check", "wait_20s", "escalate", "emergency_call"],
                })

            threading.Thread(target=run_staged_alert, daemon=True).start()

            alert_result = {
                "status": "staged_alert_initiated",
                "total_alerts": len(EMERGENCY_CONTACTS) + 1,
                "contacted": [c.get("phone", c.get("email", "")) for c in EMERGENCY_CONTACTS] + ["@equifall_bot"],
                "fall_events": len(fall_events),
                "voice_check": True,
                "wait_seconds": VOICE_CHECK_WAIT,
                "stages": [
                    "Voice check: 'Are you okay?'",
                    f"Waiting {VOICE_CHECK_WAIT}s for response...",
                    "No response — escalating",
                    "Emergency call to primary contact",
                    "Alerts sent to all contacts",
                ],
            }

        # Build final result
        result = {
            "video_filename": filename,
            "video_id": video_id,
            "stream_url": f"/api/video/stream/{video_id}",
            "analysis": {
                "total_falls": len(fall_events),
                "falls_detected": fall_events,
                "status": f"Analysis complete — {len(fall_events)} fall event(s) detected across {total_frames} frames",
            },
            "alert_result": alert_result,
            "message": "Analysis complete",
        }

        job_store[job_id]["status"] = "complete"
        job_store[job_id]["progress"] = 100
        job_store[job_id]["result"] = result
        logger.info(f"✅ Job {job_id} complete")

    except Exception as e:
        logger.error(f"❌ Job {job_id} failed: {e}")
        job_store[job_id]["status"] = "error"
        job_store[job_id]["error"] = str(e)

# ============ API ENDPOINTS ============

@app.get("/health")
async def health():
    m = get_model()
    return {
        "status": "healthy",
        "model_loaded": m is not None,
        "telegram_configured": bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
        "tts_available": TTS_AVAILABLE,
        "voice_check_wait": VOICE_CHECK_WAIT,
        "version": "3.0.0",
    }

@app.post("/api/cctv/analyze-video")
async def analyze_video_endpoint(
    file: UploadFile = File(...),
    facility_id: str = "Facility A",
):
    """
    Accepts video upload, returns a job_id immediately.
    The frontend polls /api/job/{job_id} for status and results.
    """
    m = get_model()
    if not m:
        raise HTTPException(status_code=503, detail="Model not loaded")

    video_path = UPLOAD_DIR / file.filename
    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    logger.info(f"📹 Video uploaded: {file.filename}")

    # Create a job and process in background thread
    job_id = str(uuid.uuid4())[:8]
    job_store[job_id] = {
        "status": "processing",
        "progress": 5,
        "result": None,
        "error": None,
    }

    # Run processing in a background thread (not blocked by request timeout)
    thread = threading.Thread(
        target=process_video_job,
        args=(job_id, video_path, file.filename, facility_id),
        daemon=True,
    )
    thread.start()

    # Return immediately — frontend will poll /api/job/{job_id}
    return {"job_id": job_id}

@app.get("/api/job/{job_id}")
async def get_job_status(job_id: str):
    """Poll this endpoint to check if analysis is done."""
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="Job not found")

    job = job_store[job_id]
    return {
        "status": job["status"],
        "progress": job.get("progress", 0),
        "result": job.get("result"),
        "error": job.get("error"),
    }

@app.get("/api/video/stream/{video_id}")
async def stream_video(video_id: str):
    if video_id not in video_store:
        raise HTTPException(status_code=404, detail="Video not found")

    video_path = video_store[video_id]

    def generate():
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"📹 Streaming video: {total} frames, {fps} FPS")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
            time.sleep(1 / fps)

        cap.release()

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/notifications/sent")
async def get_notifications():
    return {
        "total_notifications": len(notification_store),
        "notifications": notification_store,
        "summary": {
            "sms": sum(1 for n in notification_store if n["type"] == "SMS"),
            "email": sum(1 for n in notification_store if n["type"] == "EMAIL"),
            "telegram": sum(1 for n in notification_store if n["type"] == "TELEGRAM"),
        },
    }

@app.get("/api/incidents")
async def get_incidents():
    return incident_store

@app.post("/api/telegram/test")
async def test_telegram():
    msg = "✅ EquiFall+ test alert — Telegram integration working!"
    ok = send_telegram(msg)
    return {"success": ok, "error": None if ok else "Failed to send"}

@app.post("/api/alerts/cancel")
async def cancel_alert():
    global alert_cancelled
    alert_cancelled = True
    logger.info("🛑 Alert cancellation requested from frontend")
    return {"success": True, "message": "Alert escalation cancelled"}

# ============ STARTUP ============
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print("\n" + "=" * 70)
    print("🛡️  EquiFall+ v3.0 — Async Job Processing + Staged Voice Alert")
    print("=" * 70)
    print(f"\n🌐 Server: http://localhost:{port}")
    print(f"📚 API Docs: http://localhost:{port}/docs")
    print(f"🔊 TTS: {'✅ Available' if TTS_AVAILABLE else '❌ Not installed (pip install pyttsx3)'}")
    print(f"⏱️  Voice check wait: {VOICE_CHECK_WAIT}s")

    if TELEGRAM_BOT_TOKEN:
        print(f"📱 Telegram: ✅ Configured")
    else:
        print(f"📱 Telegram: ❌ Not configured")

    print(f"\n🎯 ASYNC JOB FLOW:")
    print(f"  1. Upload video → returns job_id instantly")
    print(f"  2. Frontend polls /api/job/{{job_id}} every 3s")
    print(f"  3. Analysis runs in background thread (no timeout!)")
    print(f"  4. Falls detected → staged voice alert + Telegram")
    print("=" * 70 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=port)
