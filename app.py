from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
import cv2
import logging
from datetime import datetime
import uvicorn
from ultralytics import YOLO
import os
from dotenv import load_dotenv
import tempfile
import shutil
from pathlib import Path
import requests
import threading
import time
import uuid

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("⚠️  pyttsx3 not installed — voice alerts will be printed only")

load_dotenv()

app = FastAPI(title="EquiFall+ API", version="3.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

MODEL_PATH = os.getenv("MODEL_PATH", "yolov8n-pose.pt")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
VOICE_CHECK_WAIT = int(os.getenv("VOICE_CHECK_WAIT", "20"))
ANALYSIS_FRAME_SKIP = max(1, int(os.getenv("ANALYSIS_FRAME_SKIP", "4")))
MAX_RENDER_WIDTH = max(320, int(os.getenv("MAX_RENDER_WIDTH", "960")))
JPEG_QUALITY = max(50, min(95, int(os.getenv("JPEG_QUALITY", "82"))))

UPLOAD_DIR = Path(tempfile.mkdtemp(prefix="equifall_"))
video_store: dict[str, str] = {}
incident_store: list = []
notification_store: list = []
job_store: dict[str, dict] = {}
alert_cancelled = False
store_lock = threading.Lock()

EMERGENCY_CONTACTS = [
    {"name": "Nurse Aisha", "phone": "+60 12-345 6789", "type": "SMS"},
    {"name": "Nursing Station", "email": "nurse.station@hospital.my", "type": "EMAIL"},
    {"name": "Dr. Rahman (On-Call)", "phone": "+60 11-987 6543", "type": "SMS"},
    {"name": "Facility Manager", "email": "admin@equifall.care", "type": "EMAIL"},
    {"name": "Family: Mrs. Chen", "phone": "+60 14-222 3344", "type": "SMS"},
    {"name": "Safety Officer", "email": "safety@carehome.my", "type": "EMAIL"},
]

model = None
_model_lock = threading.Lock()
_tts_lock = threading.Lock()


def update_job(job_id: str, **updates):
    with store_lock:
        if job_id in job_store:
            job_store[job_id].update(updates)


def get_job(job_id: str):
    with store_lock:
        return job_store.get(job_id)


def get_model():
    global model
    if model is None:
        with _model_lock:
            if model is None:
                logger.info("📦 Loading YOLO model...")
                model = YOLO(MODEL_PATH)
                logger.info(f"✅ Model loaded: {MODEL_PATH}")
    return model


def speak(text: str):
    if not TTS_AVAILABLE:
        logger.info(f"🔊 [TTS disabled] {text}")
        return

    def _run():
        with _tts_lock:
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", 150)
                engine.setProperty("volume", 1.0)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
                logger.info(f"🔊 Voice alert: {text}")
            except Exception as exc:
                logger.error(f"❌ TTS error: {exc}")

    threading.Thread(target=_run, daemon=True).start()


def send_telegram(message: str) -> tuple[bool, str | None]:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        detail = "Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID"
        logger.warning(f"⚠️ Telegram not configured: {detail}")
        return False, detail

    try:
        response = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "HTML",
            },
            timeout=15,
        )
        if response.ok:
            logger.info("✅ Telegram alert sent")
            return True, None
        detail = f"Telegram API {response.status_code}: {response.text[:500]}"
        logger.error(f"❌ {detail}")
        return False, detail
    except Exception as exc:
        detail = f"Telegram request failed: {exc}"
        logger.error(f"❌ {detail}")
        return False, detail


def send_all_alerts(location: str, falls_count: int) -> list:
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
            notif["message"] = (
                f"🚨 EQUIFALL ALERT: {falls_count} fall(s) at {location}. "
                "No response from patient after voice check. Immediate assistance needed."
            )
            contacted.append(contact["phone"])
        else:
            notif["subject"] = f"⚠️ EquiFall+ Alert: {falls_count} Fall(s) — {location}"
            notif["message"] = (
                f"Fall detected. Voice check received no response after {VOICE_CHECK_WAIT}s. "
                "Emergency contacts are being notified."
            )
            contacted.append(contact["email"])

        notification_store.append(notif)

    telegram_msg = (
        f"🚨 <b>EQUIFALL+ FALL ALERT</b>\n\n"
        f"📍 Location: {location}\n"
        f"🔢 Falls: {falls_count}\n"
        f"🎤 Voice Check: No response after {VOICE_CHECK_WAIT}s\n"
        f"📞 Emergency call initiated to {EMERGENCY_CONTACTS[0]['name']}\n"
        f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}\n\n"
        "Immediate assistance required."
    )
    tg_ok, _ = send_telegram(telegram_msg)

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
    global alert_cancelled
    alert_cancelled = False

    speak("Attention. A possible fall has been detected. Are you okay? Please respond verbally.")
    logger.info("🔊 Voice check: Asked 'Are you okay?'")

    logger.info(f"⏳ Waiting {VOICE_CHECK_WAIT} seconds for patient response...")
    for _ in range(VOICE_CHECK_WAIT):
        time.sleep(1)
        if alert_cancelled:
            logger.info("✅ Alert cancelled during wait — false alarm confirmed")
            return []

    if alert_cancelled:
        logger.info("✅ Alert cancelled — false alarm confirmed")
        return []

    logger.info("🚨 No response after wait — escalating!")
    speak(f"Alert! Fall detected at {location}. No response from patient. Contacting emergency services now.")
    return send_all_alerts(location, falls_count)


def is_fall(keypoints, confidence_threshold=0.5):
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

        required = [nose, left_hip, right_hip, left_ankle, right_ankle, left_shoulder, right_shoulder]
        if any(len(p) > 2 and p[2] < confidence_threshold for p in required):
            return False

        hip_y = (left_hip[1] + right_hip[1]) / 2
        shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
        ankle_y = (left_ankle[1] + right_ankle[1]) / 2
        nose_y = nose[1]

        torso_height = abs(shoulder_y - hip_y)
        if torso_height < 10:
            return False

        hip_ankle_diff = ankle_y - hip_y
        shoulder_width = max(abs(left_shoulder[0] - right_shoulder[0]), 1)
        shoulder_hip_ratio = abs(shoulder_y - hip_y) / shoulder_width
        nose_low = nose_y > hip_y - torso_height * 0.3
        is_on_ground = hip_ankle_diff < torso_height * 0.5
        is_horizontal = shoulder_hip_ratio < 1.2

        return (is_on_ground and is_horizontal) or (is_on_ground and nose_low) or (is_horizontal and nose_low)
    except Exception:
        return False


def draw_pose(frame, person_kps, color):
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
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    for x, y, c in person_kps:
        if c > 0.5:
            cv2.circle(frame, (int(x), int(y)), 4, color, -1)


def analyze_and_render_video(video_path: str, output_path: str, facility_id: str, job_id: str | None = None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Cannot open video file")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

    scale = min(1.0, MAX_RENDER_WIDTH / max(src_w, 1))
    out_w = max(2, int(src_w * scale))
    out_h = max(2, int(src_h * scale))
    out_w -= out_w % 2
    out_h -= out_h % 2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    falls = []
    current_fall_start = None
    consecutive_fall = 0
    MIN_CONSECUTIVE = 2
    frame_idx = 0
    last_keypoints = []
    last_boxes = []
    last_is_fall = False

    logger.info(
        f"🔍 Analyzing video: {total_frames} frames, {fps:.2f} FPS, "
        f"render={out_w}x{out_h}, skip={ANALYSIS_FRAME_SKIP}"
    )

    yolo = get_model()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if scale != 1.0:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

        run_inference = frame_idx % ANALYSIS_FRAME_SKIP == 0
        if run_inference:
            results = yolo(frame, verbose=False)
            current_keypoints = []
            current_boxes = []
            current_frame_fall = False

            for result in results:
                if result.keypoints is not None and len(result.keypoints) > 0:
                    kps = result.keypoints.data.cpu().numpy()
                    current_keypoints = kps
                    for person_kps in kps:
                        if is_fall(person_kps):
                            current_frame_fall = True

                if result.boxes is not None and len(result.boxes) > 0:
                    current_boxes = result.boxes.xyxy.cpu().numpy().tolist()

            last_keypoints = current_keypoints
            last_boxes = current_boxes
            last_is_fall = current_frame_fall

            if current_frame_fall:
                consecutive_fall += 1
                if consecutive_fall >= MIN_CONSECUTIVE and current_fall_start is None:
                    current_fall_start = frame_idx
                    logger.warning(f"🚨 FALL DETECTED at frame {frame_idx}")
            else:
                if current_fall_start is not None and consecutive_fall >= MIN_CONSECUTIVE:
                    falls.append({"start_frame": current_fall_start, "end_frame": frame_idx})
                current_fall_start = None
                consecutive_fall = 0

        box_color = (0, 0, 255) if last_is_fall else (0, 255, 0)
        label = "FALL" if last_is_fall else "Person"

        for box in last_boxes:
            x1, y1, x2, y2 = [int(v) for v in box[:4]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

        for person_kps in last_keypoints:
            draw_pose(frame, person_kps, box_color)

        if last_is_fall:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (out_w, 44), (0, 0, 200), -1)
            cv2.putText(overlay, "FALL DETECTED", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

        writer.write(frame)

        if job_id and total_frames > 0:
            pct = int((frame_idx / total_frames) * 85) + 10
            update_job(job_id, progress=min(pct, 95))

        if frame_idx % max(1, total_frames // 5) == 0:
            logger.info(f"⏳ Progress: {(frame_idx / max(total_frames, 1)) * 100:.1f}%")

        frame_idx += 1

    if current_fall_start is not None:
        falls.append({"start_frame": current_fall_start, "end_frame": max(0, frame_idx - 1)})

    cap.release()
    writer.release()

    fall_events = []
    for i, fall in enumerate(falls):
        start_sec = fall["start_frame"] / fps
        end_sec = fall["end_frame"] / fps
        duration = max(0.0, end_sec - start_sec)
        fall_events.append({
            "event_id": i + 1,
            "start_frame": fall["start_frame"],
            "end_frame": fall["end_frame"],
            "start_time": f"{int(start_sec // 60):02d}:{start_sec % 60:05.2f}",
            "end_time": f"{int(end_sec // 60):02d}:{end_sec % 60:05.2f}",
            "duration": round(duration, 2),
        })

    logger.info(f"✓ Analysis complete. Found {len(fall_events)} fall event(s)")
    return fall_events, total_frames, fps, out_w, out_h


def process_video_job(job_id: str, video_path: Path, filename: str, facility_id: str):
    try:
        update_job(job_id, status="processing", progress=10)

        video_id = f"video_{uuid.uuid4().hex[:10]}"
        annotated_path = UPLOAD_DIR / f"annotated_{video_id}.mp4"

        fall_events, total_frames, fps, out_w, out_h = analyze_and_render_video(
            str(video_path), str(annotated_path), facility_id, job_id
        )

        video_store[video_id] = str(annotated_path)
        update_job(job_id, progress=96)

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
                    "contacted": contacted,
                    "alert_stages": ["voice_check", "wait", "escalate", "dispatch"],
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
                    "Voice check: Are you okay?",
                    f"Waiting {VOICE_CHECK_WAIT}s for response",
                    "No response: escalating",
                    "Emergency call initiated",
                    "Alerts sent to all contacts",
                ],
            }

        result = {
            "video_filename": filename,
            "video_id": video_id,
            "stream_url": f"/api/video/stream/{video_id}",
            "video_file_url": f"/api/video/file/{video_id}",
            "analysis": {
                "total_falls": len(fall_events),
                "falls_detected": fall_events,
                "status": f"Analysis complete — {len(fall_events)} fall event(s) detected across {total_frames} frames",
                "output_resolution": f"{out_w}x{out_h}",
                "frame_skip": ANALYSIS_FRAME_SKIP,
            },
            "alert_result": alert_result,
            "message": "Analysis complete",
        }

        update_job(job_id, status="complete", progress=100, result=result)
        logger.info(f"✅ Job {job_id} complete")
    except Exception as exc:
        logger.exception(f"❌ Job {job_id} failed: {exc}")
        update_job(job_id, status="error", error=str(exc))


@app.get("/")
async def root():
    return {"name": "EquiFall+ API", "status": "ok", "version": "3.1.0"}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "telegram_configured": bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
        "tts_available": TTS_AVAILABLE,
        "voice_check_wait": VOICE_CHECK_WAIT,
        "analysis_frame_skip": ANALYSIS_FRAME_SKIP,
        "max_render_width": MAX_RENDER_WIDTH,
        "version": "3.1.0",
    }


@app.post("/api/cctv/analyze-video")
async def analyze_video_endpoint(file: UploadFile = File(...), facility_id: str = "Facility A"):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    video_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{Path(file.filename).name}"
    with open(video_path, "wb") as out:
        shutil.copyfileobj(file.file, out)

    logger.info(f"📹 Video uploaded: {file.filename}")

    job_id = uuid.uuid4().hex[:8]
    with store_lock:
        job_store[job_id] = {
            "status": "queued",
            "progress": 5,
            "result": None,
            "error": None,
        }

    thread = threading.Thread(
        target=process_video_job,
        args=(job_id, video_path, file.filename, facility_id),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id}


@app.get("/api/job/{job_id}")
async def get_job_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "status": job.get("status", "queued"),
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
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        logger.info(f"📹 Streaming video {video_id} at {fps:.2f} FPS")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if not ok:
                    continue
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                time.sleep(1 / max(fps, 1.0))
        finally:
            cap.release()

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/video/file/{video_id}")
async def video_file(video_id: str):
    if video_id not in video_store:
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(video_store[video_id], media_type="video/mp4", filename=f"{video_id}.mp4")


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
    ok, error = send_telegram("✅ EquiFall+ test alert — Telegram integration working!")
    return {"success": ok, "error": error}


@app.post("/api/alerts/cancel")
async def cancel_alert():
    global alert_cancelled
    alert_cancelled = True
    logger.info("🛑 Alert cancellation requested from frontend")
    return {"success": True, "message": "Alert escalation cancelled"}


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
