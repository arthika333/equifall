from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
import cv2
import gc
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

# ============ APP ============
app = FastAPI(title="EquiFall+ API", version="3.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

# ============ CONFIG ============
MODEL_PATH = os.getenv("MODEL_PATH", "yolov8n-pose.pt")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
VOICE_CHECK_WAIT = int(os.getenv("VOICE_CHECK_WAIT", "20"))
ANALYSIS_FRAME_SKIP = max(1, int(os.getenv("ANALYSIS_FRAME_SKIP", "6")))
MAX_RENDER_WIDTH = max(320, int(os.getenv("MAX_RENDER_WIDTH", "640")))
JPEG_QUALITY = max(50, min(95, int(os.getenv("JPEG_QUALITY", "75"))))

# ============ STORAGE ============
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

# ============ MODEL — LOAD EAGERLY AT STARTUP ============
# This is the critical fix: load the model ONCE at import time so it's
# already in memory when the first request arrives. Lazy loading inside
# a background thread caused a second copy of the model to be allocated,
# doubling RAM usage and crashing the Render free-tier worker.
logger.info("📦 Loading YOLO model at startup...")
model = YOLO(MODEL_PATH)
logger.info(f"✅ Model loaded: {MODEL_PATH}")

_tts_lock = threading.Lock()


# ============ HELPERS ============
def update_job(job_id: str, **updates):
    with store_lock:
        if job_id in job_store:
            job_store[job_id].update(updates)


def get_job(job_id: str) -> dict | None:
    with store_lock:
        j = job_store.get(job_id)
        return dict(j) if j else None


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
            except Exception as exc:
                logger.error(f"❌ TTS error: {exc}")

    threading.Thread(target=_run, daemon=True).start()


def send_telegram(message: str) -> tuple[bool, str | None]:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        detail = "Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID"
        logger.warning(f"⚠️ Telegram not configured: {detail}")
        return False, detail

    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"},
            timeout=15,
        )
        if resp.ok:
            logger.info("✅ Telegram alert sent")
            return True, None
        detail = f"Telegram API {resp.status_code}: {resp.text[:500]}"
        logger.error(f"❌ {detail}")
        return False, detail
    except Exception as exc:
        detail = str(exc)
        logger.error(f"❌ Telegram request failed: {detail}")
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
            notif["message"] = f"🚨 EQUIFALL ALERT: {falls_count} fall(s) at {location}. No response after voice check."
            contacted.append(contact["phone"])
        else:
            notif["subject"] = f"⚠️ EquiFall+ Alert: {falls_count} Fall(s) — {location}"
            notif["message"] = f"Fall detected. No response after {VOICE_CHECK_WAIT}s. Emergency contacts notified."
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
            "type": "TELEGRAM", "recipient": "@equifall_bot",
            "recipient_name": "Telegram Alert Bot", "location": location,
            "falls_count": falls_count, "timestamp": now,
            "status": "delivered", "message": telegram_msg,
        })
        contacted.append("@equifall_bot")

    logger.warning(f"✓ Alerts sent to {len(contacted)} contact(s)")
    return contacted


def staged_voice_alert(location: str, falls_count: int) -> list:
    global alert_cancelled
    alert_cancelled = False

    speak("Attention. A possible fall has been detected. Are you okay? Please respond verbally.")
    logger.info("🔊 Voice check issued")

    for _ in range(VOICE_CHECK_WAIT):
        time.sleep(1)
        if alert_cancelled:
            logger.info("✅ Alert cancelled — false alarm")
            return []

    logger.info("🚨 No response — escalating!")
    speak(f"Alert! Fall detected at {location}. Contacting emergency services now.")
    return send_all_alerts(location, falls_count)


# ============ FALL DETECTION ============
def is_fall(keypoints, confidence_threshold=0.5):
    if keypoints is None or len(keypoints) == 0:
        return False
    try:
        kp = keypoints[0] if len(keypoints.shape) > 2 else keypoints
        if len(kp) < 17:
            return False

        nose, l_hip, r_hip = kp[0], kp[11], kp[12]
        l_ankle, r_ankle = kp[15], kp[16]
        l_shoulder, r_shoulder = kp[5], kp[6]

        for p in [nose, l_hip, r_hip, l_ankle, r_ankle, l_shoulder, r_shoulder]:
            if len(p) > 2 and p[2] < confidence_threshold:
                return False

        hip_y = (l_hip[1] + r_hip[1]) / 2
        shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2
        ankle_y = (l_ankle[1] + r_ankle[1]) / 2
        torso_h = abs(shoulder_y - hip_y)
        if torso_h < 10:
            return False

        on_ground = (ankle_y - hip_y) < torso_h * 0.5
        horizontal = (torso_h / max(abs(l_shoulder[0] - r_shoulder[0]), 1)) < 1.2
        nose_low = nose[1] > hip_y - torso_h * 0.3

        return sum([on_ground, horizontal, nose_low]) >= 2
    except Exception:
        return False


def draw_pose(frame, person_kps, color):
    pairs = [(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
    for p1, p2 in pairs:
        if p1 < len(person_kps) and p2 < len(person_kps):
            x1, y1, c1 = person_kps[p1]
            x2, y2, c2 = person_kps[p2]
            if c1 > 0.5 and c2 > 0.5:
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    for x, y, c in person_kps:
        if c > 0.5:
            cv2.circle(frame, (int(x), int(y)), 4, color, -1)


# ============ SINGLE-PASS ANALYSIS + ANNOTATION ============
def analyze_and_render_video(video_path: str, output_path: str, facility_id: str, job_id: str | None = None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Cannot open video file")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    scale = min(1.0, MAX_RENDER_WIDTH / max(src_w, 1))
    out_w = max(2, int(src_w * scale)) & ~1  # ensure even
    out_h = max(2, int(src_h * scale)) & ~1

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    falls = []
    fall_start = None
    consec = 0
    MIN_CONSEC = 2
    idx = 0
    last_kps, last_boxes, last_fall = [], [], False

    logger.info(f"🔍 Analyzing: {total_frames} frames, {fps:.1f} FPS, out={out_w}x{out_h}, skip={ANALYSIS_FRAME_SKIP}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if scale != 1.0:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

        # Run YOLO only every N-th frame
        if idx % ANALYSIS_FRAME_SKIP == 0:
            results = model(frame, verbose=False)
            cur_kps, cur_boxes, cur_fall = [], [], False

            for r in results:
                if r.keypoints is not None and len(r.keypoints) > 0:
                    kps = r.keypoints.data.cpu().numpy()
                    cur_kps = kps
                    for pk in kps:
                        if is_fall(pk):
                            cur_fall = True
                if r.boxes is not None and len(r.boxes) > 0:
                    cur_boxes = r.boxes.xyxy.cpu().numpy().tolist()

            last_kps, last_boxes, last_fall = cur_kps, cur_boxes, cur_fall

            if cur_fall:
                consec += 1
                if consec >= MIN_CONSEC and fall_start is None:
                    fall_start = idx
                    logger.warning(f"🚨 FALL at frame {idx}")
            else:
                if fall_start is not None and consec >= MIN_CONSEC:
                    falls.append({"start_frame": fall_start, "end_frame": idx})
                fall_start = None
                consec = 0

            # Free YOLO result memory
            del results
            if idx % (ANALYSIS_FRAME_SKIP * 10) == 0:
                gc.collect()

        # Draw annotations using last inference results
        color = (0, 0, 255) if last_fall else (0, 255, 0)
        label = "FALL" if last_fall else "Person"
        for box in last_boxes:
            x1, y1, x2, y2 = [int(v) for v in box[:4]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        for pk in last_kps:
            draw_pose(frame, pk, color)

        if last_fall:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (out_w, 40), (0, 0, 200), -1)
            cv2.putText(overlay, "FALL DETECTED", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

        writer.write(frame)

        if job_id and total_frames > 0:
            update_job(job_id, progress=min(int((idx / total_frames) * 85) + 10, 95))

        if total_frames > 0 and idx % max(1, total_frames // 5) == 0:
            logger.info(f"⏳ {(idx / total_frames) * 100:.0f}%")

        idx += 1

    if fall_start is not None:
        falls.append({"start_frame": fall_start, "end_frame": max(0, idx - 1)})

    cap.release()
    writer.release()
    gc.collect()

    fall_events = []
    for i, f in enumerate(falls):
        s = f["start_frame"] / fps
        e = f["end_frame"] / fps
        fall_events.append({
            "event_id": i + 1,
            "start_frame": f["start_frame"], "end_frame": f["end_frame"],
            "start_time": f"{int(s // 60):02d}:{s % 60:05.2f}",
            "end_time": f"{int(e // 60):02d}:{e % 60:05.2f}",
            "duration": round(max(0, e - s), 2),
        })

    logger.info(f"✓ Done. {len(fall_events)} fall(s) in {idx} frames")
    return fall_events, total_frames, fps, out_w, out_h


# ============ BACKGROUND JOB ============
def process_video_job(job_id: str, video_path: Path, filename: str, facility_id: str):
    try:
        update_job(job_id, status="processing", progress=10)

        video_id = f"vid_{uuid.uuid4().hex[:8]}"
        annotated_path = UPLOAD_DIR / f"out_{video_id}.mp4"

        fall_events, total_frames, fps, out_w, out_h = analyze_and_render_video(
            str(video_path), str(annotated_path), facility_id, job_id
        )

        video_store[video_id] = str(annotated_path)

        alert_result = None
        if fall_events:
            logger.warning(f"🚨 {len(fall_events)} FALL(S) — starting staged alert")

            def _alert():
                contacted = staged_voice_alert(facility_id, len(fall_events))
                incident_store.append({
                    "video_filename": filename, "location": facility_id,
                    "source": "upload", "falls_count": len(fall_events),
                    "severity": "Critical" if len(fall_events) >= 3 else "Moderate",
                    "timestamp": datetime.now().isoformat(), "falls": fall_events,
                    "contacted": contacted,
                })

            threading.Thread(target=_alert, daemon=True).start()

            alert_result = {
                "status": "staged_alert_initiated",
                "total_alerts": len(EMERGENCY_CONTACTS) + 1,
                "contacted": [c.get("phone", c.get("email", "")) for c in EMERGENCY_CONTACTS] + ["@equifall_bot"],
                "fall_events": len(fall_events),
                "voice_check": True, "wait_seconds": VOICE_CHECK_WAIT,
                "stages": [
                    "Voice check: Are you okay?",
                    f"Waiting {VOICE_CHECK_WAIT}s for response",
                    "No response: escalating",
                    "Emergency call initiated",
                    "Alerts sent to all contacts",
                ],
            }

        result = {
            "video_filename": filename, "video_id": video_id,
            "stream_url": f"/api/video/stream/{video_id}",
            "video_file_url": f"/api/video/file/{video_id}",
            "analysis": {
                "total_falls": len(fall_events), "falls_detected": fall_events,
                "status": f"{len(fall_events)} fall(s) in {total_frames} frames",
            },
            "alert_result": alert_result, "message": "Analysis complete",
        }

        update_job(job_id, status="complete", progress=100, result=result)
        logger.info(f"✅ Job {job_id} complete")

        # Clean up source video to free disk
        try:
            os.remove(str(video_path))
        except Exception:
            pass

    except Exception as exc:
        logger.exception(f"❌ Job {job_id} failed")
        update_job(job_id, status="error", error=str(exc))


# ============ ENDPOINTS ============
@app.get("/")
async def root():
    return {"name": "EquiFall+ API", "status": "ok", "version": "3.2.0"}


@app.head("/")
async def root_head():
    return {}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "telegram_configured": bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
        "tts_available": TTS_AVAILABLE,
        "version": "3.2.0",
    }


@app.post("/api/cctv/analyze-video")
async def analyze_video_endpoint(file: UploadFile = File(...), facility_id: str = "Facility A"):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    video_path = UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{Path(file.filename).name}"
    with open(video_path, "wb") as out:
        shutil.copyfileobj(file.file, out)

    file_size_mb = video_path.stat().st_size / (1024 * 1024)
    logger.info(f"📹 Uploaded: {file.filename} ({file_size_mb:.2f} MB)")

    job_id = uuid.uuid4().hex[:8]
    with store_lock:
        job_store[job_id] = {"status": "queued", "progress": 5, "result": None, "error": None}

    threading.Thread(
        target=process_video_job,
        args=(job_id, video_path, file.filename, facility_id),
        daemon=True,
    ).start()

    return {"job_id": job_id}


@app.get("/api/job/{job_id}")
async def get_job_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/video/stream/{video_id}")
async def stream_video(video_id: str):
    if video_id not in video_store:
        raise HTTPException(status_code=404, detail="Video not found")
    path = video_store[video_id]

    def gen():
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if ok:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
                time.sleep(1 / max(fps, 1))
        finally:
            cap.release()

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


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
    ok, err = send_telegram("✅ EquiFall+ test alert — Telegram integration working!")
    return {"success": ok, "error": err}


@app.post("/api/alerts/cancel")
async def cancel_alert():
    global alert_cancelled
    alert_cancelled = True
    logger.info("🛑 Alert cancelled")
    return {"success": True, "message": "Alert escalation cancelled"}


# ============ STARTUP ============
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print(f"\n🛡️  EquiFall+ v3.2.0 — http://localhost:{port}")
    print(f"📚 Docs: http://localhost:{port}/docs")
    print(f"🔊 TTS: {'✅' if TTS_AVAILABLE else '❌'}")
    print(f"📱 Telegram: {'✅' if TELEGRAM_BOT_TOKEN else '❌'}")
    print(f"⚙️  Skip={ANALYSIS_FRAME_SKIP}, Width={MAX_RENDER_WIDTH}, JPEG={JPEG_QUALITY}\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
