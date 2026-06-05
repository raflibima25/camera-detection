import base64
import glob
import os
import sys
import time
import uuid
import warnings
from pathlib import Path
from typing import List
from urllib.parse import quote

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


def _setup_cuda() -> None:
    candidates: list[str] = []
    for var in ("CUDA_PATH", "CUDA_PATH_V12_8"):
        val = os.environ.get(var, "")
        if val:
            candidates.append(os.path.join(val, "bin"))
    cuda_base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    for v12 in sorted(glob.glob(os.path.join(cuda_base, "v12*")), reverse=True):
        candidates.append(os.path.join(v12, "bin"))
    for cuda_bin in candidates:
        if os.path.isfile(os.path.join(cuda_bin, "cudart64_12.dll")):
            try:
                os.add_dll_directory(cuda_bin)
            except (AttributeError, OSError):
                pass
            if cuda_bin.lower() not in os.environ.get("PATH", "").lower():
                os.environ["PATH"] = cuda_bin + os.pathsep + os.environ.get("PATH", "")
            break


_setup_cuda()

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")

from app import config
from app.face_db import FaceDB

app = FastAPI(title="Face Enrollment — Camera Detection")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# --- Guided capture configuration ---
# "type" is used by _check_pose() to validate head orientation:
#   front  → face straight toward camera
#   right  → turn right (yaw_ratio > +threshold, frame mirrored selfie-style)
#   left   → turn left  (yaw_ratio < -threshold)
#   smile  → face straight toward camera (smile is not validated geometrically)
POSES = [
    {"label": "Look forward",      "hint": "Face the camera directly",              "icon": "⊙", "type": "front"},
    {"label": "Look to the right", "hint": "Slowly turn your head to the right ~30°", "icon": "→", "type": "right"},
    {"label": "Look to the left",  "hint": "Slowly turn your head to the left ~30°",  "icon": "←", "type": "left"},
    {"label": "Smile",             "hint": "Face forward with a natural smile",      "icon": "◡", "type": "smile"},
    {"label": "Look forward",      "hint": "Return to center with a neutral expression", "icon": "⊙", "type": "front"},
]
_MIN_DET_SCORE     = 0.60   # minimum face detection confidence
_MIN_FACE_RATIO    = 0.08   # face area must be >= 8% of frame area
_CAPTURE_COOLDOWN  = 1.8    # seconds between captures
_SESSION_TTL       = 300    # session expires after 5 minutes of inactivity

# Threshold yaw: frame dikirim browser sudah di-mirror (selfie view).
# Dalam frame mirror: yaw > 0 → hidung ke kanan gambar → orang toleh ke kanan (THEIR right).
_YAW_FRONT_MAX  = 0.10   # |yaw| < ini untuk pose depan/senyum
_YAW_SIDE_MIN   = 0.13   # |yaw| > ini untuk pose samping


def _check_pose(face, pose_type: str) -> tuple[bool, str]:
    """
    Estimate yaw from 5-point keypoints and validate whether it matches the requested pose.
    Returns (passed, failure_message).
    """
    kps = face.kps  # shape [5, 2]: [left_eye, right_eye, nose, left_mouth, right_mouth] (image coords)
    left_eye_x  = float(kps[0][0])
    right_eye_x = float(kps[1][0])
    nose_x      = float(kps[2][0])

    eye_width = right_eye_x - left_eye_x
    if eye_width < 8:
        return False, "Face not clearly readable, try moving closer"

    # Normalized horizontal nose offset from the eye midpoint
    # Mirrored frame: yaw > 0 → turn right; yaw < 0 → turn left
    yaw = (nose_x - (left_eye_x + right_eye_x) / 2.0) / eye_width

    if pose_type in ("front", "smile"):
        if abs(yaw) > _YAW_FRONT_MAX:
            direction = "right" if yaw > 0 else "left"
            return False, f"Face the camera directly (currently turned to the {direction})"
        return True, ""

    if pose_type == "right":
        if yaw < _YAW_SIDE_MIN:
            if abs(yaw) < _YAW_FRONT_MAX:
                return False, "Turn your head further to the right (~30°)"
            return False, "That looks like left — turn to the right"
        return True, ""

    if pose_type == "left":
        if yaw > -_YAW_SIDE_MIN:
            if abs(yaw) < _YAW_FRONT_MAX:
                return False, "Turn your head further to the left (~30°)"
            return False, "That looks like right — turn to the left"
        return True, ""

    return True, ""  # unknown pose type → pass

# Flag file: written when a web enrollment session is active → app.main releases the camera
_FLAG = _ROOT / "data" / "camera_paused.flag"

# --- Shared state ---
_db: FaceDB | None = None
_face_app = None
_sessions: dict[str, dict] = {}


def _get_db() -> FaceDB:
    global _db
    if _db is None:
        _db = FaceDB()
    return _db


def _get_face_app():
    global _face_app
    if _face_app is None:
        from insightface.app import FaceAnalysis
        fa = FaceAnalysis(
            name="buffalo_l",
            root=str(_ROOT / "models"),
            providers=config.get_ort_providers(),
            allowed_modules=["detection", "recognition"],
        )
        fa.prepare(ctx_id=0, det_size=(config.FACE_DET_SIZE, config.FACE_DET_SIZE))
        _face_app = fa
        print("[Web] InsightFace loaded")
    return _face_app


def _clean_sessions() -> None:
    """Remove expired sessions."""
    now = time.time()
    expired = [sid for sid, s in _sessions.items() if now - s["ts"] > _SESSION_TTL]
    for sid in expired:
        del _sessions[sid]
    if not _sessions:
        _FLAG.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────
#  Lifecycle
# ─────────────────────────────────────────────────────────────

@app.on_event("startup")
async def _startup() -> None:
    _get_db()
    _FLAG.unlink(missing_ok=True)  # clean up stale flag from previous session
    print("[Web] Server ready — open http://localhost:8000")


# ─────────────────────────────────────────────────────────────
#  Halaman utama (upload foto)
# ─────────────────────────────────────────────────────────────

@app.get("/")
async def index(request: Request, msg: str = "", err: str = ""):
    faces = _get_db().list_grouped()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "faces": faces,   # list of (name, count, first_enrolled)
        "msg": msg,
        "err": err,
    })


@app.post("/enroll")
async def enroll(
    name: str = Form(...),
    photo: List[UploadFile] = File(...),
):
    name = name.strip()
    if not name:
        return RedirectResponse("/?err=Name+cannot+be+empty", status_code=303)

    db = _get_db()
    ok_files: list[str] = []
    err_files: list[str] = []

    for up in photo:
        data = await up.read()
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        fname = up.filename or "foto"
        if img is None:
            err_files.append(f"{fname} (failed to read)")
            continue
        try:
            detected = _get_face_app().get(img)
        except Exception as e:
            err_files.append(f"{fname} ({str(e)[:40]})")
            continue
        if not detected:
            err_files.append(f"{fname} (no face detected)")
            continue
        if len(detected) > 1:
            detected.sort(
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                reverse=True,
            )
        db.enroll(name, detected[0].embedding)
        ok_files.append(fname)

    if ok_files:
        msg = f"{name}: {len(ok_files)} embeddings saved from {len(photo)} photo(s)"
        if err_files:
            msg += f" — failed: {', '.join(err_files)}"
        return RedirectResponse(f"/?msg={quote(msg)}", status_code=303)

    err = f"All photos failed to process: {'; '.join(err_files)}"
    return RedirectResponse(f"/?err={quote(err)}", status_code=303)


@app.post("/delete/by-name")
async def delete_by_name(name: str = Form(...)):
    count = _get_db().delete_by_name(name)
    return RedirectResponse(f"/?msg={quote(f'All {count} embeddings for {name} deleted')}", status_code=303)


@app.post("/delete/{face_id}")
async def delete_face(face_id: int):
    _get_db().delete(face_id)
    return RedirectResponse("/", status_code=303)


# ─────────────────────────────────────────────────────────────
#  Enrollment via camera (guided multi-pose)
# ─────────────────────────────────────────────────────────────

@app.get("/desktop/frame")
async def desktop_frame():
    """Serve the latest frame from app.main for preview in web enrollment."""
    path = _ROOT / "data" / "latest_frame.jpg"
    if not path.exists():
        return JSONResponse({"error": "Desktop app is not running or no frame available yet"}, status_code=404)
    # Baca ke memori dulu agar Content-Length konsisten meski file dioverwrite saat streaming
    data = path.read_bytes()
    from fastapi.responses import Response
    return Response(content=data, media_type="image/jpeg")


@app.get("/webcam")
async def webcam_page(request: Request):
    return templates.TemplateResponse("webcam.html", {
        "request": request,
        "total_poses": len(POSES),
        "poses": POSES,
    })


@app.post("/enroll/session/start")
async def session_start(name: str = Form(...)):
    name = name.strip()
    if not name:
        return JSONResponse({"error": "Name cannot be empty"}, status_code=400)
    _clean_sessions()
    sid = uuid.uuid4().hex[:10]
    _sessions[sid] = {
        "name": name,
        "embeddings": [],
        "pose_index": 0,
        "last_capture": 0.0,
        "ts": time.time(),
    }
    # Signal app.main to release the camera so the browser can use getUserMedia
    _FLAG.parent.mkdir(exist_ok=True)
    _FLAG.touch()
    return {
        "session_id": sid,
        "name": name,
        "poses": POSES,
        "total": len(POSES),
    }


@app.post("/enroll/camera/release")
async def camera_release(request: Request):
    """Called by browser after stream stops → app.main may reacquire the camera."""
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    sid = body.get("session_id", "") if isinstance(body, dict) else ""
    if sid and sid in _sessions:
        del _sessions[sid]
    if not _sessions:
        _FLAG.unlink(missing_ok=True)
    return {"ok": True}


@app.post("/enroll/frame")
async def enroll_frame(request: Request):
    body = await request.json()
    sid = body.get("session_id", "")
    frame_b64: str = body.get("frame", "")

    session = _sessions.get(sid)
    if not session:
        return JSONResponse({"status": "error", "reason": "Session is invalid or expired"})

    # Cooldown — give the user time to change pose
    now = time.time()
    session["ts"] = now
    if now - session["last_capture"] < _CAPTURE_COOLDOWN:
        return JSONResponse({"status": "cooldown"})

    # Read frame: from desktop app file or from browser base64
    use_desktop = body.get("use_desktop_frame", False)
    if use_desktop:
        frame_path = _ROOT / "data" / "latest_frame.jpg"
        if not frame_path.exists():
            return JSONResponse({"status": "error", "reason": "Desktop app is not running"})
        img = cv2.imread(str(frame_path))
    else:
        try:
            raw = base64.b64decode(frame_b64.split(",")[-1])
            arr = np.frombuffer(raw, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            return JSONResponse({"status": "error", "reason": "Invalid frame"})

    if img is None:
        return JSONResponse({"status": "error", "reason": "Could not read frame"})

    # Face detection
    try:
        faces = _get_face_app().get(img)
    except Exception as e:
        return JSONResponse({"status": "error", "reason": str(e)[:80]})

    if len(faces) == 0:
        return JSONResponse({"status": "no_face"})
    if len(faces) > 1:
        return JSONResponse({"status": "multiple_faces", "count": len(faces)})

    face = faces[0]

    # Quality gate
    if face.det_score < _MIN_DET_SCORE:
        return JSONResponse({
            "status": "low_quality",
            "reason": f"Confidence too low ({face.det_score:.2f})",
        })

    h, w = img.shape[:2]
    x1, y1, x2, y2 = face.bbox.astype(int)
    face_area_ratio = ((x2 - x1) * (y2 - y1)) / (h * w)
    if face_area_ratio < _MIN_FACE_RATIO:
        return JSONResponse({"status": "too_far"})

    # Validate head orientation matches the requested pose
    pose_type = POSES[session["pose_index"]].get("type", "front")
    pose_ok, pose_reason = _check_pose(face, pose_type)
    if not pose_ok:
        return JSONResponse({"status": "wrong_pose", "reason": pose_reason})

    # Simpan embedding ke session
    session["embeddings"].append(face.embedding.copy())
    session["pose_index"] += 1
    session["last_capture"] = now
    pose_index = session["pose_index"]
    total = len(POSES)

    if pose_index >= total:
        # All poses complete → flush to DB
        name = session["name"]
        db = _get_db()
        for emb in session["embeddings"]:
            db.enroll(name, emb)
        del _sessions[sid]
        return JSONResponse({
            "status": "complete",
            "name": name,
            "total_embeddings": total,
        })

    return JSONResponse({
        "status": "captured",
        "poses_done": pose_index,
        "total": total,
        "next_pose": POSES[pose_index],
    })
