# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Summary

A Python desktop application that opens a USB webcam and runs three real-time detections:
1. **Face recognition** (InsightFace `buffalo_l`) — recognize name + face landmarks
2. **Person + pose skeleton** (MediaPipe) — detect people + body keypoints
3. **License plate** (fast-alpr + EasyOCR fallback) — read and normalize plate text

Primary output = **`cv2.imshow` window**. Face enrollment is done via a local FastAPI web page that shares `data/faces.db` with the desktop app.

---

## How to Run

```bash
# Install dependencies (once)
pip install -r requirements.txt
# EasyOCR installs opencv-python-headless which breaks the GUI window — fix it:
pip uninstall opencv-python-headless -y && pip install "opencv-python==4.10.0.84"

# Desktop app (live camera)
python -m app.main              # press 'q' to quit

# CLI face enrollment
python -m app.enroll --photo /path/photo.jpg --name "Full Name"
python -m app.enroll --list
python -m app.enroll --delete 3

# Web face enrollment (must run alongside app.main for camera handoff to work)
uvicorn enroll_web.server:app --reload --port 8000
```

---

## `.env` Configuration

Copy `.env.example` to `.env`. All keys:

```env
CAMERA_INDEX=0              # USB webcam index
CAMERA_WIDTH=1280           # 0 = use camera default
CAMERA_HEIGHT=720
DEVICE=directml             # directml (Windows GPU, no CUDA Toolkit) | cuda | cpu
FACE_THRESHOLD=0.5          # cosine similarity threshold (0.0–1.0)
FACE_DET_SIZE=320           # InsightFace det_size: 320 (fast) or 640 (accurate)
POSE_MODEL_COMPLEXITY=1     # MediaPipe complexity: 0 (lite) | 1 | 2 (accurate)
PLATE_CONFIDENCE_THRESHOLD=0.4      # OCR confidence gate
PLATE_DETECTOR_CONF_THRESHOLD=0.3   # YOLO detection gate (lower → more plates detected at edges)
PLATE_PERSIST_SECONDS=1.2           # how long a plate stays visible when OCR misses a frame
ENABLE_FACE=true
ENABLE_POSE=true
ENABLE_PLATE=true
```

`DEVICE=directml` is the recommended default on Windows — it uses DirectX 12 GPU without requiring the CUDA Toolkit. Use `cuda` only if CUDA 12.x + cuDNN are installed.

---

## Architecture

### Threading model

Three parallel threads keep the display loop non-blocking:

| Class | File | Role |
|---|---|---|
| `CameraReader` | `app/camera.py` | Captures frames in a background thread; `read()` always returns the latest frame without blocking. |
| `InferenceThread` | `app/inference_thread.py` | Wraps a `Detector` in a background thread. `submit(frame)` is non-blocking and drops older unprocessed frames. `get_results()` returns the latest inference output. |
| Main loop | `app/main.py` | Reads camera frame → submits to all `InferenceThread`s → draws latest results → `imshow`. Never waits on inference. |

Each detector runs at its own natural speed. The display runs at camera FPS. The HUD shows both camera FPS and per-detector AI FPS.

### Detector interface

All detectors implement `app/detectors/base.py`:

```python
class Detector(ABC):
    def process(self, frame: np.ndarray) -> list:
        """Return list of annotation dataclasses."""
```

Annotation types (`base.py`): `FaceAnnotation`, `PoseAnnotation`, `PlateAnnotation`.

**Separation rule**: detectors only return data; all `cv2.draw*` calls live in `app/overlay.py`.

### Execution provider fallback

`FaceDetector` (`app/detectors/face.py`) tries providers in order: configured → CPU. For `DEVICE=cuda` it also checks whether `onnxruntime_providers_cuda.dll` can actually be loaded before attempting inference, and falls back to CPU with a helpful message if not.

`PlateDetector` (`app/detectors/plate.py`) uses fast-alpr with the same provider chain. If fast-alpr fails to load entirely, it falls back to EasyOCR. The plate detector also runs a `_PlateStabilizer` — IoU-based temporal tracking that holds the last known box for `PLATE_PERSIST_SECONDS` to eliminate frame-to-frame flickering.

### Camera handoff (desktop ↔ web enrollment)

`app.main` and `enroll_web.server` share one physical webcam. Coordination via a flag file:

1. Web server writes `data/camera_paused.flag` when a guided capture session starts.
2. `app.main` detects the flag in its loop, releases `CameraReader`, and polls until the flag is gone.
3. Web server deletes the flag (via `/enroll/camera/release` or session expiry) — `app.main` reacquires the camera.

### Web enrollment flows

Two flows in `enroll_web/server.py`:

- **Photo upload** (`/enroll` POST): upload one or more photos, extract embedding from the largest detected face per photo, save to `data/faces.db`.
- **Guided webcam capture** (`/webcam` page): a 5-pose guided session (front → right → left → smile → front). The browser uses `getUserMedia`. Each frame is sent to `/enroll/frame` as base64 JSON; the server validates head orientation via 5-point keypoint yaw estimation before accepting the capture. Sessions expire after 5 minutes.

### Face database

`app/face_db.py` stores 512-dim float32 embeddings as blobs in SQLite. Matching is cosine similarity against all stored embeddings (linear scan). The desktop auto-reloads the DB every 5 seconds if `data/faces.db` mtime changed.

Public API: `enroll(name, embedding)`, `match(embedding) -> (name, score)`, `list_all()`, `list_grouped()`, `delete(id)`, `delete_by_name(name)`.

---

## Key Constraints

- **Never call `cv2.draw*` inside `detectors/`** — drawing goes in `overlay.py` only.
- **Never hardcode `cv2.VideoCapture(0)`** — always use `config.CAMERA_INDEX`.
- **No web live camera feed** — live feed is `cv2.imshow` only; the web server handles enrollment only.
- **No database other than local SQLite** unless explicitly requested.
- Do not commit `data/` or `models/` directories.
