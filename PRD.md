# PRD — Desktop Camera Detection System

**Version:** 1.1  
**Date:** 2026-06-05  
**Status:** Approved — All phases complete

---

## 1. Product Summary

A Python desktop application that opens a USB webcam and runs three real-time detection pipelines:

1. **Face Recognition** — identify faces and display the name of enrolled persons
2. **Person Detection + Pose Skeleton** — detect the primary person and draw body skeleton on the camera feed
3. **License Plate Recognition** — detect and read vehicle license plate text

The primary output is a **desktop window (`cv2.imshow`)**. Each detector runs in its own background thread so the display loop never blocks on inference. Face enrollment is done through a **browser web page** that shares a local SQLite database with the desktop application.

---

## 2. Goals

| Goal | Success Indicator |
|---|---|
| Real-time detection on camera | Display runs at camera FPS (≥15) with all detectors active; inference FPS shown in HUD |
| Skeleton/landmark visualization | Body skeleton lines + face landmarks rendered smoothly in the window |
| Face identity recognition | Enrolled person's name and similarity score appear above their face |
| License plate reading | Plate text is read, normalized, and shown in an overlay box without flickering |
| Easy face enrollment | Upload photo in browser OR use guided webcam capture → recognized immediately without restart |

---

## 3. Users

- **Owner/operator** — runs the desktop application, enrolls faces via browser, monitors live feed.

---

## 4. Functional Requirements

### 4.1 Desktop Application

| ID | Requirement |
|---|---|
| F-01 | Open USB webcam (configurable index and resolution) and display real-time feed in a window |
| F-02 | Window closes when the `q` key is pressed |
| F-03 | Each detector can be independently enabled/disabled via `.env` config |
| F-04 | Camera capture and each inference pipeline run in separate background threads; the display loop never blocks on inference |
| F-05 | Auto-reload face data when `faces.db` is updated (checked every 5 seconds via mtime, without restart) |
| F-06 | HUD overlay shows: camera FPS (top-left), per-detector AI FPS, active detector names at the bottom, and person count (top-right when pose is active) |
| F-07 | When web enrollment requests camera access, the desktop releases the camera and reacquires it automatically when enrollment is complete |

### 4.2 Face Recognition

| ID | Requirement |
|---|---|
| F-10 | Detect all faces in every frame using InsightFace `buffalo_l` |
| F-11 | Draw face bounding box (green = known, red = unknown) and 5-point facial landmarks with connection lines |
| F-12 | Match face embedding against database; display name and cosine similarity score above the face box |
| F-13 | Unrecognized faces (similarity below threshold) are labeled "Unknown" |
| F-14 | Similarity threshold and face detection input size are configurable via `.env` |
| F-15 | Enroll new faces via CLI: `python -m app.enroll --photo photo.jpg --name "Name"` |
| F-16 | Face inference runs on a downscaled copy (max 640 px wide) to reduce GPU load; bounding boxes are scaled back to original resolution for display |

### 4.3 Person Detection + Pose Skeleton

| ID | Requirement |
|---|---|
| F-20 | Detect the primary person in the frame using MediaPipe Pose |
| F-21 | Draw body skeleton: joint connection lines (33 landmarks, full-body connections) with visibility gating (landmarks below 0.5 visibility are not drawn) |
| F-22 | Display person count (0 or 1) in the top-right corner of the window |
| F-23 | Model complexity is configurable (0 = lite/fast, 1 = default, 2 = accurate) |

### 4.4 License Plate Recognition

| ID | Requirement |
|---|---|
| F-30 | Detect the license plate area using fast-alpr (YOLO-based detector); fall back to EasyOCR if fast-alpr fails to load |
| F-31 | Draw a bounding box around the detected plate and display OCR text + confidence below the box |
| F-32 | Normalize plate text: strip non-alphanumeric characters, uppercase |
| F-33 | Apply temporal stabilization: track each plate by IoU across frames; keep the last known box and highest-confidence text visible for a configurable duration (`PLATE_PERSIST_SECONDS`) to eliminate per-frame flickering |
| F-34 | OCR confidence threshold and YOLO detector confidence threshold are independently configurable |

### 4.5 Web Face Enrollment

| ID | Requirement |
|---|---|
| F-40 | Browser page with a form to upload name + one or more face photos (batch) |
| F-41 | Compute face embedding (InsightFace `buffalo_l`) for each uploaded photo; if multiple faces are detected in one photo, use the largest face |
| F-42 | Save embedding + name to `data/faces.db`; desktop recognizes the face immediately on next DB reload |
| F-43 | Display enrolled faces grouped by name (name, embedding count, enrollment date) with a per-name delete button |
| F-44 | Guided webcam enrollment flow (`/webcam` page): capture 5 poses in sequence (front → right → left → smile → front) using the browser's `getUserMedia` camera |
| F-45 | For each webcam capture, validate head orientation from 5-point facial keypoints (yaw estimation) before accepting; reject with user-facing guidance if orientation does not match the requested pose |
| F-46 | Webcam enrollment requires exactly one face per frame; reject frames with zero or multiple faces |
| F-47 | Webcam enrollment sessions expire after 5 minutes of inactivity |
| F-48 | During guided webcam enrollment the desktop app releases its camera handle so the browser can acquire the same device; the desktop reacquires automatically after the session ends |
| F-49 | When the desktop app is running alongside the web server, the web enrollment page can optionally use the desktop's current frame (`/desktop/frame`) as a live preview |
| F-50 | Delete all embeddings for a person by name (bulk delete) |

---

## 5. Non-Functional Requirements

| ID | Requirement |
|---|---|
| NF-01 | Display runs at camera FPS (≥15 FPS) with all detectors active on GPU |
| NF-02 | Detection latency per inference frame ≤ 100 ms on GPU |
| NF-03 | No internet connection required for operation (models downloaded once on first run) |
| NF-04 | Python 3.12; all dependencies pinned in `requirements.txt` |
| NF-05 | Configuration via `.env` file — no hardcoded camera index, paths, or thresholds |
| NF-06 | Face data (embeddings) stored locally in `data/` which is excluded from version control |
| NF-07 | Primary platform: Windows (DirectML GPU inference, CUDA DLL path auto-discovery). Also supports CPU-only on any OS. |
| NF-08 | GPU inference provider priority: DirectML (Windows, no CUDA Toolkit needed) → CUDA → CPU. Automatic fallback to CPU if the configured provider is unavailable. |

---

## 6. Architecture

```
  ┌──────────────────────── Desktop Application ──────────────────────────────┐
  │                                                                            │
  │  USB Cam ──▶ CameraReader (thread)                                        │
  │                    │ latest frame                                          │
  │                    ├──▶ InferenceThread[FaceDetector]    ──▶ FaceAnnotation[]   │
  │                    ├──▶ InferenceThread[PoseDetector]    ──▶ PoseAnnotation[]   │
  │                    └──▶ InferenceThread[PlateDetector]   ──▶ PlateAnnotation[]  │
  │                                    │ latest results (non-blocking)         │
  │                             overlay.py ──▶ cv2.imshow()                   │
  │                                                                            │
  │  data/latest_frame.jpg  ◀── saved at ≤8 FPS (web enrollment preview)      │
  │  data/camera_paused.flag ◀── written/deleted to coordinate camera handoff │
  └────────────────────────────────────────────────────────────────────────────┘
              │ reads/writes                    ▲ camera handoff flag
        data/faces.db                          │
        (SQLite + 512-d embeddings)            │
              ▲ writes                         │
  ┌──────────── Web Face Enrollment (browser) ─────────────────────────────────┐
  │  Photo upload  ──▶ FastAPI ──▶ InsightFace embed ──▶ faces.db              │
  │  Webcam capture (/webcam): 5-pose guided session                            │
  │    └──▶ /enroll/frame (base64 JSON) ──▶ yaw validation ──▶ faces.db        │
  │  http://localhost:8000                                                      │
  └─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Technology Stack

| Component | Technology | Version |
|---|---|---|
| Language | Python | 3.12 |
| Skeleton/pose | MediaPipe | 0.10.14 |
| Face recognition | InsightFace (`buffalo_l`) | 0.7.3 |
| License plate | fast-alpr + EasyOCR | 0.4.0 / 1.7.2 |
| Video capture + overlay | OpenCV (`cv2`) | 4.10.0 |
| Face database | SQLite + numpy cosine | Python built-in |
| GPU inference | onnxruntime-gpu (DirectML or CUDA) | 1.21.0 |
| Web enrollment | FastAPI + Jinja2 | 0.115.0 / 3.1.4 |

**GPU inference options:**
- `DEVICE=directml` — DirectX 12 GPU (Windows); no CUDA Toolkit required. **Recommended default.**
- `DEVICE=cuda` — CUDA 12.x + cuDNN 9.x required; auto-discovers CUDA DLL path.
- `DEVICE=cpu` — CPU only; no GPU required.

---

## 8. Phase Roadmap

All phases are complete.

### Phase 0 — Scaffold & Camera Window ✅
Camera opens, raw feed displays in a window, `q` closes it.

### Phase 1 — Face Recognition ✅
InsightFace `buffalo_l` detects and recognizes enrolled faces. Name + similarity score overlay. CLI enrollment. Auto-reload on DB change.

### Phase 2 — Person Detection + Pose Skeleton ✅
MediaPipe Pose draws full-body 33-landmark skeleton with visibility gating. Person count shown in HUD.

### Phase 3 — License Plate Recognition ✅
fast-alpr + EasyOCR fallback. Plate text normalized. Temporal stabilizer eliminates flicker. Dual confidence thresholds configurable.

### Phase 4 — Web Face Enrollment + Threading Polish ✅
FastAPI enrollment server. Two enrollment flows: photo upload (batch) and guided 5-pose webcam capture with head orientation validation. Camera handoff between desktop and browser. Desktop frame preview in web UI. Threaded camera reader and per-detector inference threads so display never blocks.

---

## 9. License Notes

| Library | License | Notes |
|---|---|---|
| MediaPipe | Apache 2.0 | Free to use |
| OpenCV | Apache 2.0 | Free to use |
| InsightFace model `buffalo_l` | Non-commercial (ArcFace) | Personal/research use only |
| fast-alpr | MIT | Free to use |
| EasyOCR | Apache 2.0 | Free to use |
| FastAPI | MIT | Free to use |

> InsightFace model `buffalo_l` uses the ArcFace model which **may not be used in commercial products** without a license from the model owner. For personal and non-commercial research use, there are no restrictions.

---

## 10. Privacy & Biometric Data

- Face embeddings are stored **locally in `data/`** — not sent to any external server.
- The `data/` and `models/` folders are in `.gitignore` and must never be committed.
- Delete embeddings for persons who no longer need to be recognized.
- The web enrollment server should only be accessible on localhost; do not expose it on a public network.
