import os
import sys
import time
import warnings

# Suppress log noise
os.environ.setdefault("GLOG_minloglevel", "3")          # MediaPipe C++ logs
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")      # TFLite logs
os.environ.setdefault("ORT_LOGGING_LEVEL", "3")          # onnxruntime C++ errors (TensorRT dll dll)
os.environ.setdefault("ORT_TENSORRT_ENGINE_CACHE_ENABLE", "0")

# Add CUDA bin to DLL search path programmatically.
# Scan filesystem directly — does not depend on terminal env vars that may be stale.
def _setup_cuda_dll_path() -> None:
    import glob

    candidates: list[str] = []

    # 1. Check env vars (present if terminal was restarted after CUDA install)
    for var in ("CUDA_PATH", "CUDA_PATH_V12_8", "CUDA_PATH_V12_9"):
        val = os.environ.get(var, "")
        if val:
            candidates.append(os.path.join(val, "bin"))

    # 2. Scan filesystem directly at standard Windows CUDA locations
    cuda_base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    for v12 in sorted(glob.glob(os.path.join(cuda_base, "v12*")), reverse=True):
        candidates.append(os.path.join(v12, "bin"))

    for cuda_bin in candidates:
        if not os.path.isdir(cuda_bin):
            continue
        # Ensure cudart64_12.dll is actually present
        if not os.path.exists(os.path.join(cuda_bin, "cudart64_12.dll")):
            continue
        try:
            os.add_dll_directory(cuda_bin)
        except (AttributeError, OSError):
            pass
        if cuda_bin.lower() not in os.environ.get("PATH", "").lower():
            os.environ["PATH"] = cuda_bin + os.pathsep + os.environ.get("PATH", "")
        print(f"[INFO] CUDA DLL path registered: {cuda_bin}")
        break  # one version is enough

_setup_cuda_dll_path()

import cv2

warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

from pathlib import Path

from app import config
from app.camera import CameraReader
from app.inference_thread import InferenceThread

# Flag written by the web server when an enrollment session is active
_PAUSE_FLAG = Path("data") / "camera_paused.flag"


def main() -> None:
    # --- Camera initialization ---
    try:
        camera = CameraReader(config.CAMERA_INDEX, config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        print("        Try changing CAMERA_INDEX in the .env file")
        sys.exit(1)

    print(f"[INFO] Camera index {config.CAMERA_INDEX} opened (threaded reader)")

    # --- Load active detectors ---
    face_thread = None
    pose_thread = None
    plate_thread = None
    face_db = None

    if config.ENABLE_FACE:
        from app.face_db import FaceDB
        from app.detectors.face import FaceDetector
        face_db = FaceDB()
        face_thread = InferenceThread(FaceDetector(face_db))

    if config.ENABLE_POSE:
        from app.detectors.pose import PoseDetector
        pose_thread = InferenceThread(PoseDetector())

    if config.ENABLE_PLATE:
        from app.detectors.plate import PlateDetector
        plate_thread = InferenceThread(PlateDetector())

    from app.overlay import draw_faces, draw_plates, draw_pose

    print("[INFO] Press 'q' to quit")

    while camera.read() is None:
        time.sleep(0.01)

    # Latest frame for web enrollment (save clean frame before overlay)
    _enroll_frame_path = "data/latest_frame.jpg"
    _last_frame_save = 0.0

    # --- FPS state ---
    prev_time = time.time()
    last_reload = time.time()
    last_face_count = 0
    last_pose_count = 0
    last_plate_count = 0
    face_ai_fps = 0.0
    pose_ai_fps = 0.0
    plate_ai_fps = 0.0
    fps_update_time = time.time()

    while True:
        # Handoff: release camera to browser when web enrollment is active
        if _PAUSE_FLAG.exists():
            print("[INFO] Releasing camera for web enrollment...")
            camera.release()
            while _PAUSE_FLAG.exists():
                time.sleep(0.1)
            time.sleep(0.3)  # give OS time to fully release camera driver
            print("[INFO] Reacquiring camera...")
            try:
                camera = CameraReader(config.CAMERA_INDEX, config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
            except RuntimeError as e:
                print(f"[ERROR] Failed to reacquire camera: {e}")
                time.sleep(1.0)
                continue
            while camera.read() is None:
                time.sleep(0.01)
            prev_time = time.time()
            fps_update_time = time.time()
            print("[INFO] Camera is active again")
            continue

        frame = camera.read()
        if frame is None:
            continue

        now = time.time()

        # Auto-reload face DB every 5 seconds
        if face_db and now - last_reload >= 5.0:
            face_db.check_reload()
            last_reload = now

        # Save clean frame for web enrollment (max 8 FPS to avoid excessive I/O)
        if now - _last_frame_save >= 0.125:
            cv2.imwrite(_enroll_frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            _last_frame_save = now

        # --- Submit frame to all inference threads (non-blocking) ---
        if face_thread:
            face_thread.submit(frame)
        if pose_thread:
            pose_thread.submit(frame)
        if plate_thread:
            plate_thread.submit(frame)

        # --- Draw latest inference results ---
        if face_thread:
            draw_faces(frame, face_thread.get_results())
        if plate_thread:
            draw_plates(frame, plate_thread.get_results())
        if pose_thread:
            pose_anns = pose_thread.get_results()
            draw_pose(frame, pose_anns)
            # Person count in top-right corner
            person_count = len(pose_anns)
            label = f"Persons: {person_count}"
            (lw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.putText(
                frame, label,
                (frame.shape[1] - lw - 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 230, 245), 2,
            )

        # --- HUD ---
        cam_fps = 1.0 / (now - prev_time + 1e-9)
        prev_time = now

        # Update AI FPS every 0.5 seconds
        if now - fps_update_time >= 0.5:
            dt = now - fps_update_time
            if face_thread:
                c = face_thread.inference_count
                face_ai_fps = (c - last_face_count) / dt
                last_face_count = c
            if pose_thread:
                c = pose_thread.inference_count
                pose_ai_fps = (c - last_pose_count) / dt
                last_pose_count = c
            if plate_thread:
                c = plate_thread.inference_count
                plate_ai_fps = (c - last_plate_count) / dt
                last_plate_count = c
            fps_update_time = now

        cv2.putText(
            frame, f"CAM: {cam_fps:.0f} FPS",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2,
        )
        y = 56
        if face_thread:
            cv2.putText(
                frame, f"Face AI: {face_ai_fps:.1f} FPS",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1,
            )
            y += 22
        if pose_thread:
            cv2.putText(
                frame, f"Pose AI: {pose_ai_fps:.1f} FPS",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 230, 245), 1,
            )
            y += 22
        if plate_thread:
            cv2.putText(
                frame, f"Plate AI: {plate_ai_fps:.1f} FPS",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 220), 1,
            )

        # Active detector status at the bottom
        active = [d for d, on in [
            ("FACE", config.ENABLE_FACE),
            ("POSE", config.ENABLE_POSE),
            ("PLATE", config.ENABLE_PLATE),
        ] if on]
        cv2.putText(
            frame, "  |  ".join(active) if active else "—",
            (10, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (180, 180, 180), 1,
        )

        cv2.imshow("Camera Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # --- Clean shutdown ---
    if face_thread:
        face_thread.stop()
    if pose_thread:
        pose_thread.stop()
    if plate_thread:
        plate_thread.stop()
    camera.release()
    cv2.destroyAllWindows()
    print("[INFO] Application closed")


if __name__ == "__main__":
    main()
