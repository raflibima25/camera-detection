import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis

from app import config
from app.detectors.base import Detector, FaceAnnotation
from app.face_db import FaceDB

_MAX_INFERENCE_WIDTH = 640


def _cuda_dll_loadable() -> bool:
    """Check if onnxruntime_providers_cuda.dll can be loaded (CUDA dependencies in PATH)."""
    import ctypes, os
    cuda_dll = os.path.join(os.path.dirname(ort.__file__), "capi", "onnxruntime_providers_cuda.dll")
    if not os.path.exists(cuda_dll):
        return False
    try:
        ctypes.CDLL(cuda_dll)
        return True
    except OSError:
        return False


class FaceDetector(Detector):
    def __init__(self, db: FaceDB) -> None:
        self._db = db
        self._app = self._load_with_fallback()

    def _load_with_fallback(self) -> FaceAnalysis:
        """
        Try loading with the provider from config. If it fails during warmup
        (e.g. DirectML incompatible with the model), automatically fall back to CPU.
        """
        det_size = config.FACE_DET_SIZE
        providers_chain = [
            config.get_ort_providers(),
            ["CPUExecutionProvider"],
        ]

        available_providers = ort.get_available_providers()

        for providers in providers_chain:
            requested = providers[0]

            if requested != "CPUExecutionProvider":
                if requested not in available_providers:
                    print(f"[WARN] {requested} not available")
                    if requested == "CUDAExecutionProvider":
                        print("       Make sure onnxruntime-gpu is installed and CUDA bin is in PATH")
                    continue
                # Verify CUDA DLL can actually be loaded (not just listed as available)
                if requested == "CUDAExecutionProvider" and not _cuda_dll_loadable():
                    print("[WARN] CUDAExecutionProvider registered but DLL failed to load (error 126)")
                    print("       Add CUDA bin to PATH:")
                    print("       C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\bin")
                    continue

            try:
                app = FaceAnalysis(
                    name="buffalo_l",
                    root="models",
                    providers=providers,
                    allowed_modules=["detection", "recognition"],
                )
                app.prepare(ctx_id=0, det_size=(det_size, det_size))
                app.get(np.zeros((det_size, det_size, 3), dtype=np.uint8))
                print(f"[INFO] InsightFace loaded — provider: {requested}")
                return app
            except Exception as e:
                if providers == ["CPUExecutionProvider"]:
                    raise RuntimeError(f"InsightFace failed to load: {e}") from e
                print(f"[WARN] {requested} failed during warmup ({type(e).__name__}: {e})")
                print("[WARN] Automatically falling back to CPUExecutionProvider")

    def process(self, frame: np.ndarray) -> list[FaceAnnotation]:
        h, w = frame.shape[:2]

        if w > _MAX_INFERENCE_WIDTH:
            scale = _MAX_INFERENCE_WIDTH / w
            inference_frame = cv2.resize(
                frame, (_MAX_INFERENCE_WIDTH, int(h * scale)),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            inference_frame = frame
            scale = 1.0

        faces = self._app.get(inference_frame)
        result = []
        for face in faces:
            bbox = (face.bbox / scale).astype(int)
            kps = (face.kps / scale).astype(int) if face.kps is not None else None
            x1, y1, x2, y2 = bbox
            name, similarity = self._db.match(face.embedding)
            result.append(FaceAnnotation(
                bbox=(x1, y1, x2, y2),
                kps=kps,
                name=name,
                similarity=similarity,
            ))
        return result
