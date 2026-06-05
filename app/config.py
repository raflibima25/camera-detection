import os
from dotenv import load_dotenv

load_dotenv()

CAMERA_INDEX: int = int(os.getenv("CAMERA_INDEX", "0"))
CAMERA_WIDTH: int  = int(os.getenv("CAMERA_WIDTH",  "1280"))
CAMERA_HEIGHT: int = int(os.getenv("CAMERA_HEIGHT", "720"))
DEVICE: str = os.getenv("DEVICE", "directml")
FACE_THRESHOLD: float = float(os.getenv("FACE_THRESHOLD", "0.5"))
FACE_DET_SIZE: int = int(os.getenv("FACE_DET_SIZE", "320"))
POSE_MODEL_COMPLEXITY: int = int(os.getenv("POSE_MODEL_COMPLEXITY", "1"))
PLATE_CONFIDENCE_THRESHOLD: float = float(os.getenv("PLATE_CONFIDENCE_THRESHOLD", "0.4"))
PLATE_DETECTOR_CONF_THRESHOLD: float = float(os.getenv("PLATE_DETECTOR_CONF_THRESHOLD", "0.3"))
PLATE_PERSIST_SECONDS: float = float(os.getenv("PLATE_PERSIST_SECONDS", "1.2"))
ENABLE_FACE: bool = os.getenv("ENABLE_FACE", "true").lower() == "true"
ENABLE_POSE: bool = os.getenv("ENABLE_POSE", "true").lower() == "true"
ENABLE_PLATE: bool = os.getenv("ENABLE_PLATE", "true").lower() == "true"


def get_ort_providers() -> list[str]:
    """Return ONNX Runtime execution providers list based on DEVICE setting."""
    if DEVICE == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if DEVICE == "directml":
        return ["DmlExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]
