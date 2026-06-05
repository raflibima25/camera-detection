import cv2
import numpy as np

from app import config
from app.detectors.base import Detector, PoseAnnotation

# MediaPipe Pose 33-landmark skeleton connections (stable, unchanged across versions)
_POSE_CONNECTIONS: list[tuple[int, int]] = [
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    # Shoulders
    (11, 12),
    # Left arm
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    # Right arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    # Torso
    (11, 23), (12, 24), (23, 24),
    # Left leg
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    # Right leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]


def _import_pose_class():
    """
    Import the MediaPipe Pose class with cross-version fallback.
    - 0.10.x old  : mp.solutions.pose.Pose
    - 0.10.x new  : mediapipe.python.solutions.pose.Pose
    """
    # Try old API (mp.solutions) — still present in certain 0.10.x builds
    try:
        import mediapipe as mp
        return mp.solutions.pose.Pose
    except AttributeError:
        pass

    # Try direct submodule import (more stable on 0.10.14+)
    try:
        from mediapipe.python.solutions.pose import Pose
        return Pose
    except ImportError:
        pass

    raise ImportError(
        "MediaPipe Pose not found.\n"
        "Install a compatible version:\n"
        "  pip install mediapipe==0.10.14"
    )


_PoseClass = _import_pose_class()


class PoseDetector(Detector):
    def __init__(self) -> None:
        self._pose = _PoseClass(
            static_image_mode=False,
            model_complexity=config.POSE_MODEL_COMPLEXITY,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        print(
            f"[INFO] MediaPipe Pose loaded "
            f"(model_complexity={config.POSE_MODEL_COMPLEXITY})"
        )

    def process(self, frame: np.ndarray) -> list[PoseAnnotation]:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._pose.process(rgb)

        if not results.pose_landmarks:
            return []

        landmarks = [
            (int(lm.x * w), int(lm.y * h), float(lm.visibility))
            for lm in results.pose_landmarks.landmark
        ]

        return [PoseAnnotation(landmarks=landmarks, connections=_POSE_CONNECTIONS)]
