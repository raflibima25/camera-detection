import threading

import cv2
import numpy as np


class CameraReader:
    """
    Read camera frames in a separate thread so the main loop never
    blocks waiting on VideoCapture.read().
    """

    def __init__(self, index: int, width: int = 0, height: int = 0) -> None:
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Camera index {index} not found")
        # Set resolution if requested (values ≤0 = use camera default)
        if width > 0:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height > 0:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame

    def read(self) -> np.ndarray | None:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    @property
    def is_open(self) -> bool:
        return self._cap.isOpened()

    def release(self) -> None:
        self._running = False
        self._thread.join(timeout=2.0)
        self._cap.release()
