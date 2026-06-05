import threading
import time

import numpy as np

from app.detectors.base import Detector


class InferenceThread:
    """
    Run a detector in a background thread. The main loop never waits for
    inference to finish — it always uses the latest available result.

    Pattern: submit(frame) → non-blocking. get_results() → non-blocking.
    The inference thread always processes the latest frame; older frames are dropped.
    """

    def __init__(self, detector: Detector) -> None:
        self._detector = detector
        self._pending: np.ndarray | None = None
        self._results: list = []
        self._pending_lock = threading.Lock()
        self._result_lock = threading.Lock()
        self._running = True
        self._inference_count = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, frame: np.ndarray) -> None:
        """Submit the latest frame. Unprocessed older frames are automatically replaced."""
        with self._pending_lock:
            self._pending = frame  # always use the latest frame

    def get_results(self) -> list:
        """Return the latest inference results. Non-blocking."""
        with self._result_lock:
            return list(self._results)

    @property
    def inference_count(self) -> int:
        return self._inference_count

    def _run(self) -> None:
        while self._running:
            frame = None
            with self._pending_lock:
                if self._pending is not None:
                    frame = self._pending
                    self._pending = None

            if frame is not None:
                try:
                    results = self._detector.process(frame)
                    with self._result_lock:
                        self._results = results
                    self._inference_count += 1
                except Exception as e:
                    print(f"[ERROR] Inference failed: {e}")
                    # Continue to next frame, do not crash the thread
            else:
                time.sleep(0.005)

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=2.0)
