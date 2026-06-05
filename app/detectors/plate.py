import re
import time

import numpy as np

from app import config
from app.detectors.base import Detector, PlateAnnotation

_MIN_LEN = 4
_MAX_LEN = 12

# Minimum IoU to match new detections to existing tracks
_IOU_MATCH_THRESH = 0.3


def _normalize_plate(text: str) -> str:
    """Uppercase, hanya alfanumerik — pola dari ai-vision-cctv/api/pipelines/alpr.py."""
    return re.sub(r"[^A-Z0-9]", "", text.upper().strip())


def _ocr_confidence(conf: float | list[float]) -> float:
    """fast-alpr baru mengembalikan confidence sebagai float atau list per karakter."""
    if isinstance(conf, list):
        return float(sum(conf) / len(conf)) if conf else 0.0
    return float(conf)


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Compute Intersection-over-Union of two bboxes (x1,y1,x2,y2)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class _PlateTrack:
    """A single tracked license plate."""
    __slots__ = ("bbox", "text", "confidence", "last_seen")

    def __init__(self, ann: PlateAnnotation) -> None:
        self.bbox: tuple[int, int, int, int] = ann.bbox
        self.text: str = ann.text
        self.confidence: float = ann.confidence
        self.last_seen: float = time.monotonic()

    def update(self, ann: PlateAnnotation) -> None:
        self.bbox = ann.bbox
        self.last_seen = time.monotonic()
        # Keep text with highest confidence → reduces text jitter
        if ann.confidence >= self.confidence:
            self.text = ann.text
            self.confidence = ann.confidence

    def annotation(self) -> PlateAnnotation:
        return PlateAnnotation(bbox=self.bbox, text=self.text, confidence=self.confidence)

    def is_alive(self, ttl: float) -> bool:
        return (time.monotonic() - self.last_seen) < ttl


class _PlateStabilizer:
    """
    Temporal persistence for plate detections.

    - Match each new detection to an existing track via IoU.
    - Tracks not seen in the current frame are still emitted for < ttl seconds.
    - Tracks past the TTL are discarded → no "ghost plates".
    """

    def __init__(self, ttl: float) -> None:
        self._ttl = ttl
        self._tracks: list[_PlateTrack] = []

    def update(self, detections: list[PlateAnnotation]) -> list[PlateAnnotation]:
        matched_track_ids: set[int] = set()

        for ann in detections:
            best_idx = -1
            best_iou = _IOU_MATCH_THRESH

            for i, track in enumerate(self._tracks):
                score = _iou(ann.bbox, track.bbox)
                if score > best_iou:
                    best_iou = score
                    best_idx = i

            if best_idx >= 0:
                self._tracks[best_idx].update(ann)
                matched_track_ids.add(best_idx)
            else:
                # New plate — create new track
                self._tracks.append(_PlateTrack(ann))
                matched_track_ids.add(len(self._tracks) - 1)

        # Keep tracks still within TTL, discard expired ones
        self._tracks = [
            t for t in self._tracks if t.is_alive(self._ttl)
        ]

        return [t.annotation() for t in self._tracks]


class PlateDetector(Detector):
    def __init__(self) -> None:
        self._alpr = None
        self._ocr = None
        self._use_fallback = False
        self._error_logged = False  # log runtime error only once
        self._stabilizer = _PlateStabilizer(ttl=config.PLATE_PERSIST_SECONDS)
        self._load()

    def _load(self) -> None:
        try:
            from fast_alpr import ALPR
            self._alpr = ALPR(
                detector_model="yolo-v9-t-640-license-plate-end2end",
                ocr_model="global-plates-mobile-vit-v2-model",
                # --- parameters previously not passed through ---
                detector_conf_thresh=config.PLATE_DETECTOR_CONF_THRESHOLD,
                detector_providers=config.get_ort_providers(),   # CUDA → CPU fallback
                ocr_device="cuda" if config.DEVICE == "cuda" else "auto",
            )
            print(
                f"[INFO] fast-alpr loaded "
                f"(det_conf={config.PLATE_DETECTOR_CONF_THRESHOLD}, "
                f"ocr_conf={config.PLATE_CONFIDENCE_THRESHOLD}, "
                f"persist={config.PLATE_PERSIST_SECONDS}s)"
            )
        except Exception as e:
            print(f"[WARN] fast-alpr failed: {e}")
            print("[WARN] Plate detector falling back to EasyOCR")
            self._use_fallback = True
            self._load_easyocr()

    def _load_easyocr(self) -> None:
        if self._ocr is None:
            import easyocr
            gpu = config.DEVICE == "cuda"
            self._ocr = easyocr.Reader(["en"], gpu=gpu)
            print("[INFO] EasyOCR loaded")

    # --- fast-alpr path ---

    def _run_alpr(self, frame: np.ndarray) -> list[PlateAnnotation]:
        results = self._alpr.predict(frame)
        annotations = []
        for r in results:
            if r.ocr is None:
                continue
            conf = _ocr_confidence(r.ocr.confidence)
            if conf < config.PLATE_CONFIDENCE_THRESHOLD:
                continue
            text = _normalize_plate(r.ocr.text)
            if not (_MIN_LEN <= len(text) <= _MAX_LEN):
                continue
            bb = r.detection.bounding_box
            annotations.append(PlateAnnotation(
                bbox=(int(bb.x1), int(bb.y1), int(bb.x2), int(bb.y2)),
                text=text,
                confidence=conf,
            ))
        return annotations

    # --- EasyOCR fallback path ---

    def _run_easyocr(self, frame: np.ndarray) -> list[PlateAnnotation]:
        self._load_easyocr()
        results = self._ocr.readtext(frame, detail=1)
        annotations = []
        for (box_pts, text, score) in results:
            if score < config.PLATE_CONFIDENCE_THRESHOLD:
                continue
            text = _normalize_plate(text)
            if not (_MIN_LEN <= len(text) <= _MAX_LEN):
                continue
            pts = np.array(box_pts, dtype=int)
            x1, y1 = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)
            annotations.append(PlateAnnotation(
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                text=text,
                confidence=float(score),
            ))
        return annotations

    # --- Detector interface ---

    def process(self, frame: np.ndarray) -> list[PlateAnnotation]:
        if self._use_fallback:
            raw = self._run_easyocr(frame)
        else:
            try:
                raw = self._run_alpr(frame)
            except Exception as e:
                if not self._error_logged:
                    print(f"[ERROR] fast-alpr: {e}")
                    self._error_logged = True
                raw = []

        # Stabilizer: keep box when OCR occasionally misses → no flickering
        return self._stabilizer.update(raw)
