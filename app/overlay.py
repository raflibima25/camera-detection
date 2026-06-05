import cv2
import numpy as np

from app.detectors.base import FaceAnnotation, PlateAnnotation, PoseAnnotation

# --- Face ---
_KPS_CONNECTIONS = [(0, 1), (0, 2), (1, 2), (2, 3), (2, 4)]
_COLOR_KNOWN   = (0, 220, 0)      # green   — known face
_COLOR_UNKNOWN = (0, 0, 220)      # red     — unknown face
_COLOR_KPS     = (255, 60, 255)   # magenta — face landmark points
_COLOR_KPS_LINE = (180, 60, 180)

# --- License plate ---
_PLATE_BOX_COLOR  = (0, 220, 220)   # yellow (BGR)
_PLATE_TEXT_COLOR = (0, 0, 0)       # black

# --- Pose skeleton ---
# BGR: body lines = dark orange, joint dots = yellow/gold
_POSE_LINE_COLOR = (66, 117, 245)   # orange (BGR)
_POSE_DOT_COLOR  = (60, 230, 245)   # yellow (BGR)
_VIS_THRESHOLD   = 0.5              # minimum visibility to draw


def draw_faces(frame: np.ndarray, annotations: list[FaceAnnotation]) -> None:
    for ann in annotations:
        color = _COLOR_KNOWN if ann.name != "Unknown" else _COLOR_UNKNOWN
        x1, y1, x2, y2 = ann.bbox

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{ann.name}  {ann.similarity:.2f}"
        (lw, lh), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )
        label_y = max(y1 - 4, lh + 4)
        cv2.rectangle(
            frame,
            (x1, label_y - lh - baseline - 2),
            (x1 + lw + 4, label_y + 2),
            color, -1,
        )
        cv2.putText(
            frame, label,
            (x1 + 2, label_y - baseline),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 255, 255), 1, cv2.LINE_AA,
        )

        if ann.kps is not None:
            for i, j in _KPS_CONNECTIONS:
                cv2.line(frame, tuple(ann.kps[i]), tuple(ann.kps[j]),
                         _COLOR_KPS_LINE, 1, cv2.LINE_AA)
            for kp in ann.kps:
                cv2.circle(frame, tuple(kp), 3, _COLOR_KPS, -1)


def draw_plates(frame: np.ndarray, annotations: list[PlateAnnotation]) -> None:
    for ann in annotations:
        x1, y1, x2, y2 = ann.bbox

        # Plate box — thick yellow border
        cv2.rectangle(frame, (x1, y1), (x2, y2), _PLATE_BOX_COLOR, 3)

        # Label: plate text + confidence
        label = f"{ann.text}  {ann.confidence:.2f}"
        (lw, lh), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
        )
        label_y = y2 + lh + baseline + 4  # show below the box
        if label_y > frame.shape[0]:      # if too close to bottom, show above
            label_y = y1 - baseline - 2
        cv2.rectangle(
            frame,
            (x1, label_y - lh - baseline),
            (x1 + lw + 4, label_y + 2),
            _PLATE_BOX_COLOR, -1,
        )
        cv2.putText(
            frame, label,
            (x1 + 2, label_y - baseline),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            _PLATE_TEXT_COLOR, 2, cv2.LINE_AA,
        )


def draw_pose(frame: np.ndarray, annotations: list[PoseAnnotation]) -> None:
    for ann in annotations:
        # Joint connection lines
        for i, j in ann.connections:
            if i >= len(ann.landmarks) or j >= len(ann.landmarks):
                continue
            x1, y1, vis1 = ann.landmarks[i]
            x2, y2, vis2 = ann.landmarks[j]
            if vis1 >= _VIS_THRESHOLD and vis2 >= _VIS_THRESHOLD:
                cv2.line(frame, (x1, y1), (x2, y2),
                         _POSE_LINE_COLOR, 2, cv2.LINE_AA)

        # Joint dots
        for x, y, vis in ann.landmarks:
            if vis >= _VIS_THRESHOLD:
                cv2.circle(frame, (x, y), 4, _POSE_DOT_COLOR, -1)
                cv2.circle(frame, (x, y), 4, (255, 255, 255), 1)  # white outline
