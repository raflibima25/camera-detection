from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class FaceAnnotation:
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    kps: np.ndarray | None           # shape (5, 2) — 5 face keypoints
    name: str
    similarity: float


@dataclass
class PoseAnnotation:
    landmarks: list[tuple[int, int, float]]  # (x, y, visibility) per landmark
    connections: list[tuple[int, int]]        # pairs of landmark indices for lines


@dataclass
class PlateAnnotation:
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    text: str                        # normalized plate text
    confidence: float


class Detector(ABC):
    @abstractmethod
    def process(self, frame: np.ndarray) -> list:
        """Process one frame, return list of annotations."""
        ...
