"""
YOLOv8 Person Detector
Uses Ultralytics YOLOv8 nano model for fast person detection.
Only detects the 'person' class (class ID 0) for queue analysis.
"""

import numpy as np
from ultralytics import YOLO
import torch


class PersonDetector:
    """Real-time person detector using YOLOv8."""

    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.3):
        """
        Args:
            model_name: YOLOv8 model variant (n/s/m/l/x)
            confidence: Minimum detection confidence threshold
        """
        self.confidence = confidence
        self.person_class = 0  # COCO class ID for 'person'

        # Auto-detect GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Detector] Loading {model_name} on {self.device}")

        self.model = YOLO(model_name)
        self.model.to(self.device)
        print(f"[Detector] Model loaded successfully")

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect persons in a frame.

        Args:
            frame: BGR image (H, W, 3)

        Returns:
            np.ndarray of shape (N, 5) with [x1, y1, x2, y2, confidence]
            Empty array (0, 5) if no detections.
        """
        results = self.model(
            frame,
            conf=self.confidence,
            classes=[self.person_class],
            verbose=False
        )

        detections = []
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    detections.append([float(x1), float(y1), float(x2), float(y2), conf])

        if detections:
            return np.array(detections, dtype=np.float64)
        return np.empty((0, 5), dtype=np.float64)

    def warmup(self, imgsz: tuple = (640, 640)):
        """Run a warmup inference to initialize the model."""
        dummy = np.zeros((*imgsz, 3), dtype=np.uint8)
        self.detect(dummy)
        print("[Detector] Warmup complete")
