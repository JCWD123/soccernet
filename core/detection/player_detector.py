"""
Player/Referee Detection using YOLOv8.
Detects all persons on the field; team classification happens downstream.
"""
import numpy as np
from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class Detection:
    bbox: np.ndarray  # [x1, y1, x2, y2] in pixel coords
    confidence: float
    class_id: int  # 0=player, 1=goalkeeper, 2=referee (our classes)
    class_name: str = "player"
    pixel_center: tuple = field(default_factory=lambda: (0, 0))

    def __post_init__(self):
        cx = (self.bbox[0] + self.bbox[2]) / 2
        cy = (self.bbox[1] + self.bbox[3]) / 2
        self.pixel_center = (cx, cy)
        # Bottom-center for pitch coordinate mapping (feet position)
        self.foot_position = (cx, self.bbox[3])


class PlayerDetector:
    """Detect players using YOLOv8 on downscaled frames."""

    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
            model_path = self.config.get("model", "yolov8n.pt")
            self.model = YOLO(model_path)
            print(f"[PlayerDetector] Loaded model: {model_path}")
        except ImportError:
            print("[PlayerDetector] ultralytics not installed, using mock mode")
            self.model = None

    def detect(self, frame: np.ndarray, confidence: float = None) -> List[Detection]:
        """Detect players in a frame.
        
        Args:
            frame: BGR image (numpy array)
            confidence: Override confidence threshold
            
        Returns:
            List of Detection objects
        """
        conf = confidence or self.config.get("confidence", 0.5)
        coco_classes = self.config.get("classes", [0])  # person
        input_size = self.config.get("input_size", 1280)

        if self.model is None:
            return self._mock_detect(frame)

        results = self.model(
            frame,
            conf=conf,
            classes=coco_classes,
            imgsz=input_size,
            verbose=False
        )

        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                det = Detection(
                    bbox=xyxy,
                    confidence=float(box.conf[0]),
                    class_id=0,  # All detected as player; team/rule later
                    class_name="player"
                )
                detections.append(det)

        return detections

    def _mock_detect(self, frame: np.ndarray) -> List[Detection]:
        """Mock detection for testing without model."""
        h, w = frame.shape[:2]
        # Generate synthetic detections for testing
        mock_detections = []
        # Simulate 20-25 players scattered on field
        np.random.seed(42)
        for i in range(22):
            cx = np.random.randint(100, w - 100)
            cy = np.random.randint(h // 4, h - 50)
            bw, bh = 40, 80
            mock_detections.append(Detection(
                bbox=np.array([cx - bw//2, cy - bh//2, cx + bw//2, cy + bh//2], dtype=np.float32),
                confidence=0.85 + np.random.random() * 0.1,
                class_id=0,
                class_name="player"
            ))
        return mock_detections
