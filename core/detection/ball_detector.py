"""
Ball Detection - Independent high-res model with tiled inference.
Ball is the hardest and most critical target. Uses tiled high-res
inference and local search window from previous tracking state.
"""
import numpy as np
from typing import List, Optional, Tuple
from .player_detector import Detection


class BallDetector:
    """Detect football using tiled high-resolution inference."""

    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.last_ball_position: Optional[Tuple[float, float]] = None
        self.last_ball_bbox: Optional[np.ndarray] = None
        self.lost_frames: int = 0
        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
            model_path = self.config.get("model", "yolov8n.pt")
            self.model = YOLO(model_path)
            print(f"[BallDetector] Loaded model: {model_path}")
        except ImportError:
            print("[BallDetector] ultralytics not installed, using mock mode")
            self.model = None

    def detect(self, frame: np.ndarray, confidence: float = None) -> List[Detection]:
        """Detect ball in frame, using tiled approach for high resolution.
        
        If we have a previous ball position, search locally first.
        Falls back to full tiled detection if local search fails.
        """
        conf = confidence or self.config.get("confidence", 0.3)
        high_conf = self.config.get("high_confidence", 0.6)
        use_tiling = self.config.get("use_tiling", True)

        if self.model is None:
            return self._mock_detect(frame)

        # Local search if we have previous position
        if self.last_ball_position is not None and self.lost_frames < 10:
            local_dets = self._detect_local(frame, conf)
            if local_dets:
                self.last_ball_position = local_dets[0].pixel_center
                self.last_ball_bbox = local_dets[0].bbox
                self.lost_frames = 0
                return local_dets

        # Full tiled detection
        if use_tiling and frame.shape[1] > 1280:
            all_dets = self._detect_tiled(frame, conf)
        else:
            all_dets = self._detect_single(frame, conf)

        if all_dets:
            # Pick highest confidence ball
            best = max(all_dets, key=lambda d: d.confidence)
            self.last_ball_position = best.pixel_center
            self.last_ball_bbox = best.bbox
            self.lost_frames = 0
            return [best]

        self.lost_frames += 1
        return []

    def _detect_single(self, frame: np.ndarray, conf: float) -> List[Detection]:
        """Single-pass detection."""
        results = self.model(
            frame, conf=conf, classes=[32],  # sports ball
            imgsz=self.config.get("input_size", 640),
            verbose=False
        )
        dets = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                dets.append(Detection(
                    bbox=xyxy,
                    confidence=float(box.conf[0]),
                    class_id=32,
                    class_name="ball"
                ))
        return dets

    def _detect_local(self, frame: np.ndarray, conf: float) -> List[Detection]:
        """Search near last known ball position."""
        h, w = frame.shape[:2]
        lx, ly = self.last_ball_position
        # Expand search window based on bbox size
        expansion = self.config.get("search_window_expansion", 2.0)
        if self.last_ball_bbox is not None:
            bw = (self.last_ball_bbox[2] - self.last_ball_bbox[0]) * expansion
            bh = (self.last_ball_bbox[3] - self.last_ball_bbox[1]) * expansion
        else:
            bw, bh = 200, 200

        x1 = max(0, int(lx - bw))
        y1 = max(0, int(ly - bh))
        x2 = min(w, int(lx + bw))
        y2 = min(h, int(ly + bh))

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return []

        dets = self._detect_single(roi, conf)
        # Adjust coordinates to full frame
        for d in dets:
            d.bbox[0] += x1
            d.bbox[1] += y1
            d.bbox[2] += x1
            d.bbox[3] += y1
            cx = (d.bbox[0] + d.bbox[2]) / 2
            cy = (d.bbox[1] + d.bbox[3]) / 2
            d.pixel_center = (cx, cy)
        return dets

    def _detect_tiled(self, frame: np.ndarray, conf: float) -> List[Detection]:
        """Tiled inference for high-res frames."""
        h, w = frame.shape[:2]
        tile_w, tile_h = self.config.get("tile_size", [960, 540])
        overlap = self.config.get("overlap", 100)

        all_dets = []
        for y in range(0, h, tile_h - overlap):
            for x in range(0, w, tile_w - overlap):
                y2 = min(y + tile_h, h)
                x2 = min(x + tile_w, w)
                tile = frame[y:y2, x:x2]
                dets = self._detect_single(tile, conf)
                for d in dets:
                    d.bbox[0] += x
                    d.bbox[1] += y
                    d.bbox[2] += x
                    d.bbox[3] += y
                    cx = (d.bbox[0] + d.bbox[2]) / 2
                    cy = (d.bbox[1] + d.bbox[3]) / 2
                    d.pixel_center = (cx, cy)
                all_dets.extend(dets)

        return all_dets

    def _mock_detect(self, frame: np.ndarray) -> List[Detection]:
        """Mock ball detection for testing."""
        h, w = frame.shape[:2]
        # Simulate ball near center-ish of frame
        cx, cy = w // 2 + np.random.randint(-200, 200), h // 2 + np.random.randint(-100, 100)
        r = 8
        return [Detection(
            bbox=np.array([cx-r, cy-r, cx+r, cy+r], dtype=np.float32),
            confidence=0.75,
            class_id=32,
            class_name="ball"
        )]
