"""
Multi-Object Tracker - wraps ultralytics tracking (ByteTrack/OC-SORT).
For fixed cameras, ByteTrack is preferred. For broadcast/moving cameras,
use BoT-SORT with camera motion compensation.
"""
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import time


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    class_name: str = "player"
    team_id: str = "unknown"  # A / B / referee / unknown
    pixel_center: tuple = field(default_factory=lambda: (0, 0))
    foot_position: tuple = field(default_factory=lambda: (0, 0))
    pitch_xy: Optional[tuple] = None  # (x, y) in meters
    age: int = 0  # frames since first seen
    hits: int = 0  # total detections matched
    time_since_update: int = 0

    def __post_init__(self):
        self.update_center()

    def update_center(self):
        cx = (self.bbox[0] + self.bbox[2]) / 2
        cy = (self.bbox[1] + self.bbox[3]) / 2
        self.pixel_center = (cx, cy)
        self.foot_position = (cx, self.bbox[3])


class MultiObjectTracker:
    """Track players and ball across frames."""

    def __init__(self, config: dict):
        self.config = config
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1
        self.frame_count = 0
        self._use_ultralytics = False
        self._init_tracker()

    def _init_tracker(self):
        """Try to initialize ultralytics tracker."""
        try:
            import ultralytics
            self._use_ultralytics = True
            tracker_type = self.config.get("tracker", "bytetrack.yaml")
            print(f"[Tracker] Using ultralytics tracker: {tracker_type}")
        except ImportError:
            print("[Tracker] ultralytics not available, using simple IoU tracker")

    def update(self, detections, frame: np.ndarray = None) -> List[Track]:
        """Update tracks with new detections.
        
        Args:
            detections: List of Detection objects
            frame: Current frame (needed for ultralytics tracking)
            
        Returns:
            List of active Track objects
        """
        self.frame_count += 1

        if self._use_ultralytics and frame is not None:
            return self._update_ultralytics(detections, frame)
        else:
            return self._update_simple(detections)

    def _update_ultralytics(self, detections, frame):
        """Use ultralytics built-in tracker."""
        # This would use model.track() in practice
        # For now, fall back to simple tracker
        return self._update_simple(detections)

    def _update_simple(self, detections) -> List[Track]:
        """Simple IoU-based tracker for when ultralytics is not available.
        Uses greedy IoU matching with distance fallback.
        """
        # Mark all tracks as potentially lost
        for t in self.tracks.values():
            t.time_since_update += 1

        if not detections:
            # Remove tracks that have been lost too long
            max_age = self.config.get("track_buffer", 30)
            lost_ids = [tid for tid, t in self.tracks.items()
                       if t.time_since_update > max_age]
            for tid in lost_ids:
                del self.tracks[tid]
            return list(self.tracks.values())

        # Match detections to existing tracks
        det_array = np.array([d.bbox for d in detections])
        track_ids = list(self.tracks.keys())

        if track_ids:
            track_bboxes = np.array([self.tracks[tid].bbox for tid in track_ids])
            iou_matrix = self._compute_iou_matrix(track_bboxes, det_array)

            # Greedy matching
            matched_tracks = set()
            matched_dets = set()
            for _ in range(min(len(track_ids), len(detections))):
                if iou_matrix.size == 0:
                    break
                max_iou_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                max_iou = iou_matrix[max_iou_idx]
                if max_iou < 0.1:  # Minimum IoU threshold
                    break
                ti, di = max_iou_idx
                tid = track_ids[ti]
                det = detections[di]

                # Update track
                self.tracks[tid].bbox = det.bbox
                self.tracks[tid].confidence = det.confidence
                self.tracks[tid].class_name = det.class_name
                self.tracks[tid].update_center()
                self.tracks[tid].time_since_update = 0
                self.tracks[tid].hits += 1
                self.tracks[tid].age += 1

                matched_tracks.add(ti)
                matched_dets.add(di)
                iou_matrix[ti, :] = 0
                iou_matrix[:, di] = 0

            # Create new tracks for unmatched detections
            for di, det in enumerate(detections):
                if di not in matched_dets:
                    new_track = Track(
                        track_id=self.next_id,
                        bbox=det.bbox,
                        confidence=det.confidence,
                        class_name=det.class_name
                    )
                    self.tracks[self.next_id] = new_track
                    self.next_id += 1
        else:
            # No existing tracks, create all new
            for det in detections:
                new_track = Track(
                    track_id=self.next_id,
                    bbox=det.bbox,
                    confidence=det.confidence,
                    class_name=det.class_name
                )
                self.tracks[self.next_id] = new_track
                self.next_id += 1

        # Remove stale tracks
        max_age = self.config.get("track_buffer", 30)
        lost_ids = [tid for tid, t in self.tracks.items()
                   if t.time_since_update > max_age]
        for tid in lost_ids:
            del self.tracks[tid]

        return list(self.tracks.values())

    def _compute_iou_matrix(self, boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
        """Compute IoU matrix between two sets of bboxes."""
        n, m = len(boxes_a), len(boxes_b)
        iou_matrix = np.zeros((n, m))

        for i in range(n):
            for j in range(m):
                iou_matrix[i, j] = self._iou(boxes_a[i], boxes_b[j])

        return iou_matrix

    @staticmethod
    def _iou(box1, box2) -> float:
        """Compute IoU between two boxes [x1,y1,x2,y2]."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0.0

    def get_active_tracks(self) -> List[Track]:
        """Get all currently active tracks."""
        return [t for t in self.tracks.values() if t.time_since_update == 0]
