"""
Team Classification using jersey color histogram + temporal majority vote.
MVP approach: LAB color space histogram of upper torso region,
then k-means or threshold to split A/B, with per-track voting.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import cv2


class TeamClassifier:
    """Classify players into Team A / Team B by jersey color."""

    def __init__(self, config: dict):
        self.config = config
        self.vote_window = config.get("vote_window", 30)
        self.min_confidence = config.get("min_confidence", 0.6)
        self.team_a_seed = config.get("team_a_color_seed", None)
        self.team_b_seed = config.get("team_b_color_seed", None)

        # Per-track vote history: track_id -> list of (team, confidence)
        self.vote_history: Dict[int, List[Tuple[str, float]]] = defaultdict(list)
        # Team centroids in LAB space (auto-computed or from seed)
        self.team_a_centroid: Optional[np.ndarray] = None
        self.team_b_centroid: Optional[np.ndarray] = None
        self.is_initialized = False

    def classify_frame(self, frame: np.ndarray, tracks: list) -> Dict[int, str]:
        """Classify all tracks in current frame.
        
        Args:
            frame: BGR image
            tracks: List of Track objects with bbox
            
        Returns:
            Dict mapping track_id -> team_id ("A", "B", or "unknown")
        """
        results = {}
        features = []

        for track in tracks:
            # Extract jersey ROI (upper 40% of bbox)
            x1, y1, x2, y2 = track.bbox.astype(int)
            h = y2 - y1
            jersey_y2 = y1 + int(h * 0.4)
            jersey_roi = frame[max(0, y1):jersey_y2, max(0, x1):min(frame.shape[1], x2)]

            if jersey_roi.size == 0:
                results[track.track_id] = "unknown"
                continue

            # Compute LAB histogram feature
            feature = self._extract_color_feature(jersey_roi)
            features.append((track.track_id, feature, track))

        if not features:
            return results

        # Initialize centroids if needed
        if not self.is_initialized:
            all_features = np.array([f for _, f, _ in features])
            self._initialize_centroids(all_features)

        # Classify each track
        for track_id, feature, track in features:
            team, confidence = self._classify_single(feature)
            self.vote_history[track_id].append((team, confidence))

            # Keep only recent votes
            if len(self.vote_history[track_id]) > self.vote_window:
                self.vote_history[track_id] = self.vote_history[track_id][-self.vote_window:]

            # Temporal majority vote
            final_team = self._majority_vote(track_id)
            results[track_id] = final_team

        return results

    def _extract_color_feature(self, roi: np.ndarray) -> np.ndarray:
        """Extract LAB color histogram feature from jersey ROI."""
        # Convert to LAB
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        # Compute histogram: L(8) x A(8) x B(8) = 512 dims, but we use reduced
        hist_l = cv2.calcHist([lab], [0], None, [8], [0, 256])
        hist_a = cv2.calcHist([lab], [1], None, [8], [0, 256])
        hist_b = cv2.calcHist([lab], [2], None, [8], [0, 256])
        # Concatenate and normalize
        hist = np.concatenate([hist_l, hist_a, hist_b]).flatten()
        hist = hist / (hist.sum() + 1e-7)
        return hist

    def _initialize_centroids(self, features: np.ndarray):
        """Initialize team centroids using k-means or seed colors."""
        if self.team_a_seed is not None and self.team_b_seed is not None:
            # Use seed colors (simplified: convert RGB to LAB-like feature)
            self.is_initialized = True
            return

        # Simple 2-cluster initialization using k-means
        if len(features) >= 2:
            from scipy.cluster.vq import kmeans2
            try:
                centroids, labels = kmeans2(features, 2, minit='points')
                self.team_a_centroid = centroids[0]
                self.team_b_centroid = centroids[1]
                self.is_initialized = True
            except Exception:
                # Fallback: just use first two features
                self.team_a_centroid = features[0]
                self.team_b_centroid = features[1] if len(features) > 1 else features[0]
                self.is_initialized = True

    def _classify_single(self, feature: np.ndarray) -> Tuple[str, float]:
        """Classify a single feature vector."""
        if self.team_a_centroid is None or self.team_b_centroid is None:
            return "unknown", 0.0

        dist_a = np.linalg.norm(feature - self.team_a_centroid)
        dist_b = np.linalg.norm(feature - self.team_b_centroid)

        total = dist_a + dist_b + 1e-7
        conf_a = 1.0 - dist_a / total
        conf_b = 1.0 - dist_b / total

        if dist_a < dist_b:
            return "A", conf_a
        else:
            return "B", conf_b

    def _majority_vote(self, track_id: int) -> str:
        """Temporal majority vote for a track."""
        votes = self.vote_history.get(track_id, [])
        if not votes:
            return "unknown"

        # Weighted vote by confidence
        score_a = sum(conf for team, conf in votes if team == "A")
        score_b = sum(conf for team, conf in votes if team == "B")

        total = score_a + score_b
        if total < 1e-7:
            return "unknown"

        if score_a / total > self.min_confidence:
            return "A"
        elif score_b / total > self.min_confidence:
            return "B"
        else:
            return "unknown"

    def set_team_seeds(self, frame: np.ndarray, team_a_bbox: list, team_b_bbox: list):
        """Set team color seeds from user-selected player bboxes.
        Call this from UI before match starts for faster convergence.
        """
        for bbox, is_a in [(team_a_bbox, True), (team_b_bbox, False)]:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            feature = self._extract_color_feature(roi)
            if is_a:
                self.team_a_centroid = feature
            else:
                self.team_b_centroid = feature
            self.is_initialized = True

    def reset(self):
        """Reset for new match."""
        self.vote_history.clear()
        self.is_initialized = False
        self.team_a_centroid = None
        self.team_b_centroid = None
