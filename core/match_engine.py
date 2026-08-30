"""
Match Engine - orchestrates all sub-engines into a unified pipeline.
This is the central coordinator that processes each frame and updates all stats.
"""
import numpy as np
from typing import Optional, Tuple, Dict, List
import time
import yaml
import os

from .detection.player_detector import PlayerDetector
from .detection.ball_detector import BallDetector
from .tracking.tracker import MultiObjectTracker
from .team.classifier import TeamClassifier
from .pitch.calibrator import PitchCalibrator
from .possession.engine import PossessionEngine
from .attack.engine import AttackEngine
from .events.detector import CornerDetector, GoalDetector, CardDetector
from .events.fusion import EventFusionEngine
from .stats.aggregator import StatsAggregator


def load_config(config_dir: str = "configs") -> dict:
    """Load all YAML configs."""
    configs = {}
    for name in ["rules", "pitch", "camera", "detection"]:
        path = os.path.join(config_dir, f"{name}.yaml")
        if os.path.exists(path):
            with open(path, 'r') as f:
                configs[name] = yaml.safe_load(f)
        else:
            configs[name] = {}
    return configs


class MatchEngine:
    """Central match processing engine.
    
    Pipeline per frame:
    1. Detect players + ball
    2. Track objects
    3. Classify teams
    4. Map to pitch coordinates
    5. Update possession
    6. Update attack/dangerous
    7. Detect events (corner, goal, card)
    8. Fuse + confirm events
    9. Update stats
    """

    def __init__(self, config_dir: str = "configs"):
        self.configs = load_config(config_dir)
        self.frame_count = 0
        self.start_time_ms = 0
        self.current_time_ms = 0
        self.fps = 25  # Default FPS

        # Initialize all engines
        detection_cfg = self.configs.get("detection", {})
        rules_cfg = self.configs.get("rules", {})
        pitch_cfg = self.configs.get("pitch", {})
        camera_cfg = self.configs.get("camera", {})

        self.player_detector = PlayerDetector(
            detection_cfg.get("player_detector", {"model": "yolov8n.pt", "confidence": 0.5, "classes": [0]})
        )
        self.ball_detector = BallDetector(
            detection_cfg.get("ball_detector", {"model": "yolov8n.pt", "confidence": 0.3, "classes": [32]})
        )
        self.tracker = MultiObjectTracker(
            detection_cfg.get("tracking", {"tracker": "bytetrack.yaml"})
        )
        self.team_classifier = TeamClassifier(
            detection_cfg.get("team_classification", {"vote_window": 30})
        )
        self.calibrator = PitchCalibrator(camera_cfg.get("camera", {}), pitch_cfg)

        self.possession_engine = PossessionEngine(rules_cfg.get("possession", {}))
        self.attack_engine = AttackEngine(
            rules_cfg.get("attack", {}),
            rules_cfg.get("dangerous_attack", {}),
            pitch_cfg
        )
        self.corner_detector = CornerDetector(rules_cfg.get("corner", {}), pitch_cfg)
        self.goal_detector = GoalDetector(rules_cfg.get("goal", {}), pitch_cfg)
        self.card_detector = CardDetector(rules_cfg.get("card", {}))
        self.fusion_engine = EventFusionEngine(rules_cfg.get("event_review", {}))
        self.stats = StatsAggregator()

    def setup_calibration(self, frame_width: int, frame_height: int,
                          image_points: list = None, pitch_points: list = None):
        """Set up pitch calibration."""
        if image_points and pitch_points:
            self.calibrator.calibrate(image_points, pitch_points)
        else:
            self.calibrator.create_default_calibration(frame_width, frame_height)

    def process_frame(self, frame: np.ndarray, timestamp_ms: float = None) -> dict:
        """Process a single frame through the full pipeline.
        
        Args:
            frame: BGR image (numpy array)
            timestamp_ms: Frame timestamp in milliseconds
            
        Returns:
            Dict with frame results and current stats
        """
        self.frame_count += 1
        if timestamp_ms is None:
            timestamp_ms = self.frame_count * (1000 / self.fps)
        self.current_time_ms = timestamp_ms

        h, w = frame.shape[:2]

        # 1. Detection
        player_dets = self.player_detector.detect(frame)
        ball_dets = self.ball_detector.detect(frame)

        # 2. Tracking
        all_dets = player_dets + ball_dets
        tracks = self.tracker.update(all_dets, frame)

        # Separate player and ball tracks
        player_tracks = [t for t in tracks if t.class_name == "player"]
        ball_tracks = [t for t in tracks if t.class_name == "ball"]

        # 3. Team classification
        team_assignments = self.team_classifier.classify_frame(frame, player_tracks)
        for track in player_tracks:
            track.team_id = team_assignments.get(track.track_id, "unknown")

        # 4. Map to pitch coordinates
        ball_pitch_xy = None
        for track in player_tracks:
            px = track.foot_position
            pitch_xy = self.calibrator.pixel_to_pitch(px)
            if pitch_xy:
                track.pitch_xy = pitch_xy

        if ball_tracks:
            best_ball = max(ball_tracks, key=lambda t: t.confidence)
            ball_pitch_xy = self.calibrator.pixel_to_pitch(best_ball.pixel_center)

        # 5. Possession - use pitch coords if available, else pixel-space fallback
        # Check if pitch-space distances are reasonable (< 5m = likely valid calibration)
        use_pitch = ball_pitch_xy is not None
        if use_pitch:
            min_dist = float('inf')
            for pt in player_tracks:
                if pt.pitch_xy:
                    dist = ((ball_pitch_xy[0]-pt.pitch_xy[0])**2 + (ball_pitch_xy[1]-pt.pitch_xy[1])**2)**0.5
                    min_dist = min(min_dist, dist)
                    if dist < 5:  # Close enough = calibration is valid
                        break
            else:
                if min_dist > 5:
                    use_pitch = False  # All players too far = bad calibration

        if not use_pitch and ball_tracks:
            # Pixel-space fallback: use ball pixel center vs player foot positions
            # For close-up indoor video, ~100px ≈ 1m is a reasonable approximation
            best_ball = max(ball_tracks, key=lambda t: t.confidence)
            ball_px = best_ball.pixel_center
            scale = 100.0  # pixels per meter approximation
            ball_pitch_xy = (ball_px[0] / scale, ball_px[1] / scale)
            for pt in player_tracks:
                pt.pitch_xy = (pt.foot_position[0] / scale, pt.foot_position[1] / scale)

        possession = self.possession_engine.update(player_tracks, ball_pitch_xy, timestamp_ms)

        # 6. Attack
        attack_result = self.attack_engine.update(
            possession.team_id, ball_pitch_xy, timestamp_ms
        )

        # 7. Event detection
        ball_visible = len(ball_tracks) > 0
        event_candidates = []

        corner = self.corner_detector.update(ball_pitch_xy, ball_visible, timestamp_ms)
        if corner:
            event_candidates.append(corner)

        goal = self.goal_detector.update(ball_pitch_xy, timestamp_ms, ball_visible)
        if goal:
            event_candidates.append(goal)

        card = self.card_detector.update(timestamp_ms)
        if card:
            event_candidates.append(card)

        # 8. Event fusion
        confirmed = self.fusion_engine.process(event_candidates)

        # 9. Update stats
        self.stats.update_from_events(confirmed)
        self.stats.update_possession(*self.possession_engine.get_possession_pct())
        self.stats.update_attacks(self.attack_engine.get_stats())
        self.stats.update_clock(timestamp_ms)

        return {
            "frame": self.frame_count,
            "timestamp_ms": timestamp_ms,
            "players_detected": len(player_tracks),
            "ball_detected": ball_visible,
            "ball_pitch_xy": ball_pitch_xy,
            "possession_team": possession.team_id,
            "possession_state": possession.state.value,
            "attack_team": self.attack_engine.current_attack.team_id if self.attack_engine.current_attack else None,
            "new_events": [e.event_type.value for e in confirmed],
            "stats": self.stats.snapshot()
        }

    def process_video(self, video_path: str, output_path: str = None,
                      progress_callback=None, max_frames: int = None) -> dict:
        """Process an entire video file.
        
        Args:
            video_path: Path to video file
            output_path: Optional path for annotated output video
            progress_callback: Optional callback(frame_num, total_frames, stats)
            max_frames: Maximum frames to process (None = all)
            
        Returns:
            Final match stats
        """
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Setup calibration
        self.setup_calibration(width, height)

        # Setup output video writer if needed
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))

        print(f"[Pipeline] Processing {video_path}: {total_frames} frames @ {self.fps}fps")
        print(f"[Pipeline] Resolution: {width}x{height}")

        frame_num = 0
        process_interval = max(1, int(self.fps / 10))  # Process ~10 fps

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames and frame_num >= max_frames:
                break

            timestamp_ms = frame_num * (1000 / self.fps)

            # Process every Nth frame for speed
            if frame_num % process_interval == 0:
                result = self.process_frame(frame, timestamp_ms)

                if progress_callback:
                    progress_callback(frame_num, total_frames, result)

                # Annotate frame if writing output
                if writer:
                    annotated = self._annotate_frame(frame, result)
                    writer.write(annotated)
            elif writer:
                writer.write(frame)

            frame_num += 1

        cap.release()
        if writer:
            writer.release()

        final_stats = self.stats.get_full_report()
        print(f"[Pipeline] Processing complete: {frame_num} frames")
        print(f"[Pipeline] Final stats: {self.stats.snapshot()}")

        return final_stats

    def _annotate_frame(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """Draw annotations on frame for visualization."""
        import cv2
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        scale = w / 1280  # Scale font for resolution

        # Draw stats overlay with semi-transparent background
        stats = result.get("stats", {})
        team_a = stats.get("team_a", {})
        team_b = stats.get("team_b", {})

        # Dark overlay for stats
        overlay = annotated.copy()
        cv2.rectangle(overlay, (10, 10), (int(500*scale), int(280*scale)), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)

        y_offset = int(40 * scale)
        font_scale = 0.7 * scale
        thick = max(1, int(2 * scale))
        gap = int(32 * scale)

        texts = [
            f"Team A: Goals={team_a.get('goals',0)} Poss={team_a.get('possession_pct',50)}%",
            f"  ATK={team_a.get('attacks',0)} DNG={team_a.get('dangerous_attacks',0)}",
            f"  Y={team_a.get('yellow_cards',0)} R={team_a.get('red_cards',0)} C={team_a.get('corners',0)}",
            f"Team B: Goals={team_b.get('goals',0)} Poss={team_b.get('possession_pct',50)}%",
            f"  ATK={team_b.get('attacks',0)} DNG={team_b.get('dangerous_attacks',0)}",
            f"  Y={team_b.get('yellow_cards',0)} R={team_b.get('red_cards',0)} C={team_b.get('corners',0)}",
            f"Possession: {result.get('possession_team','?')} ({result.get('possession_state','?')})",
            f"Clock: {stats.get('clock_display','00:00')}"
        ]

        for text in texts:
            cv2.putText(annotated, text, (int(20*scale), y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thick)
            y_offset += gap

        # Draw player/team tracks
        for track in self.tracker.get_active_tracks():
            if track.class_name == "player":
                x1, y1, x2, y2 = track.bbox.astype(int)
                color = (0, 0, 255) if track.team_id == "A" else (255, 200, 0)  # Red vs Yellow
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, max(1, int(2*scale)))
                label = f"{track.team_id}"
                cv2.putText(annotated, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, font_scale*0.6, color, thick)
            elif track.class_name == "ball":
                cx, cy = int(track.pixel_center[0]), int(track.pixel_center[1])
                cv2.circle(annotated, (cx, cy), int(15*scale), (0, 255, 255), -1)
                cv2.putText(annotated, "BALL", (cx+20, cy), cv2.FONT_HERSHEY_SIMPLEX, font_scale*0.5, (0, 255, 255), thick)

        return annotated

    def reset(self):
        """Reset all engines for a new match."""
        self.frame_count = 0
        self.possession_engine.reset()
        self.attack_engine.reset()
        self.corner_detector = CornerDetector(
            self.configs.get("rules", {}).get("corner", {}),
            self.configs.get("pitch", {})
        )
        self.goal_detector = GoalDetector(
            self.configs.get("rules", {}).get("goal", {}),
            self.configs.get("pitch", {})
        )
        self.fusion_engine.reset()
        self.stats.reset()
        self.team_classifier.reset()
