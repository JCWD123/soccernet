"""
Pitch Calibration - Homography from image pixels to pitch coordinates.
For fixed cameras: one-time calibration using 8-12 field line intersection points.
"""
import numpy as np
import cv2
from typing import List, Tuple, Optional
import json
import os


class PitchCalibrator:
    """Compute and apply homography transformation between image and pitch."""

    def __init__(self, config: dict, pitch_config: dict):
        self.config = config
        self.pitch_config = pitch_config
        self.H: Optional[np.ndarray] = None  # image -> pitch
        self.H_inv: Optional[np.ndarray] = None  # pitch -> image
        self.image_points: List[Tuple[float, float]] = []
        self.pitch_points: List[Tuple[float, float]] = []
        self.is_calibrated = False

        # Try to load existing calibration
        self._load_calibration()

    def calibrate(self, image_points: List[Tuple[float, float]],
                  pitch_points: List[Tuple[float, float]]):
        """Compute homography from corresponding point pairs.
        
        Args:
            image_points: List of (px, py) pixel coordinates
            pitch_points: List of (mx, my) pitch coordinates in meters
        """
        if len(image_points) < 4:
            raise ValueError("Need at least 4 point pairs for homography")

        src = np.array(image_points, dtype=np.float32)
        dst = np.array(pitch_points, dtype=np.float32)

        self.H, mask = cv2.findHomography(src, dst, method=cv2.RANSAC)
        self.H_inv, _ = cv2.findHomography(dst, src, method=cv2.RANSAC)

        self.image_points = image_points
        self.pitch_points = pitch_points
        self.is_calibrated = True

        print(f"[Calibrator] Homography computed from {len(image_points)} points")

    def pixel_to_pitch(self, pixel_xy: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """Convert image pixel coordinates to pitch meters.
        
        Args:
            pixel_xy: (px, py) in image coordinates
            
        Returns:
            (mx, my) in pitch coordinates, or None if not calibrated
        """
        if not self.is_calibrated:
            return None

        pt = np.array([[[pixel_xy[0], pixel_xy[1]]]], dtype=np.float32)
        result = cv2.perspectiveTransform(pt, self.H)
        return (float(result[0, 0, 0]), float(result[0, 0, 1]))

    def pitch_to_pixel(self, pitch_xy: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """Convert pitch coordinates back to image pixels (for visualization)."""
        if not self.is_calibrated:
            return None

        pt = np.array([[[pitch_xy[0], pitch_xy[1]]]], dtype=np.float32)
        result = cv2.perspectiveTransform(pt, self.H_inv)
        return (float(result[0, 0, 0]), float(result[0, 0, 1]))

    def batch_pixel_to_pitch(self, pixel_coords: np.ndarray) -> Optional[np.ndarray]:
        """Batch convert pixel coordinates to pitch coordinates.
        
        Args:
            pixel_coords: Nx2 array of (px, py)
            
        Returns:
            Nx2 array of (mx, my) in pitch meters
        """
        if not self.is_calibrated:
            return None

        pts = pixel_coords.reshape(1, -1, 2).astype(np.float32)
        result = cv2.perspectiveTransform(pts, self.H)
        return result.reshape(-1, 2)

    def _load_calibration(self):
        """Try to load calibration from file."""
        cal_file = os.path.join("data", "calibration.json")
        if os.path.exists(cal_file):
            try:
                with open(cal_file, 'r') as f:
                    data = json.load(f)
                self.H = np.array(data["H_matrix"], dtype=np.float64)
                self.H_inv = np.array(data["H_inv"], dtype=np.float64)
                self.image_points = [tuple(p) for p in data.get("image_points", [])]
                self.pitch_points = [tuple(p) for p in data.get("pitch_points", [])]
                self.is_calibrated = True
                print("[Calibrator] Loaded calibration from file")
            except Exception as e:
                print(f"[Calibrator] Could not load calibration: {e}")

    def save_calibration(self):
        """Save calibration to file."""
        if not self.is_calibrated:
            return
        os.makedirs("data", exist_ok=True)
        cal_file = os.path.join("data", "calibration.json")
        data = {
            "H_matrix": self.H.tolist(),
            "H_inv": self.H_inv.tolist(),
            "image_points": [list(p) for p in self.image_points],
            "pitch_points": [list(p) for p in self.pitch_points]
        }
        with open(cal_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[Calibrator] Saved calibration to {cal_file}")

    def create_default_calibration(self, frame_width: int, frame_height: int):
        """Create a default/identity-like calibration for testing.
        Maps image coordinates to pitch coordinates linearly.
        This is a rough approximation - real calibration needs clicked points.
        """
        pitch_w = self.pitch_config.get("dimensions", {}).get("length_m", 105.0)
        pitch_h = self.pitch_config.get("dimensions", {}).get("width_m", 68.0)

        # Map image corners to pitch corners (approximate)
        margin_x = frame_width * 0.05
        margin_y = frame_height * 0.1
        image_points = [
            (margin_x, margin_y),  # top-left -> pitch (0, 0)
            (frame_width - margin_x, margin_y),  # top-right -> pitch (105, 0)
            (frame_width - margin_x, frame_height - margin_y),  # bottom-right -> pitch (105, 68)
            (margin_x, frame_height - margin_y),  # bottom-left -> pitch (0, 68)
            # Add center line points for better accuracy
            (frame_width / 2, margin_y),  # center top -> pitch (52.5, 0)
            (frame_width / 2, frame_height - margin_y),  # center bottom -> pitch (52.5, 68)
            (margin_x, frame_height / 2),  # left center -> pitch (0, 34)
            (frame_width - margin_x, frame_height / 2),  # right center -> pitch (105, 34)
        ]
        pitch_points = [
            (0, 0), (pitch_w, 0), (pitch_w, pitch_h), (0, pitch_h),
            (pitch_w / 2, 0), (pitch_w / 2, pitch_h),
            (0, pitch_h / 2), (pitch_w, pitch_h / 2),
        ]

        self.calibrate(image_points, pitch_points)
        print("[Calibrator] Created default calibration (approximate)")


def get_zone(pitch_xy: Tuple[float, float], pitch_config: dict) -> str:
    """Determine which zone a pitch coordinate falls in."""
    x, y = pitch_xy
    zones = pitch_config.get("zones", {})

    for zone_name, zone_def in zones.items():
        x_min = zone_def.get("x_min", float("-inf"))
        x_max = zone_def.get("x_max", float("inf"))
        y_min = zone_def.get("y_min", float("-inf"))
        y_max = zone_def.get("y_max", float("inf"))
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return zone_name

    return "field"  # On the field but not in any special zone
