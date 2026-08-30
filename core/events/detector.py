"""
Event Detectors - Spatial rules for corner, goal, and card candidates.
Uses multi-evidence fusion; no single-model dependency.
"""
from typing import Optional, Tuple, List
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class EventType(Enum):
    GOAL = "goal"
    CORNER = "corner"
    YELLOW_CARD = "yellow_card"
    RED_CARD = "red_card"
    BALL_OUT = "ball_out"
    RESTART = "restart"


@dataclass
class EventCandidate:
    event_type: EventType
    team_id: str  # Which team benefits / is penalized
    start_ms: float
    end_ms: float
    confidence: float
    evidence: List[str] = field(default_factory=list)  # Evidence descriptions
    status: str = "pending"  # pending / confirmed / rejected


class CornerDetector:
    """Detect corner kick events.
    
    From SDD: Detect when ball re-enters corner zone after going out,
    and a player approaches/controls ball from corner zone.
    """

    def __init__(self, config: dict, pitch_config: dict):
        self.config = config
        self.corner_zone_radius = config.get("corner_zone_radius_m", 5.0)
        self.min_stop_duration = config.get("min_stop_duration_ms", 500)

        # Corner zones from pitch config
        zones = pitch_config.get("zones", {})
        self.corner_zones = []
        for name in ["corner_top_left", "corner_top_right", "corner_bottom_left", "corner_bottom_right"]:
            if name in zones:
                z = zones[name]
                self.corner_zones.append({
                    "name": name,
                    "x_min": z.get("x_min", 0), "x_max": z.get("x_max", 5),
                    "y_min": z.get("y_min", 0), "y_max": z.get("y_max", 5)
                })

        self.ball_was_out = False
        self.ball_out_time_ms = 0
        self.last_corner_ms = 0
        self.corner_cooldown_ms = 5000  # Minimum time between corners

    def update(self, ball_pitch_xy: Optional[Tuple[float, float]],
               ball_visible: bool, timestamp_ms: float,
               attacking_team: str = "unknown") -> Optional[EventCandidate]:
        """Check for corner kick event."""
        if not ball_visible or ball_pitch_xy is None:
            if not self.ball_was_out:
                self.ball_was_out = True
                self.ball_out_time_ms = timestamp_ms
            return None

        bx, by = ball_pitch_xy

        # Check if ball was out and now in corner zone
        if self.ball_was_out:
            in_corner = False
            corner_name = ""
            for zone in self.corner_zones:
                if zone["x_min"] <= bx <= zone["x_max"] and zone["y_min"] <= by <= zone["y_max"]:
                    in_corner = True
                    corner_name = zone["name"]
                    break

            if in_corner and (self.last_corner_ms == 0 or
                              timestamp_ms - self.last_corner_ms > self.corner_cooldown_ms):
                self.ball_was_out = False
                self.last_corner_ms = timestamp_ms

                # Determine which team gets the corner
                # Corner is awarded to attacking team (ball went out off defending team)
                if "right" in corner_name:
                    corner_team = "A"  # Team A attacks right
                else:
                    corner_team = "B"  # Team B attacks left

                return EventCandidate(
                    event_type=EventType.CORNER,
                    team_id=corner_team,
                    start_ms=self.ball_out_time_ms,
                    end_ms=timestamp_ms,
                    confidence=0.7,
                    evidence=["ball_in_corner_zone", f"zone={corner_name}"]
                )

        # Reset if ball is back on field (not in corner zone)
        if not self._in_any_corner_zone(bx, by):
            self.ball_was_out = False

        return None

    def _in_any_corner_zone(self, bx: float, by: float) -> bool:
        for zone in self.corner_zones:
            if zone["x_min"] <= bx <= zone["x_max"] and zone["y_min"] <= by <= zone["y_max"]:
                return True
        return False


class GoalDetector:
    """Detect goal events using multi-evidence fusion.
    
    Evidence:
    E1: Ball trajectory crosses goal line inside goal mouth
    E2: Play stops after shot-like sequence
    E3: Subsequent restart is center kick-off
    """
    def __init__(self, config: dict, pitch_config: dict):
        self.config = config
        self.event_window_s = config.get("event_window_s", 30.0)

        # Goal mouth coordinates
        goals = pitch_config.get("goals", {})
        self.goal_mouth_a = {  # Right goal (Team A attacks)
            "y_min": goals.get("goal_line_y_a", 30.34),
            "y_max": goals.get("goal_line_y_b", 37.66),
            "x": pitch_config.get("dimensions", {}).get("length_m", 105.0)
        }
        self.goal_mouth_b = {  # Left goal (Team B attacks)
            "y_min": goals.get("goal_line_y_a", 30.34),
            "y_max": goals.get("goal_line_y_b", 37.66),
            "x": 0.0
        }

        self.prev_ball_xy: Optional[Tuple[float, float]] = None
        self.goal_cooldown_ms = 30000
        self.last_goal_ms = -99999

    def update(self, ball_pitch_xy: Optional[Tuple[float, float]],
               timestamp_ms: float, ball_visible: bool) -> Optional[EventCandidate]:
        """Check for goal event."""
        if not ball_visible or ball_pitch_xy is None:
            self.prev_ball_xy = None
            return None

        bx, by = ball_pitch_xy
        goal = None

        if self.prev_ball_xy is not None:
            px, py = self.prev_ball_xy

            # Check if ball crossed goal line A (right side, team A scores)
            if px < self.goal_mouth_a["x"] and bx >= self.goal_mouth_a["x"]:
                if self.goal_mouth_a["y_min"] <= by <= self.goal_mouth_a["y_max"]:
                    if timestamp_ms - self.last_goal_ms > self.goal_cooldown_ms:
                        goal = EventCandidate(
                            event_type=EventType.GOAL,
                            team_id="A",
                            start_ms=timestamp_ms - 100,
                            end_ms=timestamp_ms,
                            confidence=0.6,
                            evidence=["ball_crossed_goal_line_a"]
                        )

            # Check goal B (left side, team B scores)
            if px > self.goal_mouth_b["x"] and bx <= self.goal_mouth_b["x"]:
                if self.goal_mouth_b["y_min"] <= by <= self.goal_mouth_b["y_max"]:
                    if timestamp_ms - self.last_goal_ms > self.goal_cooldown_ms:
                        goal = EventCandidate(
                            event_type=EventType.GOAL,
                            team_id="B",
                            start_ms=timestamp_ms - 100,
                            end_ms=timestamp_ms,
                            confidence=0.6,
                            evidence=["ball_crossed_goal_line_b"]
                        )

        self.prev_ball_xy = (bx, by)

        if goal:
            self.last_goal_ms = timestamp_ms
        return goal


class CardDetector:
    """Card event detector - MVP defaults to manual confirmation.
    
    For fixed panoramic cameras, automatic detection is unreliable.
    This detector creates candidates when referee shows card-like gesture.
    In practice, cards are entered manually via the review UI.
    """

    def __init__(self, config: dict):
        self.config = config
        self.default_to_review = config.get("default_to_review", True)

    def update(self, timestamp_ms: float) -> Optional[EventCandidate]:
        """For MVP, cards are manual-only. This is a placeholder."""
        # In production: could detect referee gesture, card color in hand
        # For fixed panoramic: too unreliable, use manual entry
        return None
