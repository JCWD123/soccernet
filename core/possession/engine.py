"""
Possession Engine - State machine for ball possession tracking.

Core idea from SDD: NOT frame-by-frame nearest distance switching,
but "candidate possessor + stability time + state machine".

States: CONTROLLED / CONTESTED / LOOSE
Transitions require stable candidate for switch_hold_ms before switching.
"""
import time
from typing import Optional, Tuple, List
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class PossessionState(Enum):
    CONTROLLED = "CONTROLLED"
    CONTESTED = "CONTESTED"
    LOOSE = "LOOSE"
    UNKNOWN = "UNKNOWN"


@dataclass
class PossessionUpdate:
    team_id: str  # "A" / "B" / "unknown"
    player_track_id: Optional[int]
    state: PossessionState
    confidence: float
    timestamp_ms: float


class PossessionEngine:
    """State-machine based possession tracking.
    
    Rules (from SDD):
    1. Find nearest player to ball in pitch space
    2. If within max_distance -> candidate team
    3. Candidate must persist for switch_hold_ms before switching
    4. If no player within contested_distance -> LOOSE
    5. If player within contested_distance but not max_distance -> CONTESTED
    """

    def __init__(self, config: dict):
        self.config = config
        self.max_distance = config.get("max_player_ball_distance_m", 2.3)
        self.switch_hold_ms = config.get("switch_hold_ms", 600)
        self.unknown_timeout_ms = config.get("unknown_timeout_ms", 1200)
        self.contested_distance = config.get("contested_distance_m", 4.0)

        # Current state
        self.current_team: str = "unknown"
        self.current_player_id: Optional[int] = None
        self.current_state: PossessionState = PossessionState.UNKNOWN
        self.state_start_ms: float = 0

        # Candidate for switch
        self.candidate_team: str = "unknown"
        self.candidate_player_id: Optional[int] = None
        self.candidate_start_ms: float = 0

        # Accumulated time per team
        self.team_a_controlled_ms: float = 0
        self.team_b_controlled_ms: float = 0
        self.last_update_ms: float = 0

        # History for debugging
        self.history: List[PossessionUpdate] = []

    def update(self, players: list, ball_pitch_xy: Optional[Tuple[float, float]],
               timestamp_ms: float) -> PossessionUpdate:
        """Update possession state given current players and ball position.
        
        Args:
            players: List of Track objects with pitch_xy and team_id
            ball_pitch_xy: Ball position in pitch coordinates (meters)
            timestamp_ms: Current timestamp in milliseconds
            
        Returns:
            PossessionUpdate with current state
        """
        dt = timestamp_ms - self.last_update_ms if self.last_update_ms > 0 else 0
        self.last_update_ms = timestamp_ms

        # Accumulate time for current possessor
        self._accumulate_time(dt)

        # Find nearest player to ball
        if ball_pitch_xy is None:
            return self._make_update("unknown", None, PossessionState.UNKNOWN, 0.0, timestamp_ms)

        nearest_player, distance = self._find_nearest_player(players, ball_pitch_xy)

        if nearest_player is None:
            # No players detected
            if timestamp_ms - self.state_start_ms > self.unknown_timeout_ms:
                self._set_state("unknown", None, PossessionState.UNKNOWN, timestamp_ms)
            return self._make_update("unknown", None, self.current_state, 0.0, timestamp_ms)

        candidate_team = nearest_player.team_id
        candidate_pid = nearest_player.track_id

        # Determine state based on distance
        if distance <= self.max_distance:
            new_state = PossessionState.CONTROLLED
        elif distance <= self.contested_distance:
            new_state = PossessionState.CONTESTED
        else:
            new_state = PossessionState.LOOSE

        # State machine transitions
        # Special case: first possession (from unknown) - establish immediately
        if self.current_team == "unknown" and new_state == PossessionState.CONTROLLED:
            self._set_state(candidate_team, candidate_pid, PossessionState.CONTROLLED, timestamp_ms)
            return self._make_update(candidate_team, candidate_pid, new_state, 0.8, timestamp_ms)

        if candidate_team == self.current_team and new_state == PossessionState.CONTROLLED:
            # Same team, controlled -> extend possession
            self.candidate_start_ms = timestamp_ms
            self.current_player_id = candidate_pid
            self.current_state = PossessionState.CONTROLLED
            return self._make_update(candidate_team, candidate_pid, new_state, 0.8, timestamp_ms)

        if candidate_team != self.current_team:
            # Different team candidate
            if self.candidate_team != candidate_team:
                # New candidate
                self.candidate_team = candidate_team
                self.candidate_player_id = candidate_pid
                self.candidate_start_ms = timestamp_ms
                return self._make_update(self.current_team, self.current_player_id,
                                        PossessionState.CONTESTED, 0.3, timestamp_ms)

            # Check if candidate has persisted long enough
            hold_duration = timestamp_ms - self.candidate_start_ms
            if hold_duration >= self.switch_hold_ms and new_state == PossessionState.CONTROLLED:
                # Switch possession
                self._set_state(candidate_team, candidate_pid, PossessionState.CONTROLLED, timestamp_ms)
                return self._make_update(candidate_team, candidate_pid,
                                        PossessionState.CONTROLLED, 0.7, timestamp_ms)
            else:
                # Still contested/waiting
                return self._make_update(self.current_team, self.current_player_id,
                                        PossessionState.CONTESTED, 0.4, timestamp_ms)

        # Same team but not controlled (contested or loose)
        if new_state == PossessionState.LOOSE:
            if timestamp_ms - self.state_start_ms > self.unknown_timeout_ms:
                self._set_state("unknown", None, PossessionState.LOOSE, timestamp_ms)
            return self._make_update(self.current_team, self.current_player_id,
                                    PossessionState.LOOSE, 0.2, timestamp_ms)

        return self._make_update(self.current_team, self.current_player_id,
                                new_state, 0.5, timestamp_ms)

    def _find_nearest_player(self, players: list,
                            ball_xy: Tuple[float, float]) -> Tuple[Optional[object], float]:
        """Find nearest player to ball in pitch space."""
        if not players:
            return None, float("inf")

        bx, by = ball_xy
        nearest = None
        min_dist = float("inf")

        for p in players:
            if p.pitch_xy is None:
                continue
            px, py = p.pitch_xy
            dist = np.sqrt((bx - px) ** 2 + (by - py) ** 2)
            if dist < min_dist:
                min_dist = dist
                nearest = p

        return nearest, min_dist

    def _set_state(self, team: str, player_id: Optional[int],
                   state: PossessionState, timestamp_ms: float):
        """Set new possession state."""
        self.current_team = team
        self.current_player_id = player_id
        self.current_state = state
        self.state_start_ms = timestamp_ms
        self.candidate_team = "unknown"

    def _accumulate_time(self, dt_ms: float):
        """Accumulate possession time for current team."""
        if self.current_state == PossessionState.CONTROLLED:
            if self.current_team == "A":
                self.team_a_controlled_ms += dt_ms
            elif self.current_team == "B":
                self.team_b_controlled_ms += dt_ms

    def _make_update(self, team, pid, state, conf, ts) -> PossessionUpdate:
        """Create and record a PossessionUpdate."""
        update = PossessionUpdate(team, pid, state, conf, ts)
        self.history.append(update)
        return update

    def get_possession_pct(self) -> Tuple[float, float]:
        """Get current possession percentages.
        
        Returns:
            (team_a_pct, team_b_pct) - only CONTROLLED time counted
        """
        total = self.team_a_controlled_ms + self.team_b_controlled_ms
        if total < 1e-7:
            return (50.0, 50.0)
        a_pct = self.team_a_controlled_ms / total * 100
        return (round(a_pct, 1), round(100 - a_pct, 1))

    def reset(self):
        """Reset for new match."""
        self.current_team = "unknown"
        self.current_player_id = None
        self.current_state = PossessionState.UNKNOWN
        self.state_start_ms = 0
        self.candidate_team = "unknown"
        self.candidate_player_id = None
        self.candidate_start_ms = 0
        self.team_a_controlled_ms = 0
        self.team_b_controlled_ms = 0
        self.last_update_ms = 0
        self.history.clear()
