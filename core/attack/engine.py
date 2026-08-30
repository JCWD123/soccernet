"""
Attack & Dangerous Attack Engine.

From SDD:
- ATTACK: team has CONTROLLED possession + ball/player enters attacking half
  + forward progress >= min_progress OR ball enters final third
- DANGEROUS: ATTACK is active + (ball in penalty area OR danger zone OR shot/cross detected)
- Same possession chain = 1 attack (deduplicate)
- One attack -> at most one dangerous_attack count (configurable)
"""
from typing import Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import numpy as np


class AttackPhase(Enum):
    NONE = "NONE"
    BUILD_UP = "BUILD_UP"
    ATTACK = "ATTACK"
    DANGEROUS = "DANGEROUS"


@dataclass
class AttackChain:
    attack_id: int
    team_id: str
    start_ms: float
    end_ms: Optional[float] = None
    phase: AttackPhase = AttackPhase.BUILD_UP
    is_dangerous: bool = False
    max_x_progress: float = 0.0  # max x coordinate reached
    start_x: float = 0.0


class AttackEngine:
    """Track attack chains per team and classify as attack/dangerous."""

    def __init__(self, config: dict, danger_config: dict, pitch_config: dict):
        self.config = config
        self.danger_config = danger_config
        self.pitch_config = pitch_config

        self.require_attacking_half = config.get("require_attacking_half", True)
        self.min_forward_progress = config.get("min_forward_progress_m", 8.0)
        self.end_on_possession_loss = config.get("end_on_possession_loss", True)
        self.timeout_ms = config.get("timeout_without_progress_ms", 15000)
        self.cooldown_ms = config.get("attack_cooldown_ms", 3000)

        # Danger config
        self.one_per_attack = danger_config.get("one_per_attack", True)
        self.danger_distance = danger_config.get("final_distance_to_goal_m", 30.0)
        self.penalty_is_danger = danger_config.get("penalty_area_is_dangerous", True)
        self.central_channel_half = danger_config.get("central_channel_half_width_m", 15.0)

        # Pitch dimensions
        self.pitch_length = pitch_config.get("dimensions", {}).get("length_m", 105.0)
        self.pitch_width = pitch_config.get("dimensions", {}).get("width_m", 68.0)
        self.center_x = self.pitch_length / 2

        # Current state
        self.current_attack: Optional[AttackChain] = None
        self.attack_counter = 0
        self.last_attack_end_ms: float = 0

        # Stats per team
        self.team_a_attacks = 0
        self.team_b_attacks = 0
        self.team_a_dangerous = 0
        self.team_b_dangerous = 0

        # History
        self.completed_attacks: List[AttackChain] = []

    def update(self, possession_team: str, ball_pitch_xy: Optional[Tuple[float, float]],
               timestamp_ms: float, attacking_direction: dict = None) -> Optional[AttackChain]:
        """Update attack state.
        
        Args:
            possession_team: Current possession team ("A" / "B" / "unknown")
            ball_pitch_xy: Ball position in pitch coords (meters)
            timestamp_ms: Current timestamp
            attacking_direction: Dict {team_a: "right"/"left", team_b: ...}
            
        Returns:
            Completed attack if one just ended, else None
        """
        if ball_pitch_xy is None:
            return self._check_timeout(timestamp_ms)

        bx, by = ball_pitch_xy

        # Determine if ball is in attacking half for each team
        # Default: Team A attacks right (x > 52.5), Team B attacks left (x < 52.5)
        if attacking_direction is None:
            team_a_attacking = bx > self.center_x
            team_b_attacking = bx < self.center_x
        else:
            team_a_attacking = (bx > self.center_x) if attacking_direction.get("team_a") == "right" else (bx < self.center_x)
            team_b_attacking = not team_a_attacking

        in_attacking_half = (possession_team == "A" and team_a_attacking) or                            (possession_team == "B" and team_b_attacking)

        # Check if in final third / danger zone
        is_final_third = self._is_final_third(bx, possession_team)
        is_danger_zone = self._is_danger_zone(bx, by, possession_team)
        is_penalty_area = self._is_penalty_area(bx, by, possession_team)

        # Forward progress (for team A: higher x = more forward)
        if possession_team == "A":
            progress_from_center = bx - self.center_x
        elif possession_team == "B":
            progress_from_center = self.center_x - bx
        else:
            progress_from_center = 0

        # Attack chain management
        if self.current_attack is not None:
            attack = self.current_attack

            # Check if same possession chain
            if possession_team == attack.team_id:
                # Update progress
                attack.max_x_progress = max(attack.max_x_progress, progress_from_center)
                attack.end_ms = timestamp_ms

                # Check phase upgrades
                if attack.phase == AttackPhase.BUILD_UP:
                    if in_attacking_half and (progress_from_center >= self.min_forward_progress or is_final_third):
                        attack.phase = AttackPhase.ATTACK

                if attack.phase in (AttackPhase.ATTACK, AttackPhase.BUILD_UP):
                    if is_danger_zone or is_penalty_area:
                        attack.phase = AttackPhase.DANGEROUS
                        if not attack.is_dangerous:
                            attack.is_dangerous = True
                            if possession_team == "A":
                                self.team_a_dangerous += 1
                            else:
                                self.team_b_dangerous += 1

                # Check timeout
                if timestamp_ms - attack.end_ms > self.timeout_ms:
                    return self._end_attack(timestamp_ms)

            else:
                # Possession changed -> end current attack
                completed = self._end_attack(timestamp_ms)

                # Start new attack if new team has possession
                if possession_team in ("A", "B"):
                    self._start_attack(possession_team, progress_from_center, timestamp_ms)

                return completed

        else:
            # No current attack - check if we should start one
            if possession_team in ("A", "B") and in_attacking_half:
                # Cooldown only applies after a previous attack has ended
                cooldown_ok = (self.last_attack_end_ms == 0) or \
                              (timestamp_ms - self.last_attack_end_ms > self.cooldown_ms)
                if cooldown_ok:
                    self._start_attack(possession_team, progress_from_center, timestamp_ms)

        return None

    def _start_attack(self, team_id: str, progress: float, timestamp_ms: float):
        """Start a new attack chain."""
        self.attack_counter += 1
        self.current_attack = AttackChain(
            attack_id=self.attack_counter,
            team_id=team_id,
            start_ms=timestamp_ms,
            end_ms=timestamp_ms,
            phase=AttackPhase.BUILD_UP,
            max_x_progress=progress,
            start_x=progress
        )

    def _end_attack(self, timestamp_ms: float) -> Optional[AttackChain]:
        """End current attack chain."""
        if self.current_attack is None:
            return None

        attack = self.current_attack
        attack.end_ms = timestamp_ms

        # Count as attack if it reached ATTACK or DANGEROUS phase
        if attack.phase in (AttackPhase.ATTACK, AttackPhase.DANGEROUS):
            if attack.team_id == "A":
                self.team_a_attacks += 1
            else:
                self.team_b_attacks += 1

        self.completed_attacks.append(attack)
        self.last_attack_end_ms = timestamp_ms
        self.current_attack = None
        return attack

    def _check_timeout(self, timestamp_ms: float) -> Optional[AttackChain]:
        """Check if current attack has timed out."""
        if self.current_attack and timestamp_ms - self.current_attack.end_ms > self.timeout_ms:
            return self._end_attack(timestamp_ms)
        return None

    def _is_final_third(self, bx: float, team_id: str) -> bool:
        """Check if ball is in final third."""
        if team_id == "A":
            return bx >= self.pitch_length - self.danger_distance
        else:
            return bx <= self.danger_distance

    def _is_danger_zone(self, bx: float, by: float, team_id: str) -> bool:
        """Check if ball is in danger zone."""
        center_y = self.pitch_width / 2
        in_central = abs(by - center_y) <= self.central_channel_half
        in_distance = self._is_final_third(bx, team_id)
        return in_central and in_distance

    def _is_penalty_area(self, bx: float, by: float, team_id: str) -> bool:
        """Check if ball is in penalty area."""
        if not self.penalty_is_danger:
            return False
        zones = self.pitch_config.get("zones", {})
        if team_id == "A":
            zone = zones.get("penalty_area_a", {})
        else:
            zone = zones.get("penalty_area_b", {})

        x_min = zone.get("x_min", 0)
        x_max = zone.get("x_max", 0)
        y_min = zone.get("y_min", 0)
        y_max = zone.get("y_max", 0)
        return x_min <= bx <= x_max and y_min <= by <= y_max

    def get_stats(self) -> dict:
        """Get attack statistics."""
        return {
            "team_a": {"attacks": self.team_a_attacks, "dangerous_attacks": self.team_a_dangerous},
            "team_b": {"attacks": self.team_b_attacks, "dangerous_attacks": self.team_b_dangerous}
        }

    def reset(self):
        """Reset for new match."""
        self.current_attack = None
        self.attack_counter = 0
        self.last_attack_end_ms = 0
        self.team_a_attacks = 0
        self.team_b_attacks = 0
        self.team_a_dangerous = 0
        self.team_b_dangerous = 0
        self.completed_attacks.clear()
