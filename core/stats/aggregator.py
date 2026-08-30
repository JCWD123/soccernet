"""
Stats Aggregator - accumulates all statistics into a unified match report.

Tracks: goals, yellow_cards, red_cards, corners, attacks, dangerous_attacks, possession_pct
All stats are per-team and can be queried via API or WebSocket.
"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from ..events.detector import EventType, EventCandidate
import time


@dataclass
class TeamStats:
    goals: int = 0
    yellow_cards: int = 0
    red_cards: int = 0
    corners: int = 0
    attacks: int = 0
    dangerous_attacks: int = 0
    possession_pct: float = 50.0


@dataclass
class MatchStats:
    match_id: str = ""
    clock_ms: float = 0
    period: str = "1H"
    team_a: TeamStats = field(default_factory=TeamStats)
    team_b: TeamStats = field(default_factory=TeamStats)
    total_events: int = 0
    last_update_ms: float = 0


class StatsAggregator:
    """Aggregate match statistics from all engines."""

    def __init__(self):
        self.stats = MatchStats()
        self.event_log: List[dict] = []  # Full event history for audit

    def update_from_events(self, confirmed_events: List[EventCandidate]):
        """Update stats from newly confirmed events."""
        for event in confirmed_events:
            self._apply_event(event)

    def _apply_event(self, event: EventCandidate):
        """Apply a single confirmed event to stats."""
        team_stats = self.stats.team_a if event.team_id == "A" else self.stats.team_b

        if event.event_type == EventType.GOAL:
            team_stats.goals += 1
        elif event.event_type == EventType.YELLOW_CARD:
            team_stats.yellow_cards += 1
        elif event.event_type == EventType.RED_CARD:
            team_stats.red_cards += 1
        elif event.event_type == EventType.CORNER:
            team_stats.corners += 1

        self.stats.total_events += 1
        self.event_log.append({
            "type": event.event_type.value,
            "team": event.team_id,
            "time_ms": event.start_ms,
            "confidence": event.confidence,
            "evidence": event.evidence
        })

    def update_possession(self, a_pct: float, b_pct: float):
        """Update possession percentages."""
        self.stats.team_a.possession_pct = a_pct
        self.stats.team_b.possession_pct = b_pct

    def update_attacks(self, attack_stats: dict):
        """Update attack counts from attack engine."""
        self.stats.team_a.attacks = attack_stats.get("team_a", {}).get("attacks", 0)
        self.stats.team_a.dangerous_attacks = attack_stats.get("team_a", {}).get("dangerous_attacks", 0)
        self.stats.team_b.attacks = attack_stats.get("team_b", {}).get("attacks", 0)
        self.stats.team_b.dangerous_attacks = attack_stats.get("team_b", {}).get("dangerous_attacks", 0)

    def update_clock(self, clock_ms: float, period: str = None):
        """Update match clock."""
        self.stats.clock_ms = clock_ms
        if period:
            self.stats.period = period
        self.stats.last_update_ms = time.time() * 1000

    def snapshot(self) -> dict:
        """Get current stats as JSON-serializable dict."""
        return {
            "match_id": self.stats.match_id,
            "clock_ms": self.stats.clock_ms,
            "clock_display": self._format_clock(self.stats.clock_ms),
            "period": self.stats.period,
            "team_a": {
                "goals": self.stats.team_a.goals,
                "yellow_cards": self.stats.team_a.yellow_cards,
                "red_cards": self.stats.team_a.red_cards,
                "corners": self.stats.team_a.corners,
                "attacks": self.stats.team_a.attacks,
                "dangerous_attacks": self.stats.team_a.dangerous_attacks,
                "possession_pct": self.stats.team_a.possession_pct
            },
            "team_b": {
                "goals": self.stats.team_b.goals,
                "yellow_cards": self.stats.team_b.yellow_cards,
                "red_cards": self.stats.team_b.red_cards,
                "corners": self.stats.team_b.corners,
                "attacks": self.stats.team_b.attacks,
                "dangerous_attacks": self.stats.team_b.dangerous_attacks,
                "possession_pct": self.stats.team_b.possession_pct
            },
            "total_events": self.stats.total_events,
            "recent_events": self.event_log[-10:] if self.event_log else []
        }

    def _format_clock(self, ms: float) -> str:
        """Format milliseconds as MM:SS."""
        seconds = int(ms / 1000)
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes:02d}:{secs:02d}"

    def get_full_report(self) -> dict:
        """Get full match report including all events."""
        report = self.snapshot()
        report["all_events"] = self.event_log
        return report

    def reset(self):
        """Reset for new match."""
        self.stats = MatchStats()
        self.event_log.clear()
