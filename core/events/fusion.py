"""
Event Fusion Engine - merge, score, and confirm/reject event candidates.

From SDD:
1. Merge same-type candidates in temporal window
2. Compute weighted confidence
3. If confidence >= AUTO_CONFIRM -> confirmed
4. If REVIEW_LOW <= confidence < AUTO_CONFIRM -> pending_review
5. Else discard
6. Manual decision is immutable audit evidence
"""
from typing import List, Dict, Optional
from .detector import EventCandidate, EventType
import time


class EventFusionEngine:
    """Fuse event candidates and produce confirmed events."""

    def __init__(self, config: dict):
        self.config = config
        self.goal_auto = config.get("goal_auto_confirm", 0.90)
        self.corner_auto = config.get("corner_auto_confirm", 0.85)
        self.card_auto = config.get("card_auto_confirm", 0.95)

        # Temporal merge window (ms)
        self.merge_window_ms = {
            EventType.GOAL: 30000,
            EventType.CORNER: 5000,
            EventType.YELLOW_CARD: 5000,
            EventType.RED_CARD: 5000,
        }

        # Pending and confirmed events
        self.pending_events: List[EventCandidate] = []
        self.confirmed_events: List[EventCandidate] = []
        self.rejected_events: List[EventCandidate] = []

    def process(self, candidates: List[EventCandidate]) -> List[EventCandidate]:
        """Process new candidates: merge, score, confirm/reject.
        
        Returns:
            List of newly confirmed events
        """
        newly_confirmed = []

        for candidate in candidates:
            # Check if this merges with an existing pending event
            merged = self._try_merge(candidate)

            if not merged:
                # New candidate
                self.pending_events.append(candidate)

        # Evaluate pending events
        still_pending = []
        for event in self.pending_events:
            auto_threshold = self._get_threshold(event.event_type)

            if event.confidence >= auto_threshold:
                event.status = "confirmed"
                self.confirmed_events.append(event)
                newly_confirmed.append(event)
            elif event.confidence >= auto_threshold * 0.5:
                # Keep as pending for review
                still_pending.append(event)
            else:
                # Too low confidence -> discard
                event.status = "rejected"
                self.rejected_events.append(event)

        self.pending_events = still_pending
        return newly_confirmed

    def _try_merge(self, candidate: EventCandidate) -> bool:
        """Try to merge candidate with existing pending event."""
        window = self.merge_window_ms.get(candidate.event_type, 5000)

        for pending in self.pending_events:
            if pending.event_type != candidate.event_type:
                continue
            if pending.team_id != candidate.team_id:
                continue
            if abs(pending.end_ms - candidate.start_ms) <= window:
                # Merge
                pending.evidence.extend(candidate.evidence)
                pending.confidence = min(1.0, pending.confidence + candidate.confidence * 0.3)
                pending.end_ms = max(pending.end_ms, candidate.end_ms)
                return True

        return False

    def _get_threshold(self, event_type: EventType) -> float:
        """Get auto-confirm threshold for event type."""
        thresholds = {
            EventType.GOAL: self.goal_auto,
            EventType.CORNER: self.corner_auto,
            EventType.YELLOW_CARD: self.card_auto,
            EventType.RED_CARD: self.card_auto,
        }
        return thresholds.get(event_type, 0.85)

    def manual_confirm(self, event_id: int, decision: str, reviewer: str = "operator"):
        """Manual confirm/reject an event. Immutable audit trail."""
        for event in self.pending_events:
            if event.event_type.value + str(event.start_ms) == str(event_id):
                event.status = decision
                if decision == "confirmed":
                    self.confirmed_events.append(event)
                else:
                    self.rejected_events.append(event)
                self.pending_events.remove(event)
                return True
        return False

    def manual_add_event(self, event_type: EventType, team_id: str,
                         timestamp_ms: float) -> EventCandidate:
        """Manual event entry (e.g., card added by operator)."""
        event = EventCandidate(
            event_type=event_type,
            team_id=team_id,
            start_ms=timestamp_ms,
            end_ms=timestamp_ms,
            confidence=1.0,
            evidence=["manual_entry"],
            status="confirmed"
        )
        self.confirmed_events.append(event)
        return event

    def get_pending(self) -> List[EventCandidate]:
        return self.pending_events

    def get_confirmed(self) -> List[EventCandidate]:
        return self.confirmed_events

    def get_all_events(self) -> List[EventCandidate]:
        """Get all events (confirmed + pending)."""
        return self.confirmed_events + self.pending_events

    def reset(self):
        self.pending_events.clear()
        self.confirmed_events.clear()
        self.rejected_events.clear()
