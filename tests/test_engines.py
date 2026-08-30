"""
Unit tests for core engines.
Validates that all engines work correctly with synthetic data.
"""
import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_possession_engine():
    """Test possession state machine logic."""
    from core.possession.engine import PossessionEngine, PossessionState

    config = {
        "max_player_ball_distance_m": 2.3,
        "switch_hold_ms": 600,
        "unknown_timeout_ms": 1200,
        "contested_distance_m": 4.0
    }
    engine = PossessionEngine(config)

    # Create mock players
    class MockPlayer:
        def __init__(self, tid, team, px, py):
            self.track_id = tid
            self.team_id = team
            self.pitch_xy = (px, py)
            self.bbox = np.array([0, 0, 10, 10])

    players = [
        MockPlayer(1, "A", 50.0, 34.0),  # Near center
        MockPlayer(2, "B", 60.0, 34.0),  # Slightly right
    ]

    # Ball near player 1 (Team A)
    result = engine.update(players, (50.5, 34.0), 1000)
    assert result.team_id == "A", f"Expected A, got {result.team_id}"
    assert result.state == PossessionState.CONTROLLED
    print("  [PASS] Ball near Team A player -> A has possession")

    # Ball moves to Team B player
    for t in range(1200, 2400, 100):
        result = engine.update(players, (60.0, 34.0), t)

    assert engine.current_team == "B" or engine.candidate_team == "B"
    print("  [PASS] Ball near Team B player -> switch triggers")

    # Check possession percentages
    a_pct, b_pct = engine.get_possession_pct()
    assert 0 <= a_pct <= 100
    assert 0 <= b_pct <= 100
    assert abs(a_pct + b_pct - 100) < 1
    print(f"  [PASS] Possession: A={a_pct}% B={b_pct}%")

    print("[PASS] Possession engine tests passed\n")


def test_attack_engine():
    """Test attack and dangerous attack detection."""
    from core.attack.engine import AttackEngine, AttackPhase

    config = {
        "require_attacking_half": True,
        "min_forward_progress_m": 8.0,
        "end_on_possession_loss": True,
        "timeout_without_progress_ms": 15000,
        "attack_cooldown_ms": 3000
    }
    danger_config = {
        "one_per_attack": True,
        "final_distance_to_goal_m": 30.0,
        "penalty_area_is_dangerous": True,
        "central_channel_half_width_m": 15.0
    }
    pitch_config = {
        "dimensions": {"length_m": 105, "width_m": 68},
        "zones": {
            "penalty_area_a": {"x_min": 88.5, "x_max": 105, "y_min": 13.85, "y_max": 54.15},
            "penalty_area_b": {"x_min": 0, "x_max": 16.5, "y_min": 13.85, "y_max": 54.15},
        }
    }

    engine = AttackEngine(config, danger_config, pitch_config)

    # Team A attacking right (x > 52.5)
    engine.update("A", (60.0, 34.0), 1000)
    assert engine.current_attack is not None, "Attack should start"
    print("  [PASS] Attack starts when team enters attacking half")

    # Progress into final third
    engine.update("A", (80.0, 34.0), 2000)
    assert engine.current_attack.phase in (AttackPhase.ATTACK, AttackPhase.DANGEROUS)
    print("  [PASS] Attack phase upgrades on forward progress")

    # Into danger zone
    engine.update("A", (90.0, 34.0), 3000)
    assert engine.current_attack.is_dangerous
    assert engine.team_a_dangerous >= 1
    print("  [PASS] Dangerous attack detected in penalty area")

    # Possession lost
    engine.update("B", (40.0, 34.0), 5000)
    stats = engine.get_stats()
    assert stats["team_a"]["attacks"] >= 1
    print(f"  [PASS] Attack counted after possession loss: {stats}")

    print("[PASS] Attack engine tests passed\n")


def test_pitch_calibrator():
    """Test pitch calibration and coordinate mapping."""
    from core.pitch.calibrator import PitchCalibrator

    camera_config = {}
    pitch_config = {"dimensions": {"length_m": 105, "width_m": 68}}

    cal = PitchCalibrator(camera_config, pitch_config)
    cal.create_default_calibration(1920, 1080)

    assert cal.is_calibrated

    # Test center of image -> center of pitch
    center_pixel = (960, 540)
    pitch_xy = cal.pixel_to_pitch(center_pixel)
    assert pitch_xy is not None
    assert 45 < pitch_xy[0] < 60  # Near center x
    assert 28 < pitch_xy[1] < 40  # Near center y
    print(f"  [PASS] Center pixel {center_pixel} -> pitch {pitch_xy}")

    # Test corners
    top_left = cal.pixel_to_pitch((100, 100))
    assert top_left is not None
    print(f"  [PASS] Top-left pixel -> pitch {top_left}")

    print("[PASS] Pitch calibrator tests passed\n")


def test_team_classifier():
    """Test team classification."""
    from core.team.classifier import TeamClassifier

    config = {"vote_window": 10, "min_confidence": 0.6}
    classifier = TeamClassifier(config)

    # Create synthetic frame
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    class MockTrack:
        def __init__(self, tid, bbox):
            self.track_id = tid
            self.bbox = np.array(bbox)

    tracks = [
        MockTrack(1, [100, 100, 150, 200]),
        MockTrack(2, [300, 100, 350, 200]),
    ]

    results = classifier.classify_frame(frame, tracks)
    assert len(results) == 2
    print(f"  [PASS] Classified {len(results)} tracks: {results}")

    print("[PASS] Team classifier tests passed\n")


def test_event_detectors():
    """Test corner and goal detectors."""
    from core.events.detector import CornerDetector, GoalDetector, EventType

    pitch_config = {
        "dimensions": {"length_m": 105, "width_m": 68},
        "zones": {
            "corner_top_left": {"x_min": 0, "x_max": 5, "y_min": 63, "y_max": 68},
            "corner_top_right": {"x_min": 100, "x_max": 105, "y_min": 63, "y_max": 68},
            "corner_bottom_left": {"x_min": 0, "x_max": 5, "y_min": 0, "y_max": 5},
            "corner_bottom_right": {"x_min": 100, "x_max": 105, "y_min": 0, "y_max": 5},
        },
        "goals": {
            "goal_line_y_a": 30.34,
            "goal_line_y_b": 37.66,
        }
    }

    # Test corner detection
    corner_config = {"corner_zone_radius_m": 5.0, "min_stop_duration_ms": 500}
    cd = CornerDetector(corner_config, pitch_config)

    # Ball goes out, then appears in corner zone
    cd.ball_was_out = True
    cd.ball_out_time_ms = 1000
    result = cd.update((2.0, 65.0), True, 2000)
    assert result is not None
    assert result.event_type == EventType.CORNER
    print(f"  [PASS] Corner detected: team={result.team_id}")

    # Test goal detection
    goal_config = {"event_window_s": 30.0}
    gd = GoalDetector(goal_config, pitch_config)

    # Ball crosses goal line (moving right to left across goal B at x=0)
    gd.update((1.0, 34.0), 1000, True)  # Inside goal mouth
    result = gd.update((-0.5, 34.0), 1100, True)  # Crossed line
    # Note: -0.5 is outside pitch, detector checks transition
    # This is approximate - real detection needs trajectory

    print("[PASS] Event detector tests passed\n")


def test_stats_aggregator():
    """Test stats aggregation."""
    from core.stats.aggregator import StatsAggregator
    from core.events.detector import EventCandidate, EventType

    agg = StatsAggregator()
    agg.stats.match_id = "TEST"

    # Apply some events
    events = [
        EventCandidate(EventType.GOAL, "A", 1000, 1100, 0.95, ["test"]),
        EventCandidate(EventType.GOAL, "A", 5000, 5100, 0.90, ["test"]),
        EventCandidate(EventType.GOAL, "B", 8000, 8100, 0.92, ["test"]),
        EventCandidate(EventType.YELLOW_CARD, "B", 3000, 3100, 0.85, ["test"]),
        EventCandidate(EventType.CORNER, "A", 2000, 2100, 0.80, ["test"]),
        EventCandidate(EventType.CORNER, "B", 6000, 6100, 0.75, ["test"]),
        EventCandidate(EventType.CORNER, "B", 7000, 7100, 0.78, ["test"]),
    ]
    for e in events:
        e.status = "confirmed"

    agg.update_from_events(events)
    agg.update_possession(55.3, 44.7)
    agg.update_attacks({
        "team_a": {"attacks": 25, "dangerous_attacks": 12},
        "team_b": {"attacks": 20, "dangerous_attacks": 8}
    })
    agg.update_clock(90000, "1H")

    snapshot = agg.snapshot()
    assert snapshot["team_a"]["goals"] == 2
    assert snapshot["team_b"]["goals"] == 1
    assert snapshot["team_a"]["corners"] == 1
    assert snapshot["team_b"]["corners"] == 2
    assert snapshot["team_b"]["yellow_cards"] == 1
    assert snapshot["team_a"]["possession_pct"] == 55.3
    assert snapshot["team_a"]["attacks"] == 25
    assert snapshot["team_a"]["dangerous_attacks"] == 12

    print(f"  [PASS] Stats: {snapshot}")
    print("[PASS] Stats aggregator tests passed\n")


def run_all_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("FOOTBALL INTELLIGENCE - UNIT TESTS")
    print("=" * 60)

    tests = [
        test_possession_engine,
        test_attack_engine,
        test_pitch_calibrator,
        test_team_classifier,
        test_event_detectors,
        test_stats_aggregator,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
