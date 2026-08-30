#!/usr/bin/env python3
"""
Realistic Integration Test - simulates a soccer match scenario
with players positioned near the ball for meaningful stats.
"""
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.match_engine import MatchEngine
from core.possession.engine import PossessionEngine, PossessionState
from core.attack.engine import AttackEngine, AttackPhase
from core.events.detector import CornerDetector, GoalDetector, EventType
from core.events.fusion import EventFusionEngine
from core.stats.aggregator import StatsAggregator


def test_realistic_match():
    """Simulate a realistic match with proper player-ball relationships."""
    print("=" * 60)
    print("REALISTIC MATCH SIMULATION")
    print("=" * 60)

    # Setup engines with proper configs
    possession_cfg = {
        "max_player_ball_distance_m": 2.3,
        "switch_hold_ms": 600,
        "unknown_timeout_ms": 1200,
        "contested_distance_m": 4.0
    }
    attack_cfg = {
        "require_attacking_half": True,
        "min_forward_progress_m": 8.0,
        "end_on_possession_loss": True,
        "timeout_without_progress_ms": 15000,
        "attack_cooldown_ms": 3000
    }
    danger_cfg = {
        "one_per_attack": True,
        "final_distance_to_goal_m": 30.0,
        "penalty_area_is_dangerous": True,
        "central_channel_half_width_m": 15.0
    }
    pitch_cfg = {
        "dimensions": {"length_m": 105, "width_m": 68},
        "zones": {
            "penalty_area_a": {"x_min": 88.5, "x_max": 105, "y_min": 13.85, "y_max": 54.15},
            "penalty_area_b": {"x_min": 0, "x_max": 16.5, "y_min": 13.85, "y_max": 54.15},
            "corner_top_left": {"x_min": 0, "x_max": 5, "y_min": 63, "y_max": 68},
            "corner_top_right": {"x_min": 100, "x_max": 105, "y_min": 63, "y_max": 68},
            "corner_bottom_left": {"x_min": 0, "x_max": 5, "y_min": 0, "y_max": 5},
            "corner_bottom_right": {"x_min": 100, "x_max": 105, "y_min": 0, "y_max": 5},
        },
        "goals": {"goal_line_y_a": 30.34, "goal_line_y_b": 37.66}
    }

    possession_engine = PossessionEngine(possession_cfg)
    attack_engine = AttackEngine(attack_cfg, danger_cfg, pitch_cfg)
    corner_detector = CornerDetector({"corner_zone_radius_m": 5.0, "min_stop_duration_ms": 500}, pitch_cfg)
    goal_detector = GoalDetector({"event_window_s": 30.0}, pitch_cfg)
    fusion = EventFusionEngine({"goal_auto_confirm": 0.90, "corner_auto_confirm": 0.85, "card_auto_confirm": 0.95})
    stats = StatsAggregator()
    stats.stats.match_id = "REALISTIC_SIM"

    # Simulate match scenario
    class MockPlayer:
        def __init__(self, tid, team, px, py):
            self.track_id = tid
            self.team_id = team
            self.pitch_xy = (px, py)
            self.bbox = np.array([0, 0, 10, 10])

    np.random.seed(42)

    # Scenario: Team A attacks right, Team B attacks left
    # Ball starts at center, moves right (Team A attack)
    ball_x, ball_y = 52.5, 34.0
    ball_vx = 0.5  # Moving right (Team A attacking)
    ball_vy = 0.1

    # Team A players (attacking right)
    players_a = [
        MockPlayer(1, "A", 50.0, 34.0),   # Near ball
        MockPlayer(2, "A", 55.0, 30.0),
        MockPlayer(3, "A", 55.0, 38.0),
        MockPlayer(4, "A", 60.0, 34.0),
        MockPlayer(5, "A", 65.0, 25.0),
        MockPlayer(6, "A", 65.0, 43.0),
        MockPlayer(7, "A", 70.0, 34.0),
        MockPlayer(8, "A", 75.0, 30.0),
        MockPlayer(9, "A", 75.0, 38.0),
        MockPlayer(10, "A", 80.0, 34.0),   # Striker
        MockPlayer(11, "A", 45.0, 34.0),   # GK
    ]

    # Team B players (defending left)
    players_b = [
        MockPlayer(12, "B", 52.0, 34.0),   # Near ball (contesting)
        MockPlayer(13, "B", 48.0, 30.0),
        MockPlayer(14, "B", 48.0, 38.0),
        MockPlayer(15, "B", 45.0, 34.0),
        MockPlayer(16, "B", 40.0, 25.0),
        MockPlayer(17, "B", 40.0, 43.0),
        MockPlayer(18, "B", 35.0, 34.0),
        MockPlayer(19, "B", 30.0, 30.0),
        MockPlayer(20, "B", 30.0, 38.0),
        MockPlayer(21, "B", 25.0, 34.0),   # Striker
        MockPlayer(22, "B", 60.0, 34.0),   # GK
    ]

    print("\nSimulating 90 minutes (5400 frames @ 1fps)...")
    print("Scenario: Team A attacks right, alternates possession")

    for frame in range(5400):  # 90 minutes at 1 frame per second
        ts = frame * 1000  # ms

        # Move ball with some randomness
        ball_x += ball_vx + np.random.randn() * 0.3
        ball_y += ball_vy + np.random.randn() * 0.2

        # Bounce ball off boundaries
        if ball_x > 100 or ball_x < 5:
            ball_vx = -ball_vx
        if ball_y > 63 or ball_y < 5:
            ball_vy = -ball_vy
        ball_x = np.clip(ball_x, 0, 105)
        ball_y = np.clip(ball_y, 0, 68)

        # Every ~30 seconds, switch attack direction
        if frame % 30 == 0:
            ball_vx = -ball_vx

        # Move players to follow ball loosely
        all_players = []
        for p in players_a:
            new_x = p.pitch_xy[0] + np.random.randn() * 0.5 + (ball_x - p.pitch_xy[0]) * 0.02
            new_y = p.pitch_xy[1] + np.random.randn() * 0.3 + (ball_y - p.pitch_xy[1]) * 0.02
            new_x = np.clip(new_x, 0, 105)
            new_y = np.clip(new_y, 0, 68)
            p.pitch_xy = (new_x, new_y)
            all_players.append(p)

        for p in players_b:
            new_x = p.pitch_xy[0] + np.random.randn() * 0.5 + (ball_x - p.pitch_xy[0]) * 0.02
            new_y = p.pitch_xy[1] + np.random.randn() * 0.3 + (ball_y - p.pitch_xy[1]) * 0.02
            new_x = np.clip(new_x, 0, 105)
            new_y = np.clip(new_y, 0, 68)
            p.pitch_xy = (new_x, new_y)
            all_players.append(p)

        # Update engines
        possession = possession_engine.update(all_players, (ball_x, ball_y), ts)
        attack_engine.update(possession.team_id, (ball_x, ball_y), ts)

        # Check events
        ball_visible = True
        events = []
        corner = corner_detector.update((ball_x, ball_y), ball_visible, ts)
        if corner:
            events.append(corner)
        goal = goal_detector.update((ball_x, ball_y), ts, ball_visible)
        if goal:
            events.append(goal)

        confirmed = fusion.process(events)
        stats.update_from_events(confirmed)
        stats.update_possession(*possession_engine.get_possession_pct())
        stats.update_attacks(attack_engine.get_stats())
        stats.update_clock(ts, "1H" if ts < 2700000 else "2H")

        # Print progress
        if frame % 900 == 0:  # Every 15 minutes
            snap = stats.snapshot()
            ta = snap["team_a"]
            tb = snap["team_b"]
            print(f"\n  {snap['clock_display']} ({snap['period']}):")
            print(f"    Possession: A={ta['possession_pct']}% B={tb['possession_pct']}%")
            print(f"    Attacks:    A={ta['attacks']} B={tb['attacks']}")
            print(f"    Dangerous:  A={ta['dangerous_attacks']} B={tb['dangerous_attacks']}")
            print(f"    Goals:      A={ta['goals']} B={tb['goals']}")
            print(f"    Corners:    A={ta['corners']} B={tb['corners']}")
            print(f"    Cards(Y/R): A={ta['yellow_cards']}/{ta['red_cards']} B={tb['yellow_cards']}/{tb['red_cards']}")
            print(f"    Ball: ({ball_x:.1f}, {ball_y:.1f}) Poss={possession.team_id} State={possession.state.value}")

    # Add some manual events to test that path
    from core.events.detector import EventCandidate
    manual_goal_a = EventCandidate(EventType.GOAL, "A", 1800000, 1800100, 1.0, ["manual_entry"], "confirmed")
    manual_goal_b = EventCandidate(EventType.GOAL, "B", 3600000, 3600100, 1.0, ["manual_entry"], "confirmed")
    manual_yellow = EventCandidate(EventType.YELLOW_CARD, "B", 2000000, 2000100, 1.0, ["manual_entry"], "confirmed")
    manual_corner_a = EventCandidate(EventType.CORNER, "A", 1500000, 1500100, 1.0, ["manual_entry"], "confirmed")
    manual_corner_b = EventCandidate(EventType.CORNER, "B", 3000000, 3000100, 1.0, ["manual_entry"], "confirmed")

    stats.update_from_events([manual_goal_a, manual_goal_b, manual_yellow, manual_corner_a, manual_corner_b])

    # Final report
    final = stats.snapshot()
    print("\n" + "=" * 60)
    print("FINAL MATCH STATISTICS (90 minutes)")
    print("=" * 60)
    print(json.dumps(final, indent=2))

    # Validate
    ta = final["team_a"]
    tb = final["team_b"]
    checks = [
        ("Possession sums to ~100%", abs(ta["possession_pct"] + tb["possession_pct"] - 100) < 2),
        ("Both teams have possession", ta["possession_pct"] > 5 and tb["possession_pct"] > 5),
        ("Attacks counted", ta["attacks"] + tb["attacks"] > 0),
        ("Goals counted", ta["goals"] + tb["goals"] >= 2),  # Manual goals
        ("Corners detected", ta["corners"] + tb["corners"] >= 2),  # Manual + auto
        ("Cards counted", tb["yellow_cards"] >= 1),  # Manual card
        ("No negative stats", all(v >= 0 for v in [
            ta["goals"], tb["goals"], ta["yellow_cards"], tb["yellow_cards"],
            ta["red_cards"], tb["red_cards"], ta["corners"], tb["corners"],
            ta["attacks"], tb["attacks"], ta["dangerous_attacks"], tb["dangerous_attacks"]
        ])),
    ]

    print("\nValidation:")
    all_ok = True
    for name, ok in checks:
        s = "PASS" if ok else "FAIL"
        print(f"  [{s}] {name}")
        if not ok:
            all_ok = False

    return all_ok


if __name__ == "__main__":
    ok = test_realistic_match()
    print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    sys.exit(0 if ok else 1)
