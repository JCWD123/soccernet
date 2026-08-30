#!/usr/bin/env python3
"""
Full Pipeline Integration Test - processes synthetic_match.mp4
Verifies the complete flow: video -> detection -> tracking -> stats
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.match_engine import MatchEngine


def test_video_pipeline():
    """Process the synthetic test video end-to-end."""
    video_path = os.path.join("data", "videos", "synthetic_match.mp4")

    if not os.path.exists(video_path):
        print(f"Test video not found: {video_path}")
        print("Run: python scripts/download_test_data.py")
        return False

    print("=" * 60)
    print("FULL VIDEO PIPELINE TEST")
    print("=" * 60)
    print(f"Input: {video_path}")

    engine = MatchEngine()
    engine.stats.stats.match_id = "VIDEO_PIPELINE_TEST"

    output_path = os.path.join("data", "videos", "output_annotated.mp4")

    frame_count = [0]
    def progress(frame_num, total, result):
        frame_count[0] = frame_num
        if frame_num % 50 == 0:
            stats = result.get("stats", {})
            ta = stats.get("team_a", {})
            tb = stats.get("team_b", {})
            print(f"  Frame {frame_num:4d}/{total}: "
                  f"Poss A={ta.get('possession_pct',50):5.1f}% B={tb.get('possession_pct',50):5.1f}% | "
                  f"ATK A={ta.get('attacks',0)} B={tb.get('attacks',0)} | "
                  f"Players={result.get('players_detected',0)} Ball={'Y' if result.get('ball_detected') else 'N'}")

    # Process with max 250 frames (10 seconds) for speed
    final = engine.process_video(
        video_path,
        output_path=output_path,
        progress_callback=progress,
        max_frames=250
    )

    print("\n" + "=" * 60)
    print("FINAL VIDEO PIPELINE RESULTS")
    print("=" * 60)
    print(json.dumps(final, indent=2, default=str))

    # Save stats
    stats_path = os.path.join("data", "videos", "pipeline_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(final, f, indent=2, default=str)

    # Validation
    ta = final.get("team_a", {})
    tb = final.get("team_b", {})
    checks = [
        ("Processed frames", frame_count[0] > 0),
        ("Stats generated", final.get("clock_ms", 0) > 0),
        ("Possession sums to ~100%",
         abs(ta.get("possession_pct", 0) + tb.get("possession_pct", 0) - 100) < 2),
        ("No negative stats",
         all(v >= 0 for v in [ta.get("goals", 0), tb.get("goals", 0),
                               ta.get("yellow_cards", 0), tb.get("yellow_cards", 0),
                               ta.get("attacks", 0), tb.get("attacks", 0)])),
        ("Stats saved", os.path.exists(stats_path)),
    ]

    print("\nValidation:")
    all_ok = True
    for name, ok in checks:
        s = "PASS" if ok else "FAIL"
        print(f"  [{s}] {name}")
        if not ok:
            all_ok = False

    if os.path.exists(output_path):
        print(f"\nAnnotated video: {output_path}")
    print(f"Stats JSON: {stats_path}")

    return all_ok


if __name__ == "__main__":
    ok = test_video_pipeline()
    print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    sys.exit(0 if ok else 1)
