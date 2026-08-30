#!/usr/bin/env python3
"""
Football Intelligence System - CLI Entry Point

Usage:
    python main.py process <video_path> [--output <output_path>] [--max-frames N]
    python main.py serve [--host 0.0.0.0] [--port 8000]
    python main.py demo
"""
import argparse
import sys
import os
import json
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent))


def cmd_process(args):
    """Process a video file and output statistics."""
    from core.match_engine import MatchEngine

    engine = MatchEngine()
    engine.stats.stats.match_id = args.match_id or "CLI_MATCH"

    def progress(frame_num, total, result):
        if frame_num % 100 == 0:
            stats = result.get("stats", {})
            ta = stats.get("team_a", {})
            tb = stats.get("team_b", {})
            print(f"  Frame {frame_num}/{total} | "
                  f"Poss: A={ta.get('possession_pct',50)}% B={tb.get('possession_pct',50)}% | "
                  f"ATK: A={ta.get('attacks',0)} B={tb.get('attacks',0)} | "
                  f"Goals: A={ta.get('goals',0)} B={tb.get('goals',0)}")

    print(f"Processing: {args.video_path}")
    print(f"Output: {args.output or 'none'}")
    print("-" * 60)

    final = engine.process_video(
        args.video_path,
        output_path=args.output,
        progress_callback=progress,
        max_frames=args.max_frames
    )

    print("=" * 60)
    print("FINAL MATCH STATISTICS")
    print("=" * 60)
    print(json.dumps(final, indent=2, default=str))

    # Save to file
    output_json = args.output.replace(".mp4", "_stats.json") if args.output else "match_stats.json"
    with open(output_json, 'w') as f:
        json.dump(final, f, indent=2, default=str)
    print(f"\nStats saved to: {output_json}")


def cmd_serve(args):
    """Start the API server."""
    from apps.api.main import start_server
    print(f"Starting Football Intelligence API on {args.host}:{args.port}")
    start_server(host=args.host, port=args.port)


def cmd_demo(args):
    """Run a demo with synthetic data to verify the pipeline."""
    from core.match_engine import MatchEngine
    import numpy as np

    print("Running demo with synthetic data...")
    print("=" * 60)

    engine = MatchEngine()
    engine.stats.stats.match_id = "DEMO_MATCH"

    # Create synthetic frames (just colored rectangles for testing)
    h, w = 1080, 1920
    engine.setup_calibration(w, h)

    print(f"Processing {args.frames} synthetic frames...")

    for i in range(args.frames):
        # Generate a simple synthetic frame
        frame = np.random.randint(50, 200, (h, w, 3), dtype=np.uint8)
        # Draw green field in lower portion
        frame[h//3:, :, 1] = np.clip(120 + np.random.randint(0, 40, (h - h//3, w), dtype=np.uint8) - 20, 0, 255).astype(np.uint8)

        result = engine.process_frame(frame, timestamp_ms=i * 40)  # 25fps = 40ms/frame

        if i % 50 == 0:
            stats = result["stats"]
            ta = stats["team_a"]
            tb = stats["team_b"]
            print(f"  Frame {i}: Poss A={ta['possession_pct']}% B={tb['possession_pct']}% | "
                  f"ATK A={ta['attacks']} B={tb['attacks']} | "
                  f"DNG A={ta['dangerous_attacks']} B={tb['dangerous_attacks']}")

    final = engine.stats.snapshot()
    print("=" * 60)
    print("DEMO RESULTS:")
    print(json.dumps(final, indent=2))
    print("\nDemo complete. Pipeline verified!")


def main():
    parser = argparse.ArgumentParser(description="Football Intelligence System")
    subparsers = parser.add_subparsers(dest="command")

    # Process command
    proc = subparsers.add_parser("process", help="Process a video file")
    proc.add_argument("video_path", help="Path to video file")
    proc.add_argument("--output", "-o", help="Output annotated video path")
    proc.add_argument("--max-frames", "-n", type=int, help="Max frames to process")
    proc.add_argument("--match-id", default="CLI_MATCH", help="Match ID")

    # Serve command
    serve = subparsers.add_parser("serve", help="Start API server")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)

    # Demo command
    demo = subparsers.add_parser("demo", help="Run demo with synthetic data")
    demo.add_argument("--frames", type=int, default=250, help="Number of frames")

    args = parser.parse_args()

    if args.command == "process":
        cmd_process(args)
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "demo":
        cmd_demo(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
