#!/usr/bin/env python3
"""
Download test soccer videos for validation.
Uses publicly available soccer clips from various sources.
"""
import os
import sys
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "videos"), exist_ok=True)


def download_file(url, dest):
    """Download a file if it doesn't exist."""
    if os.path.exists(dest):
        print(f"  Already exists: {dest}")
        return True
    print(f"  Downloading: {url}")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  Saved to: {dest}")
        return True
    except Exception as e:
        print(f"  Failed: {e}")
        return False


def create_synthetic_test_video():
    """Create a synthetic test video using OpenCV."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("  OpenCV not available, skipping synthetic video")
        return None

    output_path = os.path.join(DATA_DIR, "videos", "synthetic_match.mp4")
    if os.path.exists(output_path):
        print(f"  Already exists: {output_path}")
        return output_path

    print("  Creating synthetic test video...")
    w, h, fps = 1920, 1080, 25
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    np.random.seed(42)
    ball_x, ball_y = w//2, h//2
    ball_vx, ball_vy = 4.0, 2.0

    # Player base positions
    players_a = [(200 + i*100, h//2 + np.random.randint(-200, 200)) for i in range(11)]
    players_b = [(w - 200 - i*100, h//2 + np.random.randint(-200, 200)) for i in range(11)]

    total_frames = 25 * 30  # 30 seconds

    for f in range(total_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Green field
        frame[:, :, 1] = 80
        frame[h//4:, :, 1] = 120

        # Field markings
        cx, cy = w//2, h//2
        cv2.circle(frame, (cx, cy), 100, (255, 255, 255), 2)
        cv2.line(frame, (cx, 0), (cx, h), (255, 255, 255), 2)
        cv2.rectangle(frame, (50, 50), (w-50, h-50), (255, 255, 255), 2)
        cv2.rectangle(frame, (50, h//3), (w//6, 2*h//3), (255, 255, 255), 2)
        cv2.rectangle(frame, (w-w//6, h//3), (w-50, 2*h//3), (255, 255, 255), 2)

        # Move ball
        ball_x += ball_vx + np.random.randn() * 1.5
        ball_y += ball_vy + np.random.randn() * 1.0
        if ball_x < 80 or ball_x > w-80:
            ball_vx = -ball_vx
        if ball_y < 80 or ball_y > h-80:
            ball_vy = -ball_vy
        ball_x = np.clip(ball_x, 50, w-50)
        ball_y = np.clip(ball_y, 50, h-50)

        # Draw ball
        cv2.circle(frame, (int(ball_x), int(ball_y)), 8, (255, 255, 255), -1)

        # Draw players
        for px, py in players_a:
            nx = px + int(np.random.randn() * 5)
            ny = py + int(np.random.randn() * 5)
            cv2.rectangle(frame, (nx-12, ny-25), (nx+12, ny+25), (0, 0, 200), -1)  # Red team

        for px, py in players_b:
            nx = px + int(np.random.randn() * 5)
            ny = py + int(np.random.randn() * 5)
            cv2.rectangle(frame, (nx-12, ny-25), (nx+12, ny+25), (200, 200, 0), -1)  # Yellow team

        # Scoreboard text
        cv2.putText(frame, "Team A 0 - 0 Team B", (w//2-150, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, f"{f//25 // 60:02d}:{f//25 % 60:02d}", (w//2-30, h-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        writer.write(frame)

    writer.release()
    print(f"  Created: {output_path} ({total_frames} frames, 30 seconds)")
    return output_path


def main():
    print("Setting up test data...")
    print(f"Data directory: {DATA_DIR}")

    # Create synthetic video
    video_path = create_synthetic_test_video()

    # Try to download a real soccer clip (optional)
    # Note: These are example URLs; replace with actual test clips
    print("\nTest data setup complete!")
    if video_path:
        print(f"Synthetic video: {video_path}")
    print("\nYou can also place your own soccer videos in:")
    print(f"  {os.path.join(DATA_DIR, 'videos')}")


if __name__ == "__main__":
    main()
