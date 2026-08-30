# Football Intelligence System

**Real-time football match statistics from video analysis.**

From a single fixed panoramic camera feed, this system automatically tracks 22 players and the ball, classifies teams, and produces live match statistics:

| Statistic | Method |
|-----------|--------|
| **Possession %** | Ball-nearest-player state machine with temporal hold |
| **Attacks** | Possession chain + attacking half + forward progress |
| **Dangerous Attacks** | Attack + danger zone / penalty area rules |
| **Corners** | Ball-out + corner-zone spatial detection |
| **Goals** | Ball trajectory crossing goal line (multi-evidence) |
| **Yellow/Red Cards** | Manual entry (MVP) + optional action spotting |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run demo (no video needed)
python main.py demo

# Process a video
python main.py process data/sample.mp4 -o output.mp4

# Start API server
python main.py serve --port 8000
```

## Architecture

```
Video Input → Detection (YOLOv8) → Tracking (ByteTrack)
    → Team Classification (LAB histogram + voting)
    → Pitch Calibration (homography)
    → Match State Engine
        ├── Possession (state machine)
        ├── Attack / Dangerous Attack (rules)
        └── Events (corner / goal / card fusion)
    → Stats Aggregator → API + WebSocket → Dashboard
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | /matches | Create match |
| POST | /matches/{id}/start | Start processing |
| GET | /matches/{id}/stats | Current statistics |
| GET | /matches/{id}/events | All events |
| GET | /matches/{id}/state | Match state |
| WS | /matches/{id}/live | Real-time updates |
| POST | /matches/{id}/manual-event | Add manual event |

## Configuration

All business rules are configurable in `configs/rules.yaml`:

```yaml
possession:
  max_player_ball_distance_m: 2.3
  switch_hold_ms: 600

attack:
  min_forward_progress_m: 8.0
  require_attacking_half: true

dangerous_attack:
  penalty_area_is_dangerous: true
```

## Project Structure

```
football-intelligence/
├── core/               # Core algorithms
│   ├── detection/      # Player & ball detection (YOLOv8)
│   ├── tracking/       # Multi-object tracking
│   ├── team/           # Team classification
│   ├── pitch/          # Pitch calibration & coordinates
│   ├── possession/     # Possession state machine
│   ├── attack/         # Attack & dangerous attack rules
│   ├── events/         # Event detection & fusion
│   └── stats/          # Statistics aggregation
├── apps/
│   ├── api/            # FastAPI REST + WebSocket server
│   └── review-ui/      # Streamlit review dashboard (TODO)
├── configs/            # YAML configuration files
├── data/               # Test videos & calibration data
├── tests/              # Unit & integration tests
├── main.py             # CLI entry point
└── requirements.txt
```

## Based on SDD v1.0

See `足球比赛智能统计系统_SDD_v1.0.docx` for the full software design document.
