# 足球比赛智能统计系统 — Software Design Document (SDD)

> **目标**：从单场足球视频实时/离线统计两队进球、红黄牌、角球、进攻、危险进攻与控球率
> **版本**：v1.0 | **日期**：2026-08-30
> **推荐基线**：固定高位全场摄像机 + AI 视觉 + 规则状态机 + 事件人工复核

---

## 0. 执行摘要

本系统的核心不是"识别 22 名球员"，而是构建一个连续的 **Match State（比赛状态）**：每帧识别球员、裁判和足球，区分两队，将图像坐标映射为球场坐标；再通过状态机计算控球、进攻、危险进攻，并通过事件模型、空间规则、OCR/比赛流程与人工确认共同确定进球、角球和红黄牌。

**最快落地策略**：固定高位、完整覆盖球场；不追求 22 名球员全场身份一致，不做球衣号码识别；MVP 只要求"属于哪支队伍 + 在哪里 + 谁更可能控制球"。

| 统计项 | MVP 实现 | 自动化难度 | 建议 |
|--------|----------|-----------|------|
| 控球率 | 球-最近球员/队伍 + 时间状态机 | 中 | 全自动 |
| 进攻 | 控球链 + 进入进攻半场 + 前向推进 | 中 | 全自动，可配置口径 |
| 危险进攻 | 进攻链 + 危险区域/射门/传中条件 | 中 | 全自动，可配置口径 |
| 角球 | 角旗区重新开球 + 事件识别 | 中 | 自动 + 低置信度复核 |
| 进球 | 球门线事件 + 中圈开球恢复 + OCR/事件模型 | 中高 | 自动 + 复核 |
| 黄牌/红牌 | 事件模型/裁判动作/人工按钮 | 高（全景单机位） | MVP 建议人工确认 |

---

## 1. 项目目标与范围

### 1.1 业务目标

- 输入一场足球比赛视频（RTSP、文件或直播流），输出 Team A / Team B 的实时与最终统计。
- 支持固定高位全场摄像头作为首选场景；兼容广播转播视频作为第二场景。
- 统计结果可被前端实时展示，并可回看每个统计项对应的事件时间点和证据片段。
- 所有"业务定义型指标"（进攻、危险进攻）必须配置化，而不是写死在模型里。

### 1.2 MVP 范围

| In Scope | Out of Scope（首版不做） |
|----------|------------------------|
| 球员/裁判/足球检测；两队区分；球场坐标；控球率；进攻；危险进攻；角球；进球；红黄牌候选；统计 API；审核台 | 球员姓名识别；全场稳定个人 ReID；球衣号码；个人传球成功率；xG；越位自动判罚；犯规责任人；VAR 级别判定 |

### 1.3 摄像机前提

- 固定高位，尽量完整覆盖 105m × 68m 球场；不要使用持续转动/变焦的 PTZ 作为唯一机位。
- 推荐 4K/25fps 或更高；球检测使用高分辨率 ROI/切片，球员检测可降采样。
- 两个球门、四个角旗区域尽量可见；相机标定后保持机位固定。

---

## 2. 系统总体架构

```
RTSP / MP4 / HLS
      |
      v
+-------------------+
| Ingest / FFmpeg   |
+-------------------+
      |
      +----------------------------+
      |                            |
      v                            v
Player/Referee Detector      Ball Detector (high-res/tiled)
      |                            |
      v                            v
Multi-Object Tracker          Ball Track / Kalman filter
      |                            |
      +-------------+--------------+
                    v
            Team Classification
                    |
                    v
          Pitch Calibration/Homography
                    |
                    v
               Match State
          +---------+---------+
          |         |         |
      Possession  Attack  Dangerous Attack
          |         |         |
          +----+----+---------+
               |
               v
        Event Fusion Engine
  Goal / Corner / Card / Restart
               |
               v
      Stats Aggregator + Review
               |
        FastAPI / WebSocket
               |
        Dashboard / Database
```

### 2.1 技术栈

| 层 | 推荐 | 原因 |
|----|------|------|
| 视频 | FFmpeg / GStreamer | 成熟、支持 RTSP/HLS/MP4、硬解码 |
| 视觉推理 | Python + PyTorch | 生态完整，足球研究代码易接入 |
| 检测 | Ultralytics YOLO 当前系列（MVP） | 训练/部署最快 |
| 跟踪 | OC-SORT 或 BoT-SORT | 体育快速运动 |
| 图像几何 | OpenCV | Homography、Perspective Transform、ROI |
| 事件识别 | SoccerNet Action Spotting 基线 | 已有足球事件标注 |
| OCR | PaddleOCR | 比分牌、比赛时间、广播流辅助确认 |
| API | FastAPI | Python 栈一致、WebSocket 简单 |
| 缓存 | Redis（生产可选） | 实时状态、Pub/Sub |
| 数据库 | PostgreSQL | 比赛、事件、统计、审核记录 |
| MVP 前端 | Streamlit | 最快交付审核与调参页面 |
| 正式前端 | React/Vue | 产品化 |

### 2.2 关键架构决策

1. **球员检测与足球检测分开**：球远小于球员，若共享同一低分辨率输入，球召回率会成为系统瓶颈。
2. **不要求 22 个球员 track_id 全场永久稳定**：本需求是团队统计，短期轨迹 + team_id 足够。
3. **规则优先于端到端黑盒**：控球、进攻、危险进攻必须可解释、可调参。
4. **稀有事件采用多证据融合**：进球/角球/红黄牌不依赖单一模型。
5. **保存事件前后短视频证据**，所有统计都可追溯。

---

## 3. 数据与下载

没有一个公开数据集会完全匹配"你的固定机位 + 你的球衣颜色 + 你的镜头高度"。推荐"公开数据预训练/验证 + 自有机位小规模标注微调"的组合。

### 3.1 SoccerNet Action Spotting

用途：训练/验证 Goal、Corner、Yellow card、Red card 等事件的时间定位。500 场广播足球视频（720p/224p）。

```python
from SoccerNet.Downloader import SoccerNetDownloader
dl = SoccerNetDownloader(LocalDirectory="./data/SoccerNet")
dl.downloadGames(files=["Labels-v2.json"], split=["train", "valid", "test"])
```

### 3.2 SoccerNet Game State Reconstruction

用途：球员/门将/裁判检测、team/role 属性、球场坐标等。57 train + 59 validation + 50 test 的 30 秒 1080p 主镜头片段。

```python
dl = SoccerNetDownloader(LocalDirectory="./data/SoccerNetGS")
dl.downloadDataTask(task="gamestate-2024", split=["train", "valid", "test", "challenge"])
```

### 3.3 SoccerNet Tracking

用途：足球运动场景多目标跟踪基线；标注格式兼容 MOT Challenge。

```python
dl = SoccerNetDownloader(LocalDirectory="./data/SoccerNetTracking")
dl.downloadDataTask(task="tracking", split=["train", "test", "challenge"])
```

### 3.4 SportsMOT（补充）

用途：补充足球运动员的高速运动、遮挡、多目标跟踪场景。官方仓库：[MCG-NJU/SportsMOT](https://github.com/MCG-NJU/SportsMOT)。注意 CC BY-NC 4.0 许可。

### 3.5 自有机位数据（生产准确率最关键）

标注类别建议：

| ID | 类别 |
|----|------|
| 0 | player |
| 1 | goalkeeper |
| 2 | referee |
| 3 | ball |

可选属性（不要做成检测类别）：`team_id: A / B / referee / unknown`

---

## 4. 视觉模型设计

### 4.1 球员/裁判检测

- 输入：从 4K 原始帧生成 1280–1920 宽的检测帧
- 类别：player / goalkeeper / referee
- 推理频率：10–15 FPS 通常足够；中间帧由 tracker 补齐

### 4.2 足球检测（独立模型）

足球是最关键且最困难的目标：

- 保留高分辨率；按球场 ROI 切成 2×2 或 3×2 tiles 推理
- 利用上一帧 ball track 生成局部搜索窗口
- 短时漏检使用 Kalman/运动模型预测
- 对 ball confidence 设置双阈值

### 4.3 Multi-Object Tracking

固定相机首选 OC-SORT/ByteTrack；广播流使用带相机运动补偿的 BoT-SORT。

### 4.4 球队分类

MVP 使用 jersey ROI 的颜色 + track 多帧投票：

```python
for track in active_player_tracks:
    jersey_roi = crop_upper_torso(track.bbox)
    feature = LAB_histogram(jersey_roi)
    team_prob = team_classifier(feature)
    track.team_id = temporal_majority_vote(team_prob, window=30)
```

---

## 5. 球场标定与坐标系统

### 5.1 固定机位：一次性 Homography

```python
H, mask = cv2.findHomography(image_points, pitch_points, method=cv2.RANSAC)
pitch_xy = cv2.perspectiveTransform(
    np.array([[[pixel_x, pixel_y]]], dtype=np.float32), H
)[0, 0]
```

### 5.2 球场规范坐标

- x ∈ [0, 105] meter, y ∈ [0, 68] meter
- Half 1: Team A attacks +x, Team B attacks -x
- Half 2: swap attacking direction

### 5.3 区域定义

| 区域 | 默认定义 | 用途 |
|------|---------|------|
| Own Half / Attacking Half | 以 x=52.5m 为中线 | 进攻判定 |
| Final Third | 距对方球门线 35m 内 | 进攻强度 |
| Danger Zone | 对方最后 30m + 中路高权重区域 | 危险进攻 |
| Penalty Area | 标准禁区范围 | 高危事件 |
| Corner Zone | 四个角旗附近半径/矩形区 | 角球 |
| Goal Mouth | 两门柱之间的门线区 | 进球候选 |

---

## 6. Match State 数据结构

```
MatchState
  match_id
  timestamp_ms
  period                 # 1H / 2H / ET
  attacking_direction
  ball:
      pixel_xy / pitch_xy / confidence / track_state
  players[]:
      track_id / team_id / role / pitch_xy / confidence
  possession:
      team_id / player_track_id / confidence / state  # CONTROLLED / CONTESTED / LOOSE
  attack:
      team_id / attack_id / phase                      # BUILD_UP / ATTACK / DANGEROUS
  pending_events[]
```

---

## 7. 业务指标实现

### 7.1 控球率

核心思想：不是逐帧最近距离直接切换，而是"候选控球者 + 稳定时间 + 状态机"。

```
possession_A = controlled_ms_A / (controlled_ms_A + controlled_ms_B)
possession_B = 1 - possession_A
```

- 控球切换要求候选队伍稳定约 0.4–1.0 秒
- LOOSE_BALL / UNKNOWN 默认不计入分母

### 7.2 进攻次数

"独立控球链"计数，避免每次过半场都 +1。

**ATTACK starts when:**
1. team establishes CONTROLLED possession
2. ball/player progression reaches attacking half
3. forward_progress >= min_progress OR ball enters final-third corridor

**ATTACK ends when:** opponent establishes possession / ball out / goal / timeout

### 7.3 危险进攻

同一次 ATTACK 中第一次满足危险条件时 +1：

- A) ball enters penalty area
- B) ball enters configurable danger zone
- C) detected shot / cross / key-ball event
- D) controlled dribble enters central channel near box

### 7.4 角球

在"角球被开出"时计数：ball reappears in corner zone + attacking team approaches ball.

### 7.5 进球

多证据融合：ball trajectory crosses goal line + play stops + restart is center kick-off + OCR/Event model.

### 7.6 黄牌/红牌

单高位全景机位最不适合完全自动化的指标。MVP 默认人工确认。

---

## 8. Event Fusion Engine

```
Fusion rules:
1. merge same-type candidates in temporal window
2. compute weighted confidence
3. if confidence >= AUTO_CONFIRM -> confirmed
4. if REVIEW_LOW <= confidence < AUTO_CONFIRM -> pending_review
5. else discard
6. manual decision is immutable audit evidence
```

| 事件 | 自动确认阈值 | 复核策略 |
|------|------------|---------|
| Goal | 高 | 宁可 pending，不可重复计数 |
| Corner | 中高 | 空间规则强时自动 |
| Yellow/Red | 高 | 固定全景默认进入审核 |
| Attack/Dangerous | 规则型 | 无需人工逐条确认 |

---

## 9. API 设计

| Method | Path | 说明 |
|--------|------|------|
| POST | `/matches` | 创建比赛 |
| POST | `/matches/{id}/start` | 开始处理 |
| GET | `/matches/{id}/stats` | 当前统计 |
| GET | `/matches/{id}/events` | 事件列表 |
| GET | `/matches/{id}/state` | 当前 Match State |
| WS | `/matches/{id}/live` | 实时推送状态/统计 |
| POST | `/events/{id}/confirm` | 确认候选事件 |
| POST | `/events/{id}/reject` | 驳回候选事件 |
| POST | `/matches/{id}/manual-event` | 人工补记红黄牌/进球等 |

### Stats JSON 示例

```json
{
  "match_id": "M20260830_001",
  "clock_ms": 3872000,
  "team_a": {
    "goals": 1, "yellow_cards": 2, "red_cards": 0,
    "corners": 4, "attacks": 31, "dangerous_attacks": 14,
    "possession_pct": 54.7
  },
  "team_b": {
    "goals": 0, "yellow_cards": 1, "red_cards": 0,
    "corners": 3, "attacks": 27, "dangerous_attacks": 11,
    "possession_pct": 45.3
  }
}
```

---

## 10. 项目目录

```
soccer_net/
├── apps/
│   └── api/                  # FastAPI REST + WebSocket
├── core/
│   ├── detection/            # player_detector.py / ball_detector.py
│   ├── tracking/             # Multi-object tracker
│   ├── team/                 # Team classifier (jersey color)
│   ├── pitch/                # Pitch calibration / homography
│   ├── possession/           # Possession state machine
│   ├── attack/               # Attack / dangerous attack rules
│   ├── events/               # Event detection + fusion
│   └── stats/                # Stats aggregator
├── configs/
│   ├── camera.yaml
│   ├── detection.yaml
│   ├── pitch.yaml
│   └── rules.yaml
├── models/                   # YOLO weights
├── data/                     # Test videos, calibration
├── tests/
├── docs/
│   └── SDD_v1.0.md           # This document
├── scripts/
├── main.py                   # CLI entry point
└── requirements.txt
```

---

## 11. 配置文件

```yaml
# configs/rules.yaml
possession:
  max_player_ball_distance_m: 2.3
  switch_hold_ms: 600
  unknown_timeout_ms: 1200

attack:
  require_attacking_half: true
  min_forward_progress_m: 8.0
  end_on_possession_loss: true

dangerous_attack:
  one_per_attack: true
  final_distance_to_goal_m: 30.0
  penalty_area_is_dangerous: true

event_review:
  goal_auto_confirm: 0.90
  corner_auto_confirm: 0.85
  card_auto_confirm: 0.95
```

---

## 12. 评估与验收标准

| 能力 | MVP 验收口径 |
|------|-------------|
| Player detection | 可见场上球员/裁判总体召回 ≥ 95% |
| Ball tracking | 有效比赛时间内 ball localization coverage ≥ 90% |
| Team assignment | 稳定轨迹 team_id 准确率 ≥ 98% |
| Possession | team possession accuracy ≥ 90%；全场控球率误差 ≤ 5 个百分点 |
| Attack | 与业务标注事件 F1 ≥ 0.85 |
| Dangerous attack | 与业务标注事件 F1 ≥ 0.80 |
| Corner | 事件 Precision/Recall ≥ 0.90 |
| Goal | 最终计分 100% 可通过自动 + 审核闭环保证 |
| Cards | 固定全景最终计数通过人工审核保证 |

---

## 13. 性能与部署

| 环境 | 建议 |
|------|------|
| 开发/单路 MVP | NVIDIA RTX 4070+ GPU, 32GB RAM, NVMe SSD |
| 边缘部署 | 具备 NVDEC 的 NVIDIA GPU |
| CPU-only | 不建议实时 4K；可用于离线验证 |

---

## 14. 关键风险与规避

| 风险 | 影响 | 规避 |
|------|------|------|
| 球太小/漏检 | 所有业务指标被放大影响 | 高分辨率球模型、切片、局部搜索、自有数据 |
| 球员遮挡导致 ID Switch | 控球瞬时抖动 | 团队级统计，不依赖永久 player ID；状态机抗抖 |
| 球衣颜色相似 | 队伍误分 | 赛前人工颜色种子 + 多帧投票 + 可选分类器 |
| 广播回放 | 重复 Goal/Card | shot boundary / scoreboard clock / replay detection / event 去重 |
| 红黄牌远距离不可见 | 漏牌 | 人工审核或第二机位 |
| 危险进攻口径争议 | 客户数据对不上 | 规则 profile 配置化 + 对齐标注样本 |

---

## 15. 最快实施顺序

| Sprint | 交付物 | 完成标准 |
|--------|--------|---------|
| Sprint 1：视觉底座 | 视频解码、球员/球检测、跟踪、Team A/B、标定 | 画面上能稳定显示 team_id、track_id、球场坐标 |
| Sprint 2：连续统计 | 控球状态机、进攻、危险进攻、WebSocket/Stats API | 完整 90 分钟视频可输出三项统计并回放状态 |
| Sprint 3：事件 | 角球、进球、Pending Event、审核 UI | 稀有事件可自动候选、人工确认、最终统计正确 |
| Sprint 4：增强 | 广播 OCR、Action Spotting、TensorRT、监控 | 提高自动率与性能，不改变核心业务架构 |

---

## 16. 参考资料

- [SoccerNet Action Spotting](https://www.soccer-net.org/tasks/action-spotting)
- [SoccerNet sn-spotting](https://github.com/SoccerNet/sn-spotting)
- [SoccerNet Game State Reconstruction](https://www.soccer-net.org/tasks/game-state-reconstruction)
- [SoccerNet sn-gamestate](https://github.com/SoccerNet/sn-gamestate)
- [SoccerNet sn-tracking](https://github.com/SoccerNet/sn-tracking)
- [SportsMOT](https://github.com/MCG-NJU/SportsMOT)
- [Ultralytics Track](https://docs.ultralytics.com/modes/track)
- [OpenCV Homography](https://docs.opencv.org/4.x/d7/dff/tutorial_feature_homography.html)
- [Ultralytics License](https://www.ultralytics.com/license)

---

## 17. 结论

推荐把本项目建设成"足球比赛状态引擎"，而不是一个单模型分类器。底层使用检测/跟踪/球场几何得到稳定的 player-team-ball state；上层使用可配置状态机统计控球、进攻、危险进攻；稀有事件使用多证据融合并保留人工审核。这个架构能最快完成业务需求，同时允许后续增加射门、传球、热力图、跑动距离、阵型等高级能力。
