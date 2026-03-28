# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Computer vision system that extracts analytics from padel game recordings: player positions/velocities, ball tracking, 2D court projections, heatmaps, and stroke classification. Assumes a fixed camera setup. Requires GPU with at least 8GB VRAM for default batch sizes.

## Setup & Running

```bash
conda create -n padel_analytics python=3.12 pip
conda activate padel_analytics
pip install -r requirements.txt
# Also install PyTorch separately: https://pytorch.org/get-started/locally/
```

Model weights must be downloaded from Google Drive and paths configured in `config.py`.

### Run inference pipeline
```bash
python main.py
```

### Run Streamlit dashboard
```bash
streamlit run app.py
```

## Architecture

### Pipeline Flow

1. **Court keypoints selection** — On first run, a UI pops up to select 12 court keypoints (k1-k12) used for homography. These can be cached to JSON via `config.py` (`FIXED_COURT_KEYPOINTS_LOAD_PATH`/`SAVE_PATH`).
2. **Tracking** — `TrackingRunner` orchestrates multiple `Tracker` subclasses sequentially over video frames. Each tracker can load/save predictions as JSON to `./cache/`.
3. **Homography & Projection** — `ProjectedCourt` computes a homography matrix from detected court keypoints to project player/ball positions onto a 2D court plane.
4. **Data Collection** — `DataAnalytics` collects per-frame player positions (in meters) and computes velocity/acceleration/distance features across multiple frame intervals.

### Key Abstractions

- **`Tracker` (ABC)** in `trackers/tracker.py` — Base class for all trackers. Subclasses implement `predict_sample` (batch-based) or `predict_frames` (generator-based). Uses `NoPredictSample`/`NoPredictFrames` exceptions to dispatch between modes.
- **`Object` (ABC)** in `trackers/tracker.py` — Base class for tracked entities (Player, Ball, Keypoints). Must implement `from_json`, `serialize`, and `draw`.
- **`TrackingRunner`** in `trackers/runner.py` — Runs all trackers, writes annotated output video, and collects data. Skips trackers that already have cached predictions loaded.
- **`ProjectedCourt`** in `analytics/projected_court.py` — Handles homography computation and 2D projection of players/ball onto a minimap overlay.

### Tracker Implementations

| Tracker | Model | Module |
|---------|-------|--------|
| Players (bounding box) | YOLOv8 | `trackers/players_tracker/` |
| Player pose (13 keypoints) | YOLO keypoints | `trackers/players_keypoints_tracker/` |
| Ball position | TrackNet + InpaintNet | `trackers/ball_tracker/` |
| Court keypoints | YOLO | `trackers/keypoints_tracker/` |

### Configuration

All model paths, batch sizes, load/save cache paths, and video I/O paths are in `config.py`. Tracker results are cached as JSON in `./cache/` — set `load_path` to skip re-inference.

### Court Keypoint Layout

The court uses 12 keypoints (k1-k12) in a specific layout used consistently throughout `main.py`, `projected_court.py`, and keypoints tracker:
```
k11--------------------k12
|                       |
k8-----------k9--------k10
|            |          |
k6----------------------k7
|            |          |
k3-----------k4---------k5
|                       |
k1----------------------k2
```

Court dimensions (in meters) are defined in `constants/court_dimensions.py`.
