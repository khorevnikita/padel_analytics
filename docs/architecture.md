# Architecture

## Overview

Padel Analytics is a computer vision system that extracts analytics from padel game recordings. It processes video through a multi-stage pipeline: detection, tracking, geometric projection, and data collection. The system assumes a fixed camera setup and requires a GPU with at least 8 GB VRAM for default batch sizes.

## High-Level Pipeline

```
Video Input
    |
    v
Court Keypoints Selection (manual UI or cached JSON)
    |
    v
TrackingRunner orchestrates 4 trackers sequentially:
    1. PlayerTracker        (YOLOv8 + ByteTrack)
    2. PlayerKeypointsTracker (YOLO-Pose, 13 keypoints per player)
    3. BallTracker          (TrackNet + InpaintNet)
    4. KeypointsTracker     (YOLO, 12 court keypoints)
    |
    v
Homography & 2D Projection (ProjectedCourt)
    |
    v
Data Collection (DataAnalytics) --> CSV export
    |
    v
Annotated Video Output + Streamlit Dashboard
```

## Directory Structure

```
padel_analytics/
+-- main.py                     # CLI entry point
+-- app.py                      # Streamlit dashboard
+-- config.py                   # All paths, batch sizes, hyperparameters
+-- estimate_velocity.py        # Ball velocity estimation module
+-- trackers/
|   +-- tracker.py              # Tracker and Object ABCs, TrackingResults
|   +-- runner.py               # TrackingRunner pipeline orchestrator
|   +-- __init__.py             # Public API exports
|   +-- players_tracker/        # YOLOv8 person detection + ByteTrack
|   +-- players_keypoints_tracker/  # YOLO-Pose (13 body keypoints)
|   +-- ball_tracker/           # TrackNet + InpaintNet
|   +-- keypoints_tracker/      # YOLO court keypoint detection
+-- analytics/
|   +-- data_analytics.py       # DataAnalytics, DataPoint, PlayerPosition
|   +-- projected_court.py      # Homography, 2D projection, minimap overlay
+-- constants/
|   +-- court_dimensions.py     # Court measurements in meters
|   +-- player_heights.py       # Professional player heights (reference)
+-- utils/
|   +-- conversions.py          # Pixel <-> meters conversion
|   +-- converters.py           # Image format conversions (numpy/PIL/base64)
|   +-- video.py                # Video I/O helpers
+-- visualizations/
|   +-- padel_court.py          # Plotly 2D court figure generator
|   +-- player_centric_graphs.py
+-- cache/                      # Cached tracker predictions (JSON)
+-- weights/                    # Pre-trained model checkpoints
+-- examples/                   # Sample videos
```

## Core Abstractions

### Object (ABC) -- `trackers/tracker.py`

Base class for all tracked entities (players, ball, court keypoints). Every subclass must implement:

- `from_json(x)` -- deserialize from dict/list
- `serialize()` -- convert to JSON-compatible dict
- `draw(frame, **kwargs)` -- annotate a video frame with the prediction

### TrackingResults -- `trackers/tracker.py`

Dataclass that stores a list of `Object` predictions. Supports indexing, iteration, bulk load, incremental update, and restart. Acts as the state container for each tracker.

### Tracker (ABC) -- `trackers/tracker.py`

Base class for all trackers. Key design decisions:

- **Dual prediction modes**: subclasses implement either `predict_sample` (batch of frames) or `predict_frames` (full frame generator). The dispatch mechanism uses exceptions: `NoPredictSample` and `NoPredictFrames`. The `predict_and_update` method tries `predict_frames` first, catches `NoPredictFrames`, then falls back to batch sampling via an internal `sampler` generator.
- **Caching**: each tracker can load/save predictions as JSON. If `load_path` is set, predictions are loaded in `__init__` and inference is skipped entirely.
- **Device management**: `DEVICE` property auto-detects CUDA. The `to(device)` method moves models between GPU and CPU to free VRAM between tracker runs.

### TrackingRunner -- `trackers/runner.py`

Pipeline orchestrator. For each tracker:

1. Skip if predictions already loaded from cache
2. Move model to GPU
3. Run `predict_and_update` over the frame generator
4. Save predictions to cache
5. Move model back to CPU

After all trackers finish, `draw_and_collect_data` iterates all frames once more to:
- Draw annotations from each tracker
- Compute homography and project positions onto the 2D minimap
- Collect per-frame player positions into `DataAnalytics`
- Write the annotated output video

### ProjectedCourt -- `analytics/projected_court.py`

Handles the geometric transformation from camera perspective to a top-down 2D court view:

1. Computes a canvas overlay sized at 14% x 47% of the video frame
2. Places a scaled court diagram with correct proportions (using real court dimensions in meters)
3. Computes a homography matrix `H` from detected court keypoints to projected court keypoints using `cv2.findHomography`
4. Projects player feet and ball positions through `H` onto the 2D plane
5. Converts projected pixel coordinates to meters using the court width as reference

### DataAnalytics -- `analytics/data_analytics.py`

Per-frame data collector. For each frame, accumulates `PlayerPosition` objects (player ID + position in meters). On `step()`, finalizes the current `DataPoint` and starts a new one.

`into_dataframe(fps)` produces a rich pandas DataFrame with derived features:
- Displacements at 1, 2, 3, 4 frame intervals
- Velocities and accelerations (m/s, m/s^2)
- Euclidean distance per frame
- Speed and acceleration magnitudes

## Tracker Implementations

### PlayerTracker

- **Model**: YOLOv8 (generic person detection)
- **Tracking**: supervision's ByteTrack for multi-object tracking with consistent IDs
- **Filtering**: `sv.PolygonZone` built from court corner keypoints to exclude detections outside the court
- **Output**: `Players` (list of `Player` objects with bounding boxes, IDs, confidence)

### PlayerKeypointsTracker

- **Model**: YOLO-Pose (13 body keypoints: feet, knees, torso, shoulders, elbows, hands, neck, head)
- **Preprocessing**: resizes frames to training image size (640 or 1280), then scales keypoint coordinates back
- **Output**: `PlayersKeypoints` with skeleton connectivity for stick-figure visualization

### BallTracker

- **Models**: TrackNet (ball detection across 8-frame trajectory sequences) + optional InpaintNet (fills gaps in trajectory)
- **Preprocessing**: resize to 288x512, normalize, optional background subtraction via median frame
- **Batching**: uses PyTorch `DataLoader` with `BallTrajectoryIterable`
- **Output**: `Ball` objects with position, visibility flag, and optional trajectory drawing

### KeypointsTracker

- **Model**: YOLO (detects 12 court keypoints)
- **Fixed mode**: if `fixed_keypoints_detection` is provided, returns the same keypoints for every frame (for fixed cameras with no court movement)
- **Output**: `Keypoints` (list of 12 `Keypoint` objects with ID and pixel position)

## Court Keypoint Layout

The system uses 12 keypoints in a standardized layout:

```
k11--------------------k12      (far baseline)
|                       |
k8-----------k9--------k10     (far service line)
|            |          |
k6----------------------k7      (net)
|            |          |
k3-----------k4---------k5     (near service line)
|                       |
k1----------------------k2      (near baseline)
```

These keypoints are used for:
- Homography computation (camera -> top-down projection)
- Polygon zone filtering (court boundaries for player detection)
- 2D minimap rendering

## Ball Velocity Estimation -- `estimate_velocity.py`

Standalone module for computing ball velocity between two user-selected frames:

1. Identifies the closest player to the ball at both frames
2. Uses player feet position as the ground-level z-reference
3. Optionally estimates vertical velocity (Vz) using player height as scale reference
4. Applies perspective transformation (homography to a standard 20m x 10m court)
5. Computes velocity components (Vx, Vy, Vz) with unit conversion (m/s, km/h, ft/s, mph)

## Entry Points

### `main.py` -- CLI Pipeline

1. Opens a UI to select 12 court keypoints (or loads from cached JSON)
2. Instantiates 4 trackers with config from `config.py`
3. Runs `TrackingRunner.run()`
4. Exports collected data to CSV

### `app.py` -- Streamlit Dashboard

Interactive web UI with:
- Video upload and processing
- Ball velocity estimation tool (manual frame selection)
- Player statistics: velocity over time, position heatmaps, distance/speed summaries
- Plotly court visualizations with velocity-colored scatter plots

## Configuration -- `config.py`

All configuration is module-level constants:

| Category | Examples |
|----------|----------|
| Input/Output | `INPUT_VIDEO_PATH`, `OUTPUT_VIDEO_PATH`, `COLLECT_DATA_PATH` |
| Frame range | `MAX_FRAMES` (None = full video) |
| Court keypoints | `FIXED_COURT_KEYPOINTS_LOAD_PATH`, `SAVE_PATH` |
| Per-tracker | model path, batch size, cache load/save path |

## Dependencies

| Library | Purpose |
|---------|---------|
| `ultralytics` | YOLOv8 / YOLO-Pose inference |
| `supervision` | ByteTrack, PolygonZone, VideoInfo, frame generators |
| `torch` | Deep learning backend (GPU) |
| `opencv-python` | Image processing, homography, video I/O |
| `pandas` / `numpy` | Data analysis and numerical computation |
| `streamlit` | Interactive web dashboard |
| `plotly` | Court visualizations |
| `pims` | Frame-level video access (dashboard) |
| `ffmpeg-python` | Video format conversion |

PyTorch must be installed separately following https://pytorch.org/get-started/locally/.
