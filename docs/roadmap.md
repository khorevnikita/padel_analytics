# Roadmap

This document catalogs known issues, technical debt, and growth areas for the project. Items are grouped by priority.

---

## P0 -- Critical Issues

### Frame synchronization bug in TrackingRunner

**Location**: `trackers/runner.py:157-167`

The `draw_and_collect_data` method calls `data_analytics.step(1)` after processing each frame, which appends a new frame index and finalizes the current data point. After the loop, it removes the last frame with `self.data_analytics.frames = self.data_analytics.frames[:-1]`. This is a workaround for an off-by-one error: `step()` always appends one frame ahead, so after the last iteration there is an extra empty frame.

The root cause is that `DataAnalytics.step()` simultaneously finalizes the current frame *and* prepares the next one. The commented-out assertion (`assert len(self.data_analytics) == self.total_frames`) confirms this was a known issue.

**Impact**: silent data misalignment between tracker predictions and collected analytics.

**Fix**: separate the "finalize current data point" and "prepare next frame" responsibilities. Finalize at the end of each iteration; only prepare the next frame if more frames remain.

### Command injection in Streamlit app

**Location**: `app.py:148`

```python
os.system(f"ffmpeg -y -i {upload_video_path} -vcodec libx264 tmp.mp4")
```

The `upload_video_path` comes from user text input and is interpolated directly into a shell command without sanitization.

**Fix**: replace with `subprocess.run(["ffmpeg", "-y", "-i", upload_video_path, "-vcodec", "libx264", "tmp.mp4"])` or use the `ffmpeg-python` library API.

### No tests

There are zero automated tests in the project. No test files, no test framework configuration, no CI pipeline.

**Impact**: any refactoring or feature addition risks introducing silent regressions.

**Fix**: add a `tests/` directory with pytest. Priority test targets:
- JSON serialization round-trips for all `Object` subclasses
- Coordinate conversion functions (`utils/conversions.py`)
- `DataAnalytics` accumulation and DataFrame export
- Homography projection correctness with known keypoint pairs
- `TrackingResults` state transitions (load, update, restart)

### No dependency version pinning

**Location**: `requirements.txt`

All dependencies are unpinned. `ultralytics`, `supervision`, and `torch` are fast-moving libraries where breaking changes between versions are common.

**Fix**: pin versions (e.g., `ultralytics==8.x.x`) or use a lock file (`pip freeze > requirements.lock`).

---

## P1 -- Code Quality

### Replace print statements with logging

The entire codebase uses `print()` for diagnostics. There is no way to control verbosity, filter by component, or redirect output.

**Fix**: use Python's `logging` module. Define a logger per module (`logger = logging.getLogger(__name__)`). Use `DEBUG` for frame-level output, `INFO` for pipeline milestones, `WARNING` for missing data.

### Move hardcoded constants to configuration

Scattered hardcoded values:
- Player height `1.8` in `estimate_velocity.py:229,246` -- should use `constants/player_heights.py` or a config parameter
- Player IDs `(1, 2, 3, 4)` assumed throughout `data_analytics.py`, `app.py`
- Polygon zone indices `[0], [1], [-1], [-2]` in `main.py:109-117` and `app.py:179-187`
- Velocity unit multiplier `3.6` repeated in `app.py` instead of using `VelocityConverter`

**Fix**: centralize all magic numbers in `config.py` or `constants/`.

### Structured configuration with validation

`config.py` uses module-level constants with `from config import *`. No type checking, no validation, no defaults, no environment variable support.

**Fix**: use a `dataclass` or `pydantic.BaseModel` for configuration. Validate paths exist, batch sizes are positive, etc. Support overrides via environment variables or a YAML file.

### Mutable global state

`SELECTED_KEYPOINTS` in `main.py` is a mutable global list modified by a mouse callback. `app.py` defines it at module scope without initialization, relying on it being populated from JSON.

**Fix**: encapsulate keypoint selection in a function that returns the result. Pass keypoints explicitly rather than mutating globals.

### Duplicated setup code between main.py and app.py

Tracker instantiation, keypoint loading, polygon zone creation, and runner setup are copy-pasted between `main.py` and `app.py` (~60 lines each).

**Fix**: extract a shared `create_pipeline(config) -> TrackingRunner` function.

### Object ABC methods are not enforced

`Object.from_json`, `serialize`, and `draw` are defined with `pass` bodies but are not decorated with `@abstractmethod`. A subclass that forgets to implement them will silently inherit no-op methods.

**Fix**: add `@abstractmethod` to all three methods.

---

## P2 -- Missing Features

### Missing value handling in DataAnalytics

When a player is not detected in a frame, `into_dict()` inserts `None`. Downstream DataFrame operations (velocity, acceleration) propagate `NaN` without interpolation.

**Fix**: add configurable imputation: linear interpolation for short gaps, forward-fill for single-frame drops. Flag long gaps (>N frames) as genuinely missing.

### Stroke classification

Mentioned in the project description but not implemented. Player keypoints (13 body points) are detected but not used for any classification task.

**Opportunity**: use the temporal sequence of `PlayerKeypoints` to classify stroke types (forehand, backhand, volley, smash, serve). This could be a simple rule-based system using arm angles and ball proximity, or a trained classifier.

### Automated court keypoint detection

The current workflow requires manual keypoint selection via mouse clicks on the first frame (or loading from a cached JSON). This makes the system unsuitable for batch processing or new videos without human intervention.

**Opportunity**: the `KeypointsTracker` already detects court keypoints per frame. Use the first-frame detection as the initial keypoints instead of requiring manual selection. Fall back to manual mode if confidence is low.

### Rally/point segmentation

The system processes the entire video as a single continuous stream. There is no detection of rally boundaries, points, or game state.

**Opportunity**: use ball trajectory analysis (e.g., ball leaving the court, long pauses) and player position patterns to segment the video into individual rallies.

### Team assignment

Players are tracked with arbitrary ByteTrack IDs. There is no mechanism to assign players to teams or maintain consistent identity across rallies.

**Opportunity**: use player position (which side of the net) and appearance features (jersey color clustering) to assign team labels.

---

## P3 -- Production Readiness

### CI/CD pipeline

No GitHub Actions or any CI configuration exists.

**Target**: add a workflow that runs linting (ruff/flake8), type checking (mypy), and tests (pytest) on every push.

### Docker support

The setup requires conda, manual PyTorch installation, and model weight downloads from Google Drive. This is fragile and hard to reproduce.

**Target**: provide a `Dockerfile` with CUDA support, bundled dependencies, and a documented model download step.

### Error handling in inference loops

The tracking pipeline has minimal error handling. A single corrupted frame or failed detection can crash the entire run.

**Target**: wrap per-frame processing in try/except blocks with graceful degradation (skip frame, log warning, continue).

### Performance profiling

No instrumentation beyond `timeit` around the full pipeline. No visibility into per-tracker GPU memory usage, batch throughput, or bottlenecks.

**Target**: add optional profiling output (per-tracker timing, peak VRAM, frames/second).

### Multi-camera and dynamic camera support

The system assumes a single fixed camera. Court keypoints are either manually selected once or detected per frame but always for the same court from the same angle.

**Target**: support camera motion by updating homography per frame (already partially supported via `is_fixed_keypoints=False` path). Support multiple camera angles with view stitching.

---

## Summary

| Priority | Category | Count |
|----------|----------|-------|
| P0 | Critical bugs and risks | 4 |
| P1 | Code quality and maintainability | 6 |
| P2 | Missing features | 5 |
| P3 | Production readiness | 5 |
