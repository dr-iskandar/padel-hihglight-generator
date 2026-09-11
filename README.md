# Padel Auto-Highlight POC v0.10

Edge POC:

`Camera / RTSP / video -> dual-view pose + tracking -> court geometry -> serve state machine -> ball tracking -> short-horizon trajectory prediction -> stable 9:16 portrait highlight -> MP4`

## v0.10: predict direction before the portrait camera moves

The portrait camera no longer reacts only to the ball's current pixel. Recent ball observations are mapped through the calibrated court homography into a canonical 10 m x 20 m padel court. The POC estimates the new motion vector and predicts roughly **0.3 s ahead**.

The prediction is intentionally simple and useful for camera direction:

- `LEFT`
- `CENTER`
- `RIGHT`

The source preview shows:

- a yellow circle = current ball estimate;
- a magenta circle = predicted short-horizon position;
- a magenta arrow = current -> predicted direction;
- label such as `LEFT 74%` = predicted court side and confidence.

When a sharp trajectory change is observed, the predictor drops the old history and restarts on the new direction. This is intended to react quickly after a **serve, smash, volley, wall contact, or bounce** instead of averaging the old and new paths.

Important: this is currently **post-impact trajectory prediction**, not true pre-impact intent prediction. It needs a few observed positions after the ball changes direction. That is enough to make the virtual camera anticipate the next area of play without requiring a trained shot-intent model yet.

### Trajectory tuning

Defaults are built into the code. Optional overrides:

```yaml
portrait_output:
  # Keep framing broad and stable.
  crop_height_ratio: 1.0
  safe_zone_ratio: 0.24
  prediction_safe_zone_ratio: 0.20

  # Camera motion remains intentionally slow/eased.
  pan_time_constant: 0.72
  prediction_pan_time_constant: 0.82
  player_pan_time_constant: 0.92
  recenter_time_constant: 1.80
  max_pan_speed_ratio: 0.62

  # How far toward the predicted point the director looks.
  prediction_lead_weight: 0.70
  prediction_min_confidence: 0.38

  trajectory:
    horizon_seconds: 0.32
    max_history: 8
    min_points: 3
    min_span_seconds: 0.055
    min_speed_mps: 1.2
    max_speed_mps: 55.0
    reset_angle_deg: 52.0
    reset_speed_ratio: 2.8
    hold_seconds: 0.20
    min_confidence: 0.30
```

If portrait movement still feels too active, increase `prediction_safe_zone_ratio` to `0.24` or `prediction_pan_time_constant` to `1.0`.

## Normal-speed highlight output

Output is a highlight clip, **not a speed ramp**. Recorder defaults preserve source playback speed. A 59.94 fps source remains normal-time unless you intentionally configure another workflow.

```yaml
recorder:
  preserve_source_speed: true
  pre_roll_seconds: 3.0
  post_roll_seconds: 7.0
  min_clip_seconds: 5.0
  max_clip_seconds: 15.0
```

## Far-side player detection

The camera view is split into independent near/far AI crops. Each side keeps its own YOLO Pose + ByteTrack state, with the far court using a lower detection threshold and larger inference input.

Optional overrides:

```yaml
tracking:
  dual_view: true
  frame_stride: 2
  view_padding: 0.06
  near_conf: 0.28
  far_conf: 0.18
  near_imgsz: 768
  far_imgsz: 1024
```

## Court geometry

Calibrate once for every fixed camera angle:

```bash
python tools/calibrate_zones.py --source padel.mp4 --seek 0
```

Click in this exact order:

1. far-left back corner
2. far-right back corner
3. near-right back corner
4. near-left back corner

The same homography is now used both for perspective service zones and trajectory prediction.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

```bash
python main.py --source padel.mp4
```

Outputs:

```text
clips/                 original landscape highlights
clips/portrait/        9:16 portrait highlights
```

## Current serve detector

The heuristic sequence remains:

`IN_ZONE -> PREPARING -> SWING -> MOVE TOWARD NET -> SERVE`

The next major ML upgrade would be a learned ball detector / shot classifier, while keeping the same trajectory-director and recorder pipeline.
