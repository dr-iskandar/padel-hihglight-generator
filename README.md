# Padel Auto-Highlight POC v0.6

Edge POC:

`Camera / RTSP / video -> dual-view pose + tracking -> court geometry -> serve state machine -> ball-guided smart portrait reframe -> automatic MP4 clip`

## v0.6: ball-guided composition, not ball-chasing

The portrait output now follows the **direction of play / ball**, but deliberately does **not** keep the ball glued to the exact center of the frame.

The target behavior is closer to a human camera operator:

- 9:16 portrait output;
- mostly fixed zoom so the court composition stays readable;
- vertical framing stays almost fixed;
- ball primarily guides **left/right panning**;
- a central ball safe-zone prevents tiny ball movements from moving the camera;
- camera pan speed is capped, so a fast ball cannot make the crop whip across the frame;
- ball guidance is blended with player positions to keep useful action context;
- when the ball is briefly lost, the frame holds and then falls back to player/court composition;
- detected server/player is a fallback, not the primary portrait target.

The built-in ball tracker is a lightweight classical-CV POC using yellow/green colour, motion, temporal proximity, and the calibrated playable-court mask. It is intentionally replaceable later by a learned ball detector.

### Portrait defaults

You do not need to change `config.yaml`; these defaults are built into the code. To tune them, add:

```yaml
portrait_output:
  enabled: true
  save_landscape: true
  width: 1080
  height: 1920
  show_preview: true
  show_source_crop: true

  # Keep zoom broad and stable, similar to the reference Shorts framing.
  crop_height_ratio: 0.92

  # Ball may move inside this central horizontal zone without moving camera.
  ball_safezone_ratio: 0.20

  # How strongly the ball influences horizontal framing vs player context.
  ball_weight: 0.78

  # Smooth pan and hard cap on pan velocity.
  pan_smoothing: 0.18
  max_pan_speed_ratio: 0.85

  # Keep using the last ball position briefly through detector misses.
  ball_hold_seconds: 0.55

  # Slow return to player/court composition when ball is lost.
  recenter_smoothing: 0.05
```

If the camera still feels too reactive:

```yaml
ball_safezone_ratio: 0.25
pan_smoothing: 0.12
max_pan_speed_ratio: 0.60
```

If it feels too slow:

```yaml
ball_safezone_ratio: 0.16
pan_smoothing: 0.24
max_pan_speed_ratio: 1.10
```

### Optional ball-tracker tuning

```yaml
ball_tracking:
  enabled: true
  hsv_lower: [18, 70, 105]
  hsv_upper: [48, 255, 255]
  motion_threshold: 14
  min_area: 3
  max_area: 220
  max_radius: 14
  min_confidence: 0.28
```

If the court/video uses a different ball colour or lighting, this HSV range is the first thing to tune.

## v0.4: better far-side player detection

The camera view is split into two independent AI crops:

- **near half court**
- **far half court**

Each half uses its own YOLO pose instance + ByteTrack state. The far half uses a lower detection threshold and larger inference size so distant players occupy more of the AI input.

Built-in defaults can be overridden under `tracking:`:

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

If far players are still missed:

```yaml
far_conf: 0.15
far_imgsz: 1280
```

If the Mac becomes too slow:

```yaml
frame_stride: 3
far_imgsz: 960
```

## Court geometry

The POC uses one **4-point playable-court polygon** and generates all four serve zones using perspective geometry.

Calibrate for every new camera angle:

```bash
python tools/calibrate_zones.py --source padel.mp4 --seek 0
```

Click in this exact order:

1. far-left back corner
2. far-right back corner
3. near-right back corner
4. near-left back corner

Press **Enter/Space** to save, **R** to reset, or **Q/Esc** to cancel.

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
clips/portrait/        9:16 ball-guided portrait highlights
```

## Serve detector sequence

The heuristic temporal state machine is:

`IN_ZONE -> PREPARING -> SWING -> MOVE TOWARD NET -> SERVE`

A player must also have their feet inside the playable court and inside a perspective service polygon before entering the serve sequence.
