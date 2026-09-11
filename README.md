# Padel Auto-Highlight POC v0.3

Edge POC:

`Camera / RTSP / video -> pose + tracking -> court geometry -> serve state machine -> ring buffer -> automatic MP4 clip`

## v0.3: perspective court geometry

The old rectangular serve boxes are gone. The POC now uses one **4-point playable-court polygon** and generates all four serve zones from standard padel-court geometry using a perspective transform (homography).

That gives us:

- serve zones that have the **same perspective/shape as the court**;
- automatic `near_left`, `near_right`, `far_left`, and `far_right` zones;
- players whose feet are outside the playable court are rejected from the serve state machine;
- one calibration instead of drawing four oversized rectangles manually.

The four court points must be clicked in this exact order:

`FAR LEFT -> FAR RIGHT -> NEAR RIGHT -> NEAR LEFT`

## Install

```bash
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\\Scripts\\activate      # Windows
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Calibrate court + serve zones

For every new camera angle, run this first:

```bash
python tools/calibrate_zones.py --source padel.mp4 --seek 30
```

Click the four **playable floor/back-wall corners**, not the outside glass/support structure:

1. far-left back corner
2. far-right back corner
3. near-right back corner
4. near-left back corner

The preview immediately shows the derived perspective service zones. Press **Enter/Space** to save, **R** to reset, or **Q/Esc** to cancel.

The regulation padel service line is 3 m from the back wall. For the POC, `service_depth_m` defaults to `3.3` to allow a little bbox/pose noise. Set it to `3.0` for exact regulation geometry.

## Optional AI ROI

The AI inference crop is still rectangular for speed, and is independent from the playable-court polygon:

```bash
python tools/calibrate_court_roi.py --source padel.mp4 --seek 30
```

## Run

```bash
python main.py --source padel.mp4
```

Webcam:

```bash
python main.py --source 0
```

RTSP:

```bash
python main.py --source "rtsp://USER:PASSWORD@CAMERA_IP:554/stream"
```

Saved clips appear in `clips/`.

## Serve detector sequence

The heuristic temporal state machine remains:

`IN_ZONE -> PREPARING -> SWING -> MOVE TOWARD NET -> SERVE`

A person must now also have their **feet inside the playable court and inside a perspective service polygon** before entering that sequence.

## Current POC scope

This version focuses on reliable spatial filtering + serve-triggered recording. Face search, social auto-crop, ball tracking, and a learned temporal serve classifier remain later stages.
