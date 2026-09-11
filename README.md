# Padel Auto-Highlight POC v0.4

Edge POC:

`Camera / RTSP / video -> dual-view pose + tracking -> court geometry -> serve state machine -> ring buffer -> automatic MP4 clip`

## v0.4: better far-side player detection

The camera view is now split into two independent AI crops:

- **near half court**
- **far half court**

Each half uses its own YOLO pose instance + ByteTrack state. The far half receives a lower detection threshold and larger inference size, so distant players occupy much more of the AI input than when the full 1920px frame is processed at once.

This improves far-side player/pose detection while keeping the saved highlight video in the original full-frame resolution.

Other v0.4 changes:

- independent near/far tracking ID namespaces;
- detections are accepted only when the player's feet belong to the corresponding court half;
- clean custom bbox/skeleton overlay instead of Ultralytics' large default labels;
- optional debug overlay for near/far AI-view boundaries;
- existing perspective service zones and playable-court filtering remain active.

## Court geometry

The POC uses one **4-point playable-court polygon** and generates all four serve zones from standard padel-court geometry using a perspective transform.

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

For every new camera angle, run:

```bash
python tools/calibrate_zones.py --source padel.mp4 --seek 0
```

Click:

1. far-left back corner
2. far-right back corner
3. near-right back corner
4. near-left back corner

Press **Enter/Space** to save, **R** to reset, or **Q/Esc** to cancel.

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

## Dual-view tuning

v0.4 has these built-in defaults. You can add any of them under the existing `tracking:` section in your local `config.yaml` to override them without re-calibrating the court:

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

If far players are still missed, try:

```yaml
far_conf: 0.15
far_imgsz: 1280
```

If the Mac becomes too slow, first try:

```yaml
frame_stride: 3
far_imgsz: 960
```

Set this to inspect the two AI regions:

```yaml
show_view_split: true
```

## Serve detector sequence

The heuristic temporal state machine remains:

`IN_ZONE -> PREPARING -> SWING -> MOVE TOWARD NET -> SERVE`

A person must have their **feet inside the playable court and inside a perspective service polygon** before entering that sequence.

## Current POC scope

This version focuses on reliable spatial filtering, near/far player pose detection, and serve-triggered recording. Face search, social auto-crop, ball tracking, and a learned temporal serve classifier remain later stages.
