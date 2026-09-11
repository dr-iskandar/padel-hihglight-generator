# Padel Auto-Highlight POC v0.5

Edge POC:

`Camera / RTSP / video -> dual-view pose + tracking -> court geometry -> serve state machine -> smart portrait reframe -> automatic MP4 clip`

## v0.5: Smart Portrait Reframe

The POC now creates a **9:16 portrait highlight** in addition to the original landscape clip.

The portrait output behaves like a virtual camera:

- follows the most relevant player;
- prioritizes `PREPARING`, `SWING`, and recent serve activity;
- locks onto the detected server when a serve event fires;
- smoothly pans instead of jumping between players;
- smoothly zooms based on player size;
- uses a dead-zone to reduce camera jitter;
- keeps the original full-resolution landscape clip as an optional backup.

Default portrait output:

```text
1080 x 1920
clips/portrait/
```

A second preview window named **Portrait Output 9:16** is shown while the POC runs.
The landscape preview also shows the current portrait crop rectangle.

### Optional portrait tuning

You do **not** need to edit `config.yaml` for portrait output to work. These are built-in defaults. Add a `portrait_output:` section only when you want to tune it:

```yaml
portrait_output:
  enabled: true
  save_landscape: true
  width: 1080
  height: 1920
  show_preview: true
  show_source_crop: true

  # Lower value = wider shot / more context.
  # Higher value = tighter zoom on player.
  subject_height_ratio: 0.38

  # Limits how far the virtual camera may zoom.
  min_crop_height_ratio: 0.40
  max_crop_height_ratio: 1.00

  # Motion smoothing. Increase for faster camera response.
  center_smoothing: 0.20
  zoom_smoothing: 0.14

  # Prevent tiny pose movements from making the crop shake.
  deadzone_ratio: 0.06

  # Keeps following the detected server after the serve event.
  lock_seconds: 5.0
```

If the framing is too tight, try:

```yaml
subject_height_ratio: 0.30
min_crop_height_ratio: 0.50
```

If the virtual camera follows too slowly:

```yaml
center_smoothing: 0.30
zoom_smoothing: 0.20
```

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

Webcam:

```bash
python main.py --source 0
```

RTSP:

```bash
python main.py --source "rtsp://USER:PASSWORD@CAMERA_IP:554/stream"
```

Outputs:

```text
clips/                 original landscape highlights
clips/portrait/        9:16 following + zoom highlights
```

## Serve detector sequence

The heuristic temporal state machine is:

`IN_ZONE -> PREPARING -> SWING -> MOVE TOWARD NET -> SERVE`

A player must also have their feet inside the playable court and inside a perspective service polygon before entering the serve sequence.
