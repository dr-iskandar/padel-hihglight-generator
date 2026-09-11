# Padel Auto-Highlight POC v0.2

Edge-computing proof-of-concept:

`Camera / RTSP / video -> court ROI -> player pose + tracking -> temporal serve state machine -> ring buffer -> automatic MP4 clip`

## What changed in v0.2

The original single-threshold serve trigger was replaced with a temporal state machine:

`IN_ZONE -> PREPARING -> SWING -> MOVE TOWARD NET -> SERVE`

Other changes:

- Player court position uses **feet / bottom-center of bounding box**.
- **Global cooldown** suppresses duplicate serve events caused by a tracking-ID switch.
- AI can run only on a **court ROI**, while saved clips remain full-resolution/full-frame.
- `device: auto` chooses NVIDIA CUDA, then Apple Silicon MPS, then CPU.
- `frame_stride` reduces inference load without dropping frames from the video recorder.
- Preview shows each tracked player's serve state and an interpretable serve-confidence score.
- Added a court ROI calibration tool.

> The displayed confidence is an interpretable heuristic score, not a calibrated ML probability.

## Install

```bash
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\\Scripts\\activate      # Windows

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

Saved clips appear in `clips/`.

## 1. Calibrate the AI court ROI

For the current sample, the default ROI should be usable, but for a new camera run:

```bash
python tools/calibrate_court_roi.py --source padel.mp4 --seek 30
```

Draw one rectangle around the court / playable area that should be sent to the pose model.
The resulting normalized rectangle is written to `config.yaml` as `court_roi`.

## 2. Calibrate service zones

```bash
python tools/calibrate_zones.py --source padel.mp4 --seek 30
```

Important: service zones now match the player's **feet**, not the center of the body.
Draw only the back/service areas where a server's feet are expected to be.

## State-machine tuning

The main parameters are in `config.yaml`.

## POC scope

This version focuses on validating the core edge flow: detect likely serve moments and automatically save short MP4 clips with pre-roll and post-roll. Face search, social auto-crop, ball tracking, and learned temporal classification are intentionally deferred until serve-trigger reliability is measured on real court footage.
