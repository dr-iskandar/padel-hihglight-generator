from __future__ import annotations

import argparse
import time
from typing import Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from padel_poc.config import load_config
from padel_poc.serve_detector import ServeDetector, ServeStatus
from padel_poc.clip_recorder import ClipRecorder
from padel_poc.court_geometry import derive_service_zones, normalized_polygon_to_pixels


SKELETON_COLOR = (0, 255, 255)
ZONE_COLOR = (100, 220, 255)
COURT_COLOR = (70, 255, 120)
ROI_COLOR = (255, 180, 60)


def parse_source(value):
    if isinstance(value, int):
        return value
    text = str(value).strip()
    return int(text) if text.isdigit() else text


def choose_device(requested: str):
    requested = str(requested or "auto").strip().lower()
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return 0
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def normalized_rect(rect, frame_shape):
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = map(float, rect)
    x1 = max(0, min(w - 1, int(round(x1 * w))))
    y1 = max(0, min(h - 1, int(round(y1 * h))))
    x2 = max(x1 + 1, min(w, int(round(x2 * w))))
    y2 = max(y1 + 1, min(h, int(round(y2 * h))))
    return x1, y1, x2, y2


def draw_zones(frame, zones):
    for name, points in zones.items():
        pts = normalized_polygon_to_pixels(points, frame.shape)
        cv2.polylines(frame, [pts], True, ZONE_COLOR, 2, cv2.LINE_AA)
        anchor = tuple(pts[0])
        cv2.putText(
            frame,
            name,
            (int(anchor[0]) + 4, max(20, int(anchor[1]) + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            ZONE_COLOR,
            2,
            cv2.LINE_AA,
        )


def draw_court_polygon(frame, court_polygon):
    if not court_polygon:
        return
    pts = normalized_polygon_to_pixels(court_polygon, frame.shape)
    cv2.polylines(frame, [pts], True, COURT_COLOR, 2, cv2.LINE_AA)
    cv2.putText(
        frame,
        "PLAYABLE COURT",
        (int(pts[0][0]) + 6, max(22, int(pts[0][1]) - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        COURT_COLOR,
        2,
        cv2.LINE_AA,
    )


def draw_roi(frame, roi_px):
    x1, y1, x2, y2 = roi_px
    cv2.rectangle(frame, (x1, y1), (x2, y2), ROI_COLOR, 1)
    cv2.putText(
        frame,
        "AI ROI",
        (x1 + 5, max(20, y1 + 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        ROI_COLOR,
        1,
        cv2.LINE_AA,
    )


def map_result_to_full_frame(result, roi_px):
    """Return boxes/keypoints in original full-frame coordinates."""
    x0, y0, _, _ = roi_px
    if result.boxes is None or result.keypoints is None or result.boxes.id is None:
        return [], np.empty((0, 4)), np.empty((0, 17, 2)), np.empty((0, 17))

    ids = result.boxes.id.int().cpu().tolist()
    boxes = result.boxes.xyxy.cpu().numpy().copy()
    xy = result.keypoints.xy.cpu().numpy().copy()
    conf = result.keypoints.conf.cpu().numpy().copy()

    boxes[:, [0, 2]] += x0
    boxes[:, [1, 3]] += y0
    xy[:, :, 0] += x0
    xy[:, :, 1] += y0
    return ids, boxes, xy, conf


def draw_player_status(frame, bbox, status: ServeStatus):
    x1, y1, x2, y2 = map(int, bbox)
    if status.state == "COOLDOWN":
        state_color = (0, 255, 0)
    elif status.state == "SWING":
        state_color = (0, 165, 255)
    elif status.state == "PREPARING":
        state_color = (0, 255, 255)
    else:
        state_color = (255, 255, 255)

    label = f"T{status.track_id} {status.zone or '-'} {status.state}"
    score = f"serve {int(round(status.confidence * 100)):02d}% | wrist {status.wrist_speed:.1f}"

    tx = max(4, x1)
    ty = max(42, y1 - 26)
    cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.48, state_color, 2, cv2.LINE_AA)
    cv2.putText(frame, score, (tx, ty + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.43, state_color, 1, cv2.LINE_AA)

    meter_w = max(60, min(140, x2 - x1))
    meter_h = 6
    my = max(2, ty + 24)
    cv2.rectangle(frame, (tx, my), (tx + meter_w, my + meter_h), (80, 80, 80), -1)
    cv2.rectangle(
        frame,
        (tx, my),
        (tx + int(meter_w * status.confidence), my + meter_h),
        state_color,
        -1,
    )


def main():
    parser = argparse.ArgumentParser(description="Padel auto-highlight POC v0.3")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--source", default=None, help="Override source: 0, file path, or RTSP URL")
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    source = parse_source(args.source if args.source is not None else cfg.get("source", 0))

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1 or fps > 240:
        fps = 30.0

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Could not read first frame")
    h, w = frame.shape[:2]

    tracking = cfg.get("tracking", {})
    device = choose_device(tracking.get("device", "auto"))
    frame_stride = max(1, int(tracking.get("frame_stride", 1)))
    tracker_name = str(tracking.get("tracker", "bytetrack.yaml"))
    model = YOLO(cfg.get("model", "yolo11n-pose.pt"))

    court_roi = cfg.get("court_roi", [0.0, 0.0, 1.0, 1.0])
    roi_px = normalized_rect(court_roi, frame.shape)

    geometry_cfg = cfg.get("court_geometry", {})
    court_polygon = geometry_cfg.get("court_polygon")
    if court_polygon:
        serve_zones = derive_service_zones(
            court_polygon,
            service_depth_m=float(geometry_cfg.get("service_depth_m", 3.3)),
            center_gap_m=float(geometry_cfg.get("center_gap_m", 0.0)),
            side_inset_m=float(geometry_cfg.get("side_inset_m", 0.0)),
        )
    else:
        serve_zones = cfg.get("serve_zones", {})

    serve_detector = ServeDetector(
        serve_zones,
        cfg.get("serve_detector", {}),
        fps,
        court_polygon=court_polygon,
    )
    recorder = ClipRecorder(cfg.get("output_dir", "clips"), fps, (w, h), cfg.get("recorder", {}))

    preview = bool(cfg.get("show_preview", True)) and not args.no_preview
    frame_index = 0
    last_event_text = ""
    last_event_until = 0.0
    fps_ema = 0.0
    t_prev = time.perf_counter()

    last_boxes = np.empty((0, 4), dtype=np.float32)
    last_ids = []
    last_statuses = {}
    last_annotated_crop: Optional[np.ndarray] = None

    print(f"Source: {source}")
    print(f"Resolution: {w}x{h} @ {fps:.1f} fps")
    print(f"AI device: {device} | frame stride: {frame_stride} | tracker: {tracker_name}")
    if court_polygon:
        print("Court geometry: polygon + perspective-generated service zones")
    else:
        print("Court geometry: legacy serve zones (run tools/calibrate_zones.py)")
    print("Press Q or ESC to quit.")

    pending_frame = frame
    try:
        while True:
            if pending_frame is not None:
                frame = pending_frame
                pending_frame = None
            else:
                ok, frame = cap.read()
                if not ok:
                    break

            raw_frame = frame.copy()
            recorder.push(raw_frame, frame_index)
            vis = frame.copy()
            annotated_crop_this_frame: Optional[np.ndarray] = None

            do_inference = frame_index % frame_stride == 0
            if do_inference:
                rx1, ry1, rx2, ry2 = roi_px
                crop = frame[ry1:ry2, rx1:rx2]

                results = model.track(
                    crop,
                    persist=True,
                    tracker=tracker_name,
                    conf=float(tracking.get("conf", 0.30)),
                    iou=float(tracking.get("iou", 0.45)),
                    imgsz=int(tracking.get("imgsz", 960)),
                    max_det=int(tracking.get("max_det", 8)),
                    device=device,
                    verbose=False,
                )

                result = results[0]
                if preview:
                    try:
                        last_annotated_crop = result.plot()
                    except Exception:
                        last_annotated_crop = crop.copy()
                    annotated_crop_this_frame = last_annotated_crop

                ids, boxes, xy, kp_conf = map_result_to_full_frame(result, roi_px)
                count = min(len(ids), len(boxes), len(xy), len(kp_conf))
                active_ids = []
                statuses = {}

                for i in range(count):
                    track_id = int(ids[i])
                    active_ids.append(track_id)
                    kp = np.concatenate([xy[i], kp_conf[i][:, None]], axis=1)
                    event = serve_detector.update(
                        track_id=track_id,
                        bbox_xyxy=boxes[i],
                        keypoints=kp,
                        frame_shape=frame.shape,
                        frame_index=frame_index,
                    )
                    status = serve_detector.get_status(track_id)
                    statuses[track_id] = status

                    if event is not None:
                        clip_path = recorder.trigger(frame_index, label=f"serve_t{event.track_id}")
                        last_event_text = (
                            f"SERVE | T{event.track_id} | {event.zone} | "
                            f"{event.confidence:.0%} | {clip_path.name}"
                        )
                        last_event_until = time.monotonic() + 2.5
                        print(last_event_text)

                serve_detector.prune(frame_index)
                last_boxes = boxes[:count]
                last_ids = active_ids
                last_statuses = statuses

            if preview:
                if annotated_crop_this_frame is not None:
                    rx1, ry1, rx2, ry2 = roi_px
                    if annotated_crop_this_frame.shape[:2] == (ry2 - ry1, rx2 - rx1):
                        vis[ry1:ry2, rx1:rx2] = annotated_crop_this_frame

                draw_roi(vis, roi_px)
                draw_court_polygon(vis, court_polygon)
                draw_zones(vis, serve_zones)

                for i, track_id in enumerate(last_ids):
                    if i >= len(last_boxes):
                        break
                    status = last_statuses.get(track_id, serve_detector.get_status(track_id))
                    draw_player_status(vis, last_boxes[i], status)

            now = time.perf_counter()
            inst = 1.0 / max(1e-6, now - t_prev)
            t_prev = now
            fps_ema = inst if fps_ema == 0 else 0.90 * fps_ema + 0.10 * inst

            if preview:
                status_line = (
                    f"LOOP {fps_ema:.1f} FPS | SRC {fps:.1f} | "
                    f"AI /{frame_stride} | {'RECORDING CLIP' if recorder.active else 'BUFFERING'}"
                )
                cv2.putText(
                    vis,
                    status_line,
                    (18, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.68,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                if time.monotonic() < last_event_until:
                    cv2.rectangle(vis, (10, 43), (min(w - 10, 920), 82), (0, 0, 0), -1)
                    cv2.putText(
                        vis,
                        last_event_text,
                        (18, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.58,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                cv2.imshow("Padel Highlight POC v0.3", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break

            frame_index += 1
    finally:
        recorder.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
