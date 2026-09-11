from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from padel_poc.config import load_config
from padel_poc.serve_detector import ServeDetector, ServeStatus
from padel_poc.clip_recorder import ClipRecorder
from padel_poc.court_geometry import derive_service_zones, normalized_polygon_to_pixels
from padel_poc.court_views import (
    derive_half_court_polygons,
    filter_detections_to_polygon,
    polygon_to_crop_rect,
)


ZONE_COLOR = (100, 220, 255)
COURT_COLOR = (70, 255, 120)
ROI_COLOR = (255, 180, 60)
VIEW_COLOR = (180, 180, 180)

SKELETON_EDGES = [
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


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


def draw_view_split(frame, half_polygons):
    if not half_polygons:
        return
    for side, poly in half_polygons.items():
        pts = normalized_polygon_to_pixels(poly, frame.shape)
        cv2.polylines(frame, [pts], True, VIEW_COLOR, 1, cv2.LINE_AA)
        center = np.mean(pts, axis=0).astype(int)
        cv2.putText(
            frame,
            f"{side.upper()} AI VIEW",
            (int(center[0]) - 55, int(center[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            VIEW_COLOR,
            1,
            cv2.LINE_AA,
        )


def map_result_to_full_frame(result, crop_rect, id_offset=0):
    """Return tracked boxes/keypoints in original full-frame coordinates."""
    x0, y0, _, _ = crop_rect
    if result.boxes is None or result.keypoints is None or result.boxes.id is None:
        return [], np.empty((0, 4)), np.empty((0, 17, 2)), np.empty((0, 17))

    ids = result.boxes.id.int().cpu().tolist()
    ids = [int(track_id) + int(id_offset) for track_id in ids]
    boxes = result.boxes.xyxy.cpu().numpy().copy()
    xy = result.keypoints.xy.cpu().numpy().copy()
    conf = result.keypoints.conf.cpu().numpy().copy()

    boxes[:, [0, 2]] += x0
    boxes[:, [1, 3]] += y0
    xy[:, :, 0] += x0
    xy[:, :, 1] += y0
    return ids, boxes, xy, conf


def run_view_inference(
    model,
    frame,
    crop_rect,
    half_polygon,
    frame_shape,
    tracker_name,
    device,
    conf_threshold,
    iou_threshold,
    imgsz,
    max_det,
    id_offset,
):
    x1, y1, x2, y2 = crop_rect
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return [], np.empty((0, 4)), np.empty((0, 17, 2)), np.empty((0, 17))

    result = model.track(
        crop,
        persist=True,
        tracker=tracker_name,
        conf=float(conf_threshold),
        iou=float(iou_threshold),
        imgsz=int(imgsz),
        max_det=int(max_det),
        device=device,
        verbose=False,
    )[0]

    ids, boxes, xy, kp_conf = map_result_to_full_frame(result, crop_rect, id_offset=id_offset)
    return filter_detections_to_polygon(
        ids,
        boxes,
        xy,
        kp_conf,
        half_polygon,
        frame_shape,
    )


def combine_detection_parts(parts):
    ids = []
    boxes_parts = []
    xy_parts = []
    conf_parts = []
    sides = []

    for side, (part_ids, part_boxes, part_xy, part_conf) in parts:
        ids.extend(part_ids)
        if len(part_boxes):
            boxes_parts.append(part_boxes)
            xy_parts.append(part_xy)
            conf_parts.append(part_conf)
            sides.extend([side] * len(part_ids))

    boxes = np.concatenate(boxes_parts, axis=0) if boxes_parts else np.empty((0, 4), dtype=np.float32)
    xy = np.concatenate(xy_parts, axis=0) if xy_parts else np.empty((0, 17, 2), dtype=np.float32)
    conf = np.concatenate(conf_parts, axis=0) if conf_parts else np.empty((0, 17), dtype=np.float32)
    return ids, boxes, xy, conf, sides


def state_color(status: ServeStatus):
    if status.state == "COOLDOWN":
        return (0, 255, 0)
    if status.state == "SWING":
        return (0, 165, 255)
    if status.state == "PREPARING":
        return (0, 255, 255)
    if status.state == "IN_ZONE":
        return (255, 220, 100)
    return (230, 230, 230)


def format_track_label(track_id: int, side: str, near_offset: int, far_offset: int) -> str:
    if side == "far":
        return f"F{max(0, int(track_id) - int(far_offset))}"
    if side == "near":
        return f"N{max(0, int(track_id) - int(near_offset))}"
    return f"T{track_id}"


def draw_skeleton(frame, keypoints_xy, keypoints_conf, min_conf=0.30, color=(0, 255, 255)):
    for a, b in SKELETON_EDGES:
        if a >= len(keypoints_xy) or b >= len(keypoints_xy):
            continue
        if keypoints_conf[a] < min_conf or keypoints_conf[b] < min_conf:
            continue
        pa = tuple(np.rint(keypoints_xy[a]).astype(int))
        pb = tuple(np.rint(keypoints_xy[b]).astype(int))
        cv2.line(frame, pa, pb, color, 2, cv2.LINE_AA)

    for i, point in enumerate(keypoints_xy):
        if i >= len(keypoints_conf) or keypoints_conf[i] < min_conf:
            continue
        p = tuple(np.rint(point).astype(int))
        cv2.circle(frame, p, 3, color, -1, cv2.LINE_AA)


def draw_player_overlay(
    frame,
    bbox,
    keypoints_xy,
    keypoints_conf,
    status: ServeStatus,
    side: str,
    near_offset: int,
    far_offset: int,
    min_kp_conf: float,
):
    x1, y1, x2, y2 = map(int, bbox)
    color = state_color(status)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    draw_skeleton(frame, keypoints_xy, keypoints_conf, min_conf=min_kp_conf, color=color)

    track_label = format_track_label(status.track_id, side, near_offset, far_offset)
    zone = status.zone or "-"
    title = f"{track_label} | {zone} | {status.state}"
    detail = f"serve {int(round(status.confidence * 100)):02d}%"

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.46
    thickness = 1
    (tw, th), _ = cv2.getTextSize(title, font, scale, thickness)
    (dw, dh), _ = cv2.getTextSize(detail, font, 0.42, thickness)
    label_w = max(tw, dw) + 12
    label_h = th + dh + 18
    tx = max(2, min(frame.shape[1] - label_w - 2, x1))
    ty = max(label_h + 2, y1)

    cv2.rectangle(frame, (tx, ty - label_h), (tx + label_w, ty), (20, 20, 20), -1)
    cv2.putText(frame, title, (tx + 6, ty - dh - 9), font, scale, color, thickness, cv2.LINE_AA)
    cv2.putText(frame, detail, (tx + 6, ty - 5), font, 0.42, color, thickness, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Padel auto-highlight POC v0.4")
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
    model_path = cfg.get("model", "yolo11n-pose.pt")

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

    dual_view = bool(tracking.get("dual_view", True)) and bool(court_polygon)
    near_offset = int(tracking.get("near_id_offset", 0))
    far_offset = int(tracking.get("far_id_offset", 1000))

    half_polygons = derive_half_court_polygons(court_polygon) if dual_view else None
    view_padding = float(tracking.get("view_padding", 0.04))
    view_rects = (
        {
            side: polygon_to_crop_rect(poly, frame.shape, padding=view_padding)
            for side, poly in half_polygons.items()
        }
        if dual_view
        else None
    )

    # Separate YOLO instances keep ByteTrack state independent for each half court.
    near_model = YOLO(model_path)
    far_model = YOLO(model_path) if dual_view else None

    court_roi = cfg.get("court_roi", [0.0, 0.0, 1.0, 1.0])
    roi_px = normalized_rect(court_roi, frame.shape)

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
    last_xy = np.empty((0, 17, 2), dtype=np.float32)
    last_kp_conf = np.empty((0, 17), dtype=np.float32)
    last_sides = []
    last_statuses = {}

    print(f"Source: {source}")
    print(f"Resolution: {w}x{h} @ {fps:.1f} fps")
    print(f"AI device: {device} | frame stride: {frame_stride} | tracker: {tracker_name}")
    if dual_view:
        print("Inference: DUAL VIEW (near + far court with independent trackers)")
        print(
            f"Near crop: {view_rects['near']} | imgsz={tracking.get('near_imgsz', 768)} | "
            f"conf={tracking.get('near_conf', 0.28)}"
        )
        print(
            f"Far crop:  {view_rects['far']} | imgsz={tracking.get('far_imgsz', 1024)} | "
            f"conf={tracking.get('far_conf', 0.20)}"
        )
    elif court_polygon:
        print("Inference: SINGLE VIEW with court geometry")
    else:
        print("Inference: SINGLE VIEW legacy mode (run tools/calibrate_zones.py)")
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

            do_inference = frame_index % frame_stride == 0
            if do_inference:
                if dual_view:
                    near_part = run_view_inference(
                        near_model,
                        frame,
                        view_rects["near"],
                        half_polygons["near"],
                        frame.shape,
                        tracker_name,
                        device,
                        tracking.get("near_conf", tracking.get("conf", 0.28)),
                        tracking.get("iou", 0.45),
                        tracking.get("near_imgsz", 768),
                        tracking.get("max_det_per_view", 6),
                        near_offset,
                    )
                    far_part = run_view_inference(
                        far_model,
                        frame,
                        view_rects["far"],
                        half_polygons["far"],
                        frame.shape,
                        tracker_name,
                        device,
                        tracking.get("far_conf", 0.20),
                        tracking.get("iou", 0.45),
                        tracking.get("far_imgsz", 1024),
                        tracking.get("max_det_per_view", 6),
                        far_offset,
                    )
                    ids, boxes, xy, kp_conf, sides = combine_detection_parts(
                        [("near", near_part), ("far", far_part)]
                    )
                else:
                    rx1, ry1, rx2, ry2 = roi_px
                    crop = frame[ry1:ry2, rx1:rx2]
                    result = near_model.track(
                        crop,
                        persist=True,
                        tracker=tracker_name,
                        conf=float(tracking.get("conf", 0.30)),
                        iou=float(tracking.get("iou", 0.45)),
                        imgsz=int(tracking.get("imgsz", 960)),
                        max_det=int(tracking.get("max_det", 8)),
                        device=device,
                        verbose=False,
                    )[0]
                    ids, boxes, xy, kp_conf = map_result_to_full_frame(result, roi_px, id_offset=near_offset)
                    sides = ["single"] * len(ids)

                count = min(len(ids), len(boxes), len(xy), len(kp_conf), len(sides))
                active_ids = []
                active_sides = []
                statuses = {}

                for i in range(count):
                    track_id = int(ids[i])
                    side = sides[i]
                    active_ids.append(track_id)
                    active_sides.append(side)
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
                        track_label = format_track_label(event.track_id, side, near_offset, far_offset).lower()
                        clip_path = recorder.trigger(frame_index, label=f"serve_{track_label}")
                        last_event_text = (
                            f"SERVE | {track_label.upper()} | {event.zone} | "
                            f"{event.confidence:.0%} | {clip_path.name}"
                        )
                        last_event_until = time.monotonic() + 2.5
                        print(last_event_text)

                serve_detector.prune(frame_index)
                last_boxes = boxes[:count]
                last_ids = active_ids
                last_xy = xy[:count]
                last_kp_conf = kp_conf[:count]
                last_sides = active_sides
                last_statuses = statuses

            if preview:
                if not dual_view:
                    draw_roi(vis, roi_px)
                draw_court_polygon(vis, court_polygon)
                draw_zones(vis, serve_zones)
                if bool(tracking.get("show_view_split", False)) and dual_view:
                    draw_view_split(vis, half_polygons)

                min_kp_conf = float(cfg.get("serve_detector", {}).get("min_keypoint_conf", 0.30))
                for i, track_id in enumerate(last_ids):
                    if i >= len(last_boxes) or i >= len(last_xy) or i >= len(last_kp_conf):
                        break
                    status = last_statuses.get(track_id, serve_detector.get_status(track_id))
                    side = last_sides[i] if i < len(last_sides) else "single"
                    draw_player_overlay(
                        vis,
                        last_boxes[i],
                        last_xy[i],
                        last_kp_conf[i],
                        status,
                        side,
                        near_offset,
                        far_offset,
                        min_kp_conf,
                    )

            now = time.perf_counter()
            inst = 1.0 / max(1e-6, now - t_prev)
            t_prev = now
            fps_ema = inst if fps_ema == 0 else 0.90 * fps_ema + 0.10 * inst

            if preview:
                ai_mode = "DUAL" if dual_view else "SINGLE"
                status_line = (
                    f"LOOP {fps_ema:.1f} FPS | SRC {fps:.1f} | "
                    f"AI {ai_mode} /{frame_stride} | {'RECORDING CLIP' if recorder.active else 'BUFFERING'}"
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
                    cv2.rectangle(vis, (10, 43), (min(w - 10, 1040), 82), (0, 0, 0), -1)
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

                cv2.imshow("Padel Highlight POC v0.4", vis)
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
