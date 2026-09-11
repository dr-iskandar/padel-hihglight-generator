from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

from padel_poc.court_geometry import derive_service_zones, normalized_polygon_to_pixels


POINT_NAMES = ["FAR LEFT", "FAR RIGHT", "NEAR RIGHT", "NEAR LEFT"]
ZONE_COLOR = (0, 220, 255)
COURT_COLOR = (80, 255, 120)
POINT_COLOR = (255, 255, 255)


def parse_source(text):
    text = str(text).strip()
    return int(text) if text.isdigit() else text


def main():
    ap = argparse.ArgumentParser(description="Calibrate perspective-correct padel court + serve zones")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--source", default=None)
    ap.add_argument("--seek", type=float, default=0.0, help="Seek to N seconds before calibration")
    ap.add_argument("--display-width", type=int, default=1400)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    source = parse_source(args.source if args.source is not None else cfg.get("source", 0))

    cap = cv2.VideoCapture(source)
    if args.seek > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.seek * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read from source: {source}")

    h, w = frame.shape[:2]
    scale = min(1.0, float(args.display_width) / max(1, w), 850.0 / max(1, h))
    dw, dh = int(round(w * scale)), int(round(h * scale))
    display_base = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA) if scale < 1 else frame.copy()

    geometry_cfg = cfg.setdefault("court_geometry", {})
    service_depth_m = float(geometry_cfg.get("service_depth_m", 3.3))
    center_gap_m = float(geometry_cfg.get("center_gap_m", 0.0))
    side_inset_m = float(geometry_cfg.get("side_inset_m", 0.0))

    points: list[tuple[int, int]] = []
    window = "Court Geometry Calibration"

    def mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, mouse)

    print("Click exactly 4 PLAYABLE-COURT floor corners in this order:")
    print("  1) FAR LEFT  2) FAR RIGHT  3) NEAR RIGHT  4) NEAR LEFT")
    print("Use the back-wall/floor corner of the playable court, not the outside glass structure.")
    print("R = reset | ENTER/SPACE = save after 4 points | Q/ESC = cancel")

    while True:
        vis = display_base.copy()

        for i, point in enumerate(points):
            cv2.circle(vis, point, 7, POINT_COLOR, -1, cv2.LINE_AA)
            cv2.putText(
                vis,
                f"{i + 1} {POINT_NAMES[i]}",
                (point[0] + 10, max(24, point[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                POINT_COLOR,
                2,
                cv2.LINE_AA,
            )

        if len(points) >= 2:
            cv2.polylines(vis, [np.asarray(points, np.int32)], False, COURT_COLOR, 2, cv2.LINE_AA)

        if len(points) == 4:
            cv2.polylines(vis, [np.asarray(points, np.int32)], True, COURT_COLOR, 3, cv2.LINE_AA)
            polygon_norm = [[p[0] / dw, p[1] / dh] for p in points]
            zones = derive_service_zones(
                polygon_norm,
                service_depth_m=service_depth_m,
                center_gap_m=center_gap_m,
                side_inset_m=side_inset_m,
            )
            for name, poly in zones.items():
                pts = normalized_polygon_to_pixels(poly, vis.shape)
                cv2.polylines(vis, [pts], True, ZONE_COLOR, 2, cv2.LINE_AA)
                anchor = tuple(pts[0])
                cv2.putText(vis, name, (anchor[0] + 4, anchor[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, ZONE_COLOR, 2, cv2.LINE_AA)

        next_text = "READY - ENTER to save" if len(points) == 4 else f"Click {len(points) + 1}/4: {POINT_NAMES[len(points)]}"
        cv2.rectangle(vis, (0, 0), (min(dw, 720), 42), (0, 0, 0), -1)
        cv2.putText(vis, next_text, (14, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.67, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(window, vis)
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            cv2.destroyAllWindows()
            print("Calibration cancelled.")
            return
        if key in (ord("r"), ord("R")):
            points.clear()
        if key in (13, 10, 32) and len(points) == 4:
            break

    cv2.destroyAllWindows()

    court_polygon = [[round(x / dw, 6), round(y / dh, 6)] for x, y in points]
    geometry_cfg["court_polygon"] = court_polygon
    geometry_cfg.setdefault("service_depth_m", 3.3)
    geometry_cfg.setdefault("center_gap_m", 0.0)
    geometry_cfg.setdefault("side_inset_m", 0.0)

    cfg.pop("serve_zones", None)
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    print(f"Saved court polygon to {cfg_path}")
    print("Service zones will now be generated automatically with perspective/homography.")
    print("Run: python main.py --source <your video/camera>")


if __name__ == "__main__":
    main()
