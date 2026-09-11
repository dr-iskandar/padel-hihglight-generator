from __future__ import annotations

import argparse
from pathlib import Path
import cv2
import yaml


def parse_source(text):
    text = str(text).strip()
    return int(text) if text.isdigit() else text


def main():
    ap = argparse.ArgumentParser(description="Draw serve zones on one camera frame")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--source", default=None)
    ap.add_argument("--seek", type=float, default=0.0, help="Seek to N seconds before calibration")
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

    names = ["near_left", "near_right", "far_left", "far_right"]
    h, w = frame.shape[:2]
    zones = {}

    print("IMPORTANT: zones are matched against the player's FEET / bottom-center of the box.")
    print("Draw only the back/service area where a server's feet would be.")

    for name in names:
        print(f"Draw rectangle for {name}, then press ENTER/SPACE. Press C to cancel selection.")
        x, y, rw, rh = cv2.selectROI(f"Select {name}", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(f"Select {name}")
        if rw <= 0 or rh <= 0:
            print(f"Skipped {name}")
            continue
        zones[name] = [round(x / w, 4), round(y / h, 4), round((x + rw) / w, 4), round((y + rh) / h, 4)]

    if not zones:
        print("No zones saved.")
        return

    cfg["serve_zones"] = zones
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"Saved {len(zones)} zones to {cfg_path}")


if __name__ == "__main__":
    main()
