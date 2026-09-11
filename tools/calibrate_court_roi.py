from __future__ import annotations

import argparse
from pathlib import Path
import cv2
import yaml


def parse_source(text):
    text = str(text).strip()
    return int(text) if text.isdigit() else text


def main():
    ap = argparse.ArgumentParser(description="Select the court ROI used for AI inference")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--source", default=None)
    ap.add_argument("--seek", type=float, default=0.0, help="Seek to N seconds before showing a frame")
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
    print("Draw one rectangle around the playable court / area that should be sent to AI.")
    x, y, rw, rh = cv2.selectROI("Select AI court ROI", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    if rw <= 0 or rh <= 0:
        print("No ROI saved.")
        return

    cfg["court_roi"] = [round(x / w, 4), round(y / h, 4), round((x + rw) / w, 4), round((y + rh) / h, 4)]
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"Saved court_roi={cfg['court_roi']} to {cfg_path}")


if __name__ == "__main__":
    main()
