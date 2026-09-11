from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np

from padel_poc.court_geometry import COURT_LENGTH_M, COURT_WIDTH_M, project_world_polygon


def derive_half_court_polygons(
    court_polygon: Sequence[Sequence[float]],
) -> Dict[str, List[List[float]]]:
    """Return perspective-correct near/far half-court polygons.

    Input/output coordinates are normalized full-frame [x, y].
    Court polygon order is far_left, far_right, near_right, near_left.
    """
    mid = COURT_LENGTH_M / 2.0
    far_world = [
        [0.0, 0.0],
        [COURT_WIDTH_M, 0.0],
        [COURT_WIDTH_M, mid],
        [0.0, mid],
    ]
    near_world = [
        [0.0, mid],
        [COURT_WIDTH_M, mid],
        [COURT_WIDTH_M, COURT_LENGTH_M],
        [0.0, COURT_LENGTH_M],
    ]
    return {
        "far": project_world_polygon(court_polygon, far_world),
        "near": project_world_polygon(court_polygon, near_world),
    }


def polygon_to_crop_rect(
    polygon: Sequence[Sequence[float]],
    frame_shape,
    padding: float = 0.04,
) -> Tuple[int, int, int, int]:
    """Convert a normalized polygon to a padded pixel bounding rectangle."""
    h, w = frame_shape[:2]
    pts = np.asarray(polygon, dtype=np.float32)
    xs = pts[:, 0] * w
    ys = pts[:, 1] * h

    x1 = float(xs.min())
    y1 = float(ys.min())
    x2 = float(xs.max())
    y2 = float(ys.max())

    pad_x = max(2.0, (x2 - x1) * float(padding))
    pad_y = max(2.0, (y2 - y1) * float(padding))

    x1 = max(0, int(np.floor(x1 - pad_x)))
    y1 = max(0, int(np.floor(y1 - pad_y)))
    x2 = min(w, int(np.ceil(x2 + pad_x)))
    y2 = min(h, int(np.ceil(y2 + pad_y)))

    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid crop rectangle derived from court polygon")
    return x1, y1, x2, y2


def point_in_normalized_polygon(
    x: float,
    y: float,
    polygon: Sequence[Sequence[float]],
) -> bool:
    pts = np.asarray(polygon, dtype=np.float32)
    return cv2.pointPolygonTest(pts, (float(x), float(y)), False) >= 0


def filter_detections_to_polygon(
    ids: Iterable[int],
    boxes: np.ndarray,
    keypoints_xy: np.ndarray,
    keypoints_conf: np.ndarray,
    polygon: Sequence[Sequence[float]],
    frame_shape,
):
    """Keep detections whose feet (bbox bottom-center) are inside polygon."""
    h, w = frame_shape[:2]
    ids = list(ids)
    keep = []
    for i, box in enumerate(boxes):
        x1, _y1, x2, y2 = map(float, box)
        foot_x = (x1 + x2) * 0.5 / max(1.0, float(w))
        foot_y = y2 / max(1.0, float(h))
        if point_in_normalized_polygon(foot_x, foot_y, polygon):
            keep.append(i)

    if not keep:
        return (
            [],
            np.empty((0, 4), dtype=np.float32),
            np.empty((0, 17, 2), dtype=np.float32),
            np.empty((0, 17), dtype=np.float32),
        )

    idx = np.asarray(keep, dtype=np.int32)
    return (
        [ids[i] for i in keep],
        boxes[idx],
        keypoints_xy[idx],
        keypoints_conf[idx],
    )
