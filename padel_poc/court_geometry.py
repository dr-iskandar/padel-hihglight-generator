from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import cv2
import numpy as np

COURT_WIDTH_M = 10.0
COURT_LENGTH_M = 20.0


def _quad(points: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    if arr.shape != (4, 2):
        raise ValueError("court_polygon must contain exactly 4 [x, y] points")
    return arr


def project_world_polygon(
    court_polygon: Sequence[Sequence[float]],
    world_polygon: Sequence[Sequence[float]],
) -> List[List[float]]:
    """Project canonical padel-court coordinates into normalized image coordinates.

    court_polygon order must be:
        far_left, far_right, near_right, near_left

    Canonical world coordinates are measured in meters with:
        (0, 0)          = far-left back corner
        (10, 0)         = far-right back corner
        (10, 20)        = near-right back corner
        (0, 20)         = near-left back corner
    """
    image_quad = _quad(court_polygon)
    world_quad = np.asarray(
        [[0.0, 0.0], [COURT_WIDTH_M, 0.0], [COURT_WIDTH_M, COURT_LENGTH_M], [0.0, COURT_LENGTH_M]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(world_quad, image_quad)
    pts = np.asarray(world_polygon, dtype=np.float32).reshape(1, -1, 2)
    projected = cv2.perspectiveTransform(pts, matrix).reshape(-1, 2)
    return [[round(float(x), 6), round(float(y), 6)] for x, y in projected]


def derive_service_zones(
    court_polygon: Sequence[Sequence[float]],
    service_depth_m: float = 3.3,
    center_gap_m: float = 0.0,
    side_inset_m: float = 0.0,
) -> Dict[str, List[List[float]]]:
    """Build four perspective-correct service-position polygons.

    Padel's service line is 3 m from each back wall. The default 3.3 m adds
    a small POC tolerance for pose/bbox noise. Set service_depth_m=3.0 for
    regulation geometry.
    """
    depth = float(np.clip(service_depth_m, 0.5, COURT_LENGTH_M / 2.0 - 0.1))
    side = float(np.clip(side_inset_m, 0.0, COURT_WIDTH_M / 2.0 - 0.2))
    gap = float(np.clip(center_gap_m, 0.0, COURT_WIDTH_M / 2.0 - side - 0.1))

    left_x0 = side
    left_x1 = COURT_WIDTH_M / 2.0 - gap / 2.0
    right_x0 = COURT_WIDTH_M / 2.0 + gap / 2.0
    right_x1 = COURT_WIDTH_M - side

    canonical = {
        "far_left": [[left_x0, 0.0], [left_x1, 0.0], [left_x1, depth], [left_x0, depth]],
        "far_right": [[right_x0, 0.0], [right_x1, 0.0], [right_x1, depth], [right_x0, depth]],
        "near_left": [
            [left_x0, COURT_LENGTH_M - depth],
            [left_x1, COURT_LENGTH_M - depth],
            [left_x1, COURT_LENGTH_M],
            [left_x0, COURT_LENGTH_M],
        ],
        "near_right": [
            [right_x0, COURT_LENGTH_M - depth],
            [right_x1, COURT_LENGTH_M - depth],
            [right_x1, COURT_LENGTH_M],
            [right_x0, COURT_LENGTH_M],
        ],
    }
    return {name: project_world_polygon(court_polygon, poly) for name, poly in canonical.items()}


def normalized_polygon_to_pixels(points: Iterable[Iterable[float]], frame_shape) -> np.ndarray:
    h, w = frame_shape[:2]
    arr = np.asarray(list(points), dtype=np.float32)
    px = np.empty_like(arr, dtype=np.int32)
    px[:, 0] = np.rint(arr[:, 0] * w).astype(np.int32)
    px[:, 1] = np.rint(arr[:, 1] * h).astype(np.int32)
    return px
