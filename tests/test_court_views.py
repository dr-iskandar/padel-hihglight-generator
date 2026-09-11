import numpy as np

from padel_poc.court_views import (
    derive_half_court_polygons,
    filter_detections_to_polygon,
    point_in_normalized_polygon,
    polygon_to_crop_rect,
)


def test_half_court_polygons_follow_perspective():
    court = [[0.35, 0.20], [0.65, 0.20], [0.90, 0.90], [0.10, 0.90]]
    halves = derive_half_court_polygons(court)

    assert set(halves) == {"near", "far"}
    assert len(halves["near"]) == 4
    assert len(halves["far"]) == 4

    far = np.asarray(halves["far"])
    near = np.asarray(halves["near"])
    assert far[:, 1].mean() < near[:, 1].mean()


def test_crop_rect_for_far_half_is_smaller_than_full_frame():
    court = [[0.35, 0.20], [0.65, 0.20], [0.90, 0.90], [0.10, 0.90]]
    far = derive_half_court_polygons(court)["far"]
    x1, y1, x2, y2 = polygon_to_crop_rect(far, (1080, 1920, 3), padding=0.05)

    assert 0 <= x1 < x2 <= 1920
    assert 0 <= y1 < y2 <= 1080
    assert (x2 - x1) < 1920
    assert (y2 - y1) < 1080


def test_filter_uses_feet_to_assign_player_to_half():
    near_poly = [[0.0, 0.5], [1.0, 0.5], [1.0, 1.0], [0.0, 1.0]]
    ids = [1, 2]
    boxes = np.asarray([[20, 20, 80, 80], [20, 120, 80, 190]], dtype=np.float32)
    xy = np.zeros((2, 17, 2), dtype=np.float32)
    conf = np.ones((2, 17), dtype=np.float32)

    out_ids, out_boxes, _xy, _conf = filter_detections_to_polygon(
        ids, boxes, xy, conf, near_poly, (200, 100, 3)
    )

    assert out_ids == [2]
    assert len(out_boxes) == 1
    assert point_in_normalized_polygon(0.5, 0.95, near_poly)
