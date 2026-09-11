from padel_poc.court_geometry import derive_service_zones
from padel_poc.serve_detector import ServeDetector


def test_service_zones_follow_perspective_quad():
    court = [[0.30, 0.20], [0.70, 0.20], [0.90, 0.90], [0.10, 0.90]]
    zones = derive_service_zones(court, service_depth_m=3.0)

    assert set(zones) == {"far_left", "far_right", "near_left", "near_right"}
    assert all(len(poly) == 4 for poly in zones.values())

    far_width = zones["far_right"][1][0] - zones["far_right"][0][0]
    near_width = zones["near_right"][2][0] - zones["near_right"][3][0]
    assert near_width > far_width


def test_polygon_zone_rejects_point_in_bounding_box_but_outside_shape():
    triangular = {"near_left": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]}
    normalized = {k: ServeDetector._normalize_polygon(v) for k, v in triangular.items()}

    assert ServeDetector._inside_zone(0.2, 0.2, normalized) == "near_left"
    assert ServeDetector._inside_zone(0.9, 0.9, normalized) is None


def test_court_polygon_can_reject_outside_person():
    court = ServeDetector._normalize_polygon([[0.2, 0.2], [0.8, 0.2], [0.9, 0.9], [0.1, 0.9]])
    assert ServeDetector._point_in_polygon(0.5, 0.5, court)
    assert not ServeDetector._point_in_polygon(0.02, 0.6, court)
