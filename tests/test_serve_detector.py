import numpy as np
from padel_poc.serve_detector import ServeDetector


def make_kp(offset_y=0.0, left_wrist=(90, 170), right_wrist=(110, 175)):
    kp = np.zeros((17, 3), dtype=np.float32)
    kp[:, 2] = 0.0
    kp[5] = [90.0, 100.0 + offset_y, 0.95]
    kp[6] = [110.0, 100.0 + offset_y, 0.95]
    kp[9] = [left_wrist[0], left_wrist[1] + offset_y, 0.95]
    kp[10] = [right_wrist[0], right_wrist[1] + offset_y, 0.95]
    kp[11] = [92.0, 150.0 + offset_y, 0.95]
    kp[12] = [108.0, 150.0 + offset_y, 0.95]
    return kp


def cfg():
    return {
        "min_keypoint_conf": 0.3,
        "stationary_seconds": 0.20,
        "stationary_speed_threshold": 0.8,
        "prep_memory_seconds": 0.8,
        "swing_speed_threshold": 1.0,
        "wrist_lookback_seconds": 0.10,
        "body_lookback_seconds": 0.10,
        "confirm_window_seconds": 0.6,
        "forward_move_threshold": 0.05,
        "require_forward_motion": True,
        "cooldown_seconds": 2.0,
        "global_cooldown_seconds": 1.0,
    }


def test_trigger_requires_sequence_and_forward_motion():
    detector = ServeDetector({"near_left": [0, 0, 1, 1]}, cfg(), fps=30)
    shape = (300, 200, 3)
    bbox = [50, 50, 150, 250]
    for f in range(0, 10):
        assert detector.update(1, bbox, make_kp(), shape, f) is None
    event = detector.update(1, bbox, make_kp(left_wrist=(90, 75)), shape, 10)
    assert event is None
    assert detector.get_status(1).state == "SWING"
    shifted_bbox = [50, 35, 150, 235]
    event = detector.update(1, shifted_bbox, make_kp(offset_y=-15, left_wrist=(90, 80)), shape, 13)
    assert event is not None
    assert event.track_id == 1
    assert event.zone == "near_left"


def test_fast_wrist_without_stationary_prep_does_not_trigger():
    detector = ServeDetector({"near_left": [0, 0, 1, 1]}, cfg(), fps=30)
    shape = (300, 200, 3)
    for f in range(12):
        y = 50 - f * 3
        bbox = [50, y, 150, y + 200]
        kp = make_kp(offset_y=y - 50, left_wrist=(90, 80 if f == 8 else 170))
        event = detector.update(1, bbox, kp, shape, f)
        assert event is None


def test_no_trigger_outside_zone():
    detector = ServeDetector({"zone": [0, 0, 0.2, 0.2]}, cfg(), fps=30)
    bbox = [100, 100, 180, 260]
    shape = (300, 200, 3)
    for f in range(10):
        assert detector.update(1, bbox, make_kp(), shape, f) is None


def test_global_cooldown_suppresses_id_switch_duplicate():
    detector = ServeDetector({"near_left": [0, 0, 1, 1]}, cfg(), fps=30)
    shape = (300, 200, 3)
    bbox = [50, 50, 150, 250]

    def perform_serve(track_id, start):
        event = None
        for f in range(start, start + 10):
            event = detector.update(track_id, bbox, make_kp(), shape, f)
        detector.update(track_id, bbox, make_kp(left_wrist=(90, 75)), shape, start + 10)
        shifted_bbox = [50, 35, 150, 235]
        event = detector.update(track_id, shifted_bbox, make_kp(offset_y=-15, left_wrist=(90, 80)), shape, start + 13)
        return event

    assert perform_serve(1, 0) is not None
    assert perform_serve(9, 15) is None
