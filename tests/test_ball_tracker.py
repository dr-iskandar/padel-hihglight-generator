import cv2
import numpy as np

from padel_poc.ball_tracker import BallTracker


def make_frame(x=None, y=None):
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    if x is not None and y is not None:
        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 255), -1)
    return frame


def test_detects_moving_yellow_ball_inside_court():
    tracker = BallTracker(
        {"acquire_requires_motion": True, "min_confidence": 0.2},
        court_polygon=[[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
    )
    tracker.update(make_frame(500, 350))
    obs = tracker.update(make_frame(530, 350))
    assert obs.found
    assert obs.point is not None
    assert abs(obs.point[0] - 530) < 15


def test_rejects_ball_coloured_blob_outside_court():
    tracker = BallTracker(
        {"acquire_requires_motion": False, "min_confidence": 0.2},
        court_polygon=[[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
    )
    obs = tracker.update(make_frame(40, 40))
    assert not obs.found
