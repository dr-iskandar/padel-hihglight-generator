import numpy as np

from padel_poc.trajectory_predictor import BallTrajectoryPredictor


def make_predictor():
    # Simple trapezoid/perspective court in a 1000x1000 frame.
    court = [
        [0.30, 0.20],
        [0.70, 0.20],
        [0.90, 0.90],
        [0.10, 0.90],
    ]
    return BallTrajectoryPredictor(
        frame_size=(1000, 1000),
        fps=30.0,
        court_polygon=court,
        cfg={
            "horizon_seconds": 0.30,
            "min_points": 3,
            "min_span_seconds": 0.03,
            "min_speed_mps": 0.2,
            "min_confidence": 0.10,
        },
    )


def test_predicts_leftward_destination():
    predictor = make_predictor()

    # Points move steadily toward the left side of the image/court.
    prediction = None
    for frame, x in zip([0, 2, 4, 6, 8], [650, 610, 570, 530, 490]):
        prediction = predictor.update((x, 550), 0.95, frame, found=True)

    assert prediction is not None
    assert prediction.valid
    assert prediction.point is not None
    assert prediction.point[0] < 490
    assert prediction.zone in {"LEFT", "CENTER"}
    assert prediction.confidence > 0.10


def test_direction_change_discards_old_trajectory():
    predictor = make_predictor()

    # First the ball travels right.
    for frame, x in zip([0, 2, 4, 6], [420, 455, 490, 525]):
        predictor.update((x, 550), 0.95, frame, found=True)

    # Then a hit/rebound sends it hard left. The predictor should reset the old
    # rightward history and eventually predict left of the latest observation.
    prediction = None
    for frame, x in zip([8, 10, 12, 14], [500, 465, 430, 395]):
        prediction = predictor.update((x, 550), 0.95, frame, found=True)

    assert prediction is not None
    assert prediction.valid
    assert prediction.point is not None
    assert prediction.point[0] < 395


def test_missing_ball_prediction_expires():
    predictor = make_predictor()
    for frame, x in zip([0, 2, 4, 6], [550, 530, 510, 490]):
        prediction = predictor.update((x, 550), 0.95, frame, found=True)

    assert prediction.valid
    # 30 fps with 0.20s hold -> by frame 20 it should be stale.
    stale = predictor.update(None, 0.0, 20, found=False)
    assert not stale.valid
