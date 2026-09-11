from types import SimpleNamespace

import numpy as np

from padel_poc.portrait_reframe import SmartPortraitReframer


def status(state="IDLE", confidence=0.0):
    return SimpleNamespace(state=state, confidence=confidence)


def test_portrait_output_shape_and_bounds():
    reframer = SmartPortraitReframer((1920, 1080), 30.0, {"width": 540, "height": 960})
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    reframer.update_tracks(
        [1],
        np.array([[100, 300, 300, 900]], dtype=np.float32),
        {1: status("PREPARING", 0.7)},
    )
    out = reframer.render(frame, 1)
    assert out.shape == (960, 540, 3)
    x1, y1, x2, y2 = reframer.last_focus.crop_rect
    assert 0 <= x1 < x2 <= 1920
    assert 0 <= y1 < y2 <= 1080


def test_zoom_is_stable_when_player_size_changes():
    reframer = SmartPortraitReframer((1920, 1080), 30.0, {"crop_height_ratio": 0.92})
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    reframer.update_tracks([1], np.array([[100, 200, 250, 900]], dtype=np.float32), {1: status("SWING", 1.0)})
    reframer.render(frame, 1)
    first_h = reframer.last_focus.crop_rect[3] - reframer.last_focus.crop_rect[1]

    reframer.update_tracks([1], np.array([[300, 400, 360, 600]], dtype=np.float32), {1: status("SWING", 1.0)})
    reframer.render(frame, 2)
    second_h = reframer.last_focus.crop_rect[3] - reframer.last_focus.crop_rect[1]

    assert abs(first_h - second_h) <= 2


def test_ball_inside_safezone_does_not_cause_large_pan():
    reframer = SmartPortraitReframer((1920, 1080), 60.0, {
        "crop_height_ratio": 0.92,
        "ball_safezone_ratio": 0.22,
        "pan_smoothing": 0.2,
    })
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    start_x = reframer.center_x
    reframer.update_ball((start_x + 20, 500), 0.9, 1, found=True)
    reframer.render(frame, 1)
    assert abs(reframer.center_x - start_x) < 5.0
    assert reframer.last_focus.mode == "BALL"


def test_ball_far_right_pan_is_speed_limited():
    reframer = SmartPortraitReframer((1920, 1080), 60.0, {
        "crop_height_ratio": 0.92,
        "max_pan_speed_ratio": 0.5,
        "pan_smoothing": 1.0,
    })
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    start_x = reframer.center_x
    crop_w = reframer.crop_h * reframer.aspect
    reframer.update_ball((1800, 500), 0.95, 1, found=True)
    reframer.render(frame, 1)
    max_step = 0.5 * crop_w / 60.0
    assert reframer.center_x - start_x <= max_step + 1e-3


def test_player_lock_is_fallback_when_ball_missing():
    reframer = SmartPortraitReframer((1920, 1080), 30.0, {"lock_seconds": 2.0})
    boxes = np.array([[100, 300, 300, 900], [1200, 300, 1400, 900]], dtype=np.float32)
    reframer.update_tracks(
        [1, 2],
        boxes,
        {1: status("PREPARING", 0.9), 2: status("SWING", 1.0)},
    )
    reframer.lock_target(1, frame_index=10, seconds=2.0)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    reframer.render(frame, 20)
    assert reframer.last_focus.track_id == 1
    assert reframer.last_focus.mode == "PLAYERS"
