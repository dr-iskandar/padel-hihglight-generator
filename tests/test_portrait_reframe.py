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


def test_prefers_serve_relevant_player():
    reframer = SmartPortraitReframer((1920, 1080), 30.0, {})
    boxes = np.array([[100, 300, 300, 900], [1200, 300, 1400, 900]], dtype=np.float32)
    reframer.update_tracks(
        [1, 2],
        boxes,
        {1: status("IDLE", 0.1), 2: status("PREPARING", 0.8)},
    )
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    reframer.render(frame, 1)
    assert reframer.last_focus.track_id == 2


def test_event_lock_keeps_target():
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
