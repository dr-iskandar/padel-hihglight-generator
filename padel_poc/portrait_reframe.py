from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import cv2
import numpy as np


@dataclass
class FocusInfo:
    track_id: Optional[int]
    crop_rect: tuple[int, int, int, int]
    mode: str = "COURT"
    ball_point: Optional[tuple[float, float]] = None


class SmartPortraitReframer:
    """Composition-aware 9:16 virtual camera.

    The ball guides horizontal panning, but the camera deliberately does not
    chase every ball movement. The crop keeps a mostly fixed zoom and vertical
    composition, uses a central safe zone, limits pan speed, and blends the
    ball cue with player/court context. Player tracking is a fallback when the
    ball is temporarily lost.
    """

    STATE_SCORES = {
        "SWING": 7.0,
        "COOLDOWN": 6.0,
        "PREPARING": 5.0,
        "IN_ZONE": 2.5,
        "IDLE": 0.5,
    }

    def __init__(
        self,
        frame_size,
        fps: float,
        cfg: Optional[dict] = None,
        court_polygon: Optional[Sequence[Sequence[float]]] = None,
    ):
        cfg = cfg or {}
        self.frame_w, self.frame_h = map(int, frame_size)
        self.fps = float(fps)

        self.output_w = max(180, int(cfg.get("width", 1080)))
        self.output_h = max(320, int(cfg.get("height", 1920)))
        self.aspect = self.output_w / self.output_h

        # Reference-like framing: mostly fixed zoom, nearly fixed vertical axis.
        self.crop_height_ratio = float(cfg.get("crop_height_ratio", 0.92))
        self.vertical_bias = float(cfg.get("vertical_bias", 0.02))
        self.pan_smoothing = float(cfg.get("pan_smoothing", 0.18))
        self.max_pan_speed_ratio = float(cfg.get("max_pan_speed_ratio", 0.85))
        self.ball_safezone_ratio = float(cfg.get("ball_safezone_ratio", 0.20))
        self.ball_weight = float(cfg.get("ball_weight", 0.78))
        self.ball_min_confidence = float(cfg.get("ball_min_confidence", 0.25))
        self.ball_hold_seconds = float(cfg.get("ball_hold_seconds", 0.55))
        self.recenter_smoothing = float(cfg.get("recenter_smoothing", 0.05))
        self.sticky_bonus = float(cfg.get("sticky_bonus", 1.25))
        self.lock_seconds = float(cfg.get("lock_seconds", 2.0))

        self.court_polygon = court_polygon
        self.tracks: Dict[int, tuple[np.ndarray, object]] = {}
        self.current_target: Optional[int] = None
        self.locked_target: Optional[int] = None
        self.lock_until_frame = -1

        self.ball_point: Optional[np.ndarray] = None
        self.ball_confidence = 0.0
        self.ball_last_seen_frame = -10**9

        fallback_x, fallback_y = self._fallback_center()
        self.center_x = float(fallback_x)
        self.center_y = float(fallback_y)
        self.crop_h = float(self._fixed_crop_height())
        self.last_focus = FocusInfo(None, self._crop_rect(), "COURT", None)

    @property
    def output_size(self) -> tuple[int, int]:
        return self.output_w, self.output_h

    def _fallback_center(self) -> tuple[float, float]:
        if self.court_polygon:
            pts = np.asarray(self.court_polygon, dtype=np.float32)
            if pts.shape == (4, 2):
                x = float(np.mean(pts[:, 0]) * self.frame_w)
                y_min = float(np.min(pts[:, 1]) * self.frame_h)
                y_max = float(np.max(pts[:, 1]) * self.frame_h)
                y = 0.5 * (y_min + y_max) + self.vertical_bias * (y_max - y_min)
                return x, y
        return self.frame_w * 0.5, self.frame_h * (0.5 + self.vertical_bias)

    def _max_valid_crop_height(self) -> float:
        by_width = self.frame_w / max(1e-6, self.aspect)
        return float(min(self.frame_h, by_width))

    def _fixed_crop_height(self) -> float:
        ratio = float(np.clip(self.crop_height_ratio, 0.72, 1.0))
        return min(self._max_valid_crop_height(), ratio * self.frame_h)

    def update_tracks(self, track_ids, boxes, statuses: dict):
        self.tracks = {}
        for i, track_id in enumerate(track_ids):
            if i >= len(boxes):
                break
            status = statuses.get(int(track_id))
            self.tracks[int(track_id)] = (np.asarray(boxes[i], dtype=np.float32), status)

    def update_ball(self, point: Optional[tuple[float, float]], confidence: float, frame_index: int, found: bool = True):
        if point is None:
            return
        if found and confidence >= self.ball_min_confidence:
            self.ball_point = np.asarray(point, dtype=np.float32)
            self.ball_confidence = float(confidence)
            self.ball_last_seen_frame = int(frame_index)

    def lock_target(self, track_id: int, frame_index: int, seconds: Optional[float] = None):
        # Player lock is only a fallback if ball tracking is unavailable.
        self.locked_target = int(track_id)
        duration = self.lock_seconds if seconds is None else float(seconds)
        self.lock_until_frame = int(frame_index + max(1.0, duration * self.fps))
        self.current_target = int(track_id)

    def _score(self, track_id: int, status) -> float:
        if status is None:
            score = 0.0
        else:
            state = str(getattr(status, "state", "IDLE")).upper()
            confidence = float(getattr(status, "confidence", 0.0) or 0.0)
            score = self.STATE_SCORES.get(state, 0.5) + 2.0 * confidence
        if track_id == self.current_target:
            score += self.sticky_bonus
        return score

    def _select_target(self, frame_index: int) -> Optional[int]:
        if self.locked_target is not None:
            if frame_index <= self.lock_until_frame and self.locked_target in self.tracks:
                return self.locked_target
            if frame_index > self.lock_until_frame:
                self.locked_target = None

        if not self.tracks:
            self.current_target = None
            return None

        best_id = max(self.tracks.keys(), key=lambda tid: self._score(tid, self.tracks[tid][1]))
        self.current_target = int(best_id)
        return self.current_target

    def _player_anchor_x(self, frame_index: int) -> tuple[float, Optional[int]]:
        target_id = self._select_target(frame_index)
        if not self.tracks:
            return self._fallback_center()[0], target_id

        centers = []
        weights = []
        for track_id, (bbox, status) in self.tracks.items():
            x1, _, x2, _ = map(float, bbox)
            centers.append(0.5 * (x1 + x2))
            weights.append(max(0.5, self._score(track_id, status)))

        centers_arr = np.asarray(centers, dtype=np.float32)
        weights_arr = np.asarray(weights, dtype=np.float32)
        anchor = float(np.sum(centers_arr * weights_arr) / max(1e-6, np.sum(weights_arr)))
        return anchor, target_id

    def _ball_is_fresh(self, frame_index: int) -> bool:
        hold_frames = max(1, int(round(self.ball_hold_seconds * self.fps)))
        return self.ball_point is not None and frame_index - self.ball_last_seen_frame <= hold_frames

    def _desired_center_x(self, frame_index: int) -> tuple[float, str, Optional[int]]:
        fallback_x, _ = self._fallback_center()
        player_x, target_id = self._player_anchor_x(frame_index)
        crop_w = self.crop_h * self.aspect

        if self._ball_is_fresh(frame_index):
            ball_x = float(self.ball_point[0])
            safe = max(20.0, self.ball_safezone_ratio * crop_w)
            delta = ball_x - self.center_x

            # Do not move while the ball stays in the central composition zone.
            if abs(delta) <= safe:
                ball_guided_x = self.center_x
            else:
                ball_guided_x = ball_x - np.sign(delta) * safe

            weight = float(np.clip(self.ball_weight, 0.0, 1.0))
            desired = weight * ball_guided_x + (1.0 - weight) * player_x
            return float(desired), "BALL", target_id

        if self.tracks:
            return float(player_x), "PLAYERS", target_id
        return float(fallback_x), "COURT", target_id

    def _smooth_pan(self, desired_x: float, mode: str):
        crop_w = self.crop_h * self.aspect
        alpha = self.pan_smoothing if mode == "BALL" else self.recenter_smoothing
        raw_step = alpha * (desired_x - self.center_x)
        max_step = max(2.0, self.max_pan_speed_ratio * crop_w / max(1.0, self.fps))
        self.center_x += float(np.clip(raw_step, -max_step, max_step))

    def _crop_rect(self) -> tuple[int, int, int, int]:
        crop_h = min(self.crop_h, self._max_valid_crop_height())
        crop_w = crop_h * self.aspect
        half_w = crop_w * 0.5
        half_h = crop_h * 0.5

        cx = float(np.clip(self.center_x, half_w, self.frame_w - half_w))
        cy = float(np.clip(self.center_y, half_h, self.frame_h - half_h))

        x1 = int(round(cx - half_w))
        x2 = int(round(cx + half_w))
        y1 = int(round(cy - half_h))
        y2 = int(round(cy + half_h))
        x1 = max(0, min(self.frame_w - 2, x1))
        y1 = max(0, min(self.frame_h - 2, y1))
        x2 = max(x1 + 2, min(self.frame_w, x2))
        y2 = max(y1 + 2, min(self.frame_h, y2))
        return x1, y1, x2, y2

    def render(self, frame, frame_index: int):
        self.crop_h = self._fixed_crop_height()  # intentionally stable; no wild zoom.
        _, fixed_y = self._fallback_center()
        self.center_y = fixed_y

        desired_x, mode, target_id = self._desired_center_x(frame_index)
        self._smooth_pan(desired_x, mode)

        x1, y1, x2, y2 = self._crop_rect()
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            portrait = np.zeros((self.output_h, self.output_w, 3), dtype=np.uint8)
        else:
            portrait = cv2.resize(crop, (self.output_w, self.output_h), interpolation=cv2.INTER_LINEAR)

        ball = None if self.ball_point is None else tuple(map(float, self.ball_point))
        self.last_focus = FocusInfo(target_id, (x1, y1, x2, y2), mode, ball)
        return portrait

    def draw_source_crop(self, frame, color=(255, 120, 255)):
        x1, y1, x2, y2 = self.last_focus.crop_rect
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        label = f"PORTRAIT {self.last_focus.mode}"
        cv2.putText(frame, label, (x1 + 5, max(20, y1 + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        if self.last_focus.ball_point is not None and self.last_focus.mode == "BALL":
            bx, by = map(int, self.last_focus.ball_point)
            cv2.circle(frame, (bx, by), 8, (0, 255, 255), 2, cv2.LINE_AA)
