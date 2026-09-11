from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import cv2
import numpy as np


@dataclass
class FocusInfo:
    track_id: Optional[int]
    crop_rect: tuple[int, int, int, int]


class SmartPortraitReframer:
    """Smooth 9:16 virtual camera that follows the most relevant player."""

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

        self.subject_height_ratio = float(cfg.get("subject_height_ratio", 0.38))
        self.min_crop_height_ratio = float(cfg.get("min_crop_height_ratio", 0.40))
        self.max_crop_height_ratio = float(cfg.get("max_crop_height_ratio", 1.00))
        self.center_smoothing = float(cfg.get("center_smoothing", 0.20))
        self.zoom_smoothing = float(cfg.get("zoom_smoothing", 0.14))
        self.deadzone_ratio = float(cfg.get("deadzone_ratio", 0.06))
        self.sticky_bonus = float(cfg.get("sticky_bonus", 1.25))
        self.vertical_subject_bias = float(cfg.get("vertical_subject_bias", -0.10))
        self.lock_seconds = float(cfg.get("lock_seconds", 5.0))

        self.court_polygon = court_polygon
        self.tracks: Dict[int, tuple[np.ndarray, object]] = {}
        self.current_target: Optional[int] = None
        self.locked_target: Optional[int] = None
        self.lock_until_frame = -1

        fallback_x, fallback_y = self._fallback_center()
        self.center_x = float(fallback_x)
        self.center_y = float(fallback_y)
        self.crop_h = float(self._max_valid_crop_height())
        self.last_focus = FocusInfo(None, self._crop_rect())

    @property
    def output_size(self) -> tuple[int, int]:
        return self.output_w, self.output_h

    def _fallback_center(self) -> tuple[float, float]:
        if self.court_polygon:
            pts = np.asarray(self.court_polygon, dtype=np.float32)
            if pts.shape == (4, 2):
                return (
                    float(np.mean(pts[:, 0]) * self.frame_w),
                    float(np.mean(pts[:, 1]) * self.frame_h),
                )
        return self.frame_w * 0.5, self.frame_h * 0.5

    def _max_valid_crop_height(self) -> float:
        by_width = self.frame_w / max(1e-6, self.aspect)
        return float(min(self.frame_h, by_width))

    def update_tracks(self, track_ids, boxes, statuses: dict):
        self.tracks = {}
        for i, track_id in enumerate(track_ids):
            if i >= len(boxes):
                break
            status = statuses.get(int(track_id))
            self.tracks[int(track_id)] = (np.asarray(boxes[i], dtype=np.float32), status)

    def lock_target(self, track_id: int, frame_index: int, seconds: Optional[float] = None):
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

    def _desired_camera(self, target_id: Optional[int]) -> tuple[float, float, float]:
        if target_id is None or target_id not in self.tracks:
            x, y = self._fallback_center()
            return x, y, self._max_valid_crop_height()

        bbox, _ = self.tracks[target_id]
        x1, y1, x2, y2 = map(float, bbox)
        bh = max(20.0, y2 - y1)
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2) + self.vertical_subject_bias * bh

        desired_h = bh / max(0.12, self.subject_height_ratio)
        min_h = self.min_crop_height_ratio * self.frame_h
        max_h = min(self.max_crop_height_ratio * self.frame_h, self._max_valid_crop_height())
        desired_h = float(np.clip(desired_h, min_h, max_h))
        return cx, cy, desired_h

    def _smooth_camera(self, desired_x: float, desired_y: float, desired_h: float):
        current_w = self.crop_h * self.aspect
        dead_x = max(6.0, current_w * self.deadzone_ratio)
        dead_y = max(6.0, self.crop_h * self.deadzone_ratio)

        dx = desired_x - self.center_x
        dy = desired_y - self.center_y

        if abs(dx) > dead_x:
            self.center_x += self.center_smoothing * dx
        if abs(dy) > dead_y:
            self.center_y += self.center_smoothing * dy

        self.crop_h += self.zoom_smoothing * (desired_h - self.crop_h)
        self.crop_h = float(
            np.clip(
                self.crop_h,
                self.min_crop_height_ratio * self.frame_h,
                self._max_valid_crop_height(),
            )
        )

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
        target_id = self._select_target(frame_index)
        desired_x, desired_y, desired_h = self._desired_camera(target_id)
        self._smooth_camera(desired_x, desired_y, desired_h)

        x1, y1, x2, y2 = self._crop_rect()
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            portrait = np.zeros((self.output_h, self.output_w, 3), dtype=np.uint8)
        else:
            portrait = cv2.resize(
                crop,
                (self.output_w, self.output_h),
                interpolation=cv2.INTER_LINEAR,
            )

        self.last_focus = FocusInfo(target_id, (x1, y1, x2, y2))
        return portrait

    def draw_source_crop(self, frame, color=(255, 120, 255)):
        x1, y1, x2, y2 = self.last_focus.crop_rect
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        label = "PORTRAIT"
        if self.last_focus.track_id is not None:
            label += f" -> T{self.last_focus.track_id}"
        cv2.putText(
            frame,
            label,
            (x1 + 5, max(20, y1 + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
