from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence
import math

import cv2
import numpy as np


@dataclass
class BallObservation:
    point: Optional[tuple[float, float]]
    confidence: float
    found: bool


class BallTracker:
    """Lightweight POC tracker for a yellow/green padel ball.

    The tracker combines colour, motion, compactness and temporal continuity.
    v0.7 intentionally rejects nearly-static yellow objects (scoreboard/UI,
    court branding, etc.) and resets stale tracks quickly so it can reacquire
    the real ball elsewhere in the court.
    """

    def __init__(self, cfg: Optional[dict] = None, court_polygon: Optional[Sequence[Sequence[float]]] = None):
        cfg = cfg or {}
        self.hsv_lower = np.asarray(cfg.get("hsv_lower", [18, 70, 105]), dtype=np.uint8)
        self.hsv_upper = np.asarray(cfg.get("hsv_upper", [48, 255, 255]), dtype=np.uint8)
        self.motion_threshold = int(cfg.get("motion_threshold", 12))
        self.min_area = float(cfg.get("min_area", 2.0))
        self.max_area = float(cfg.get("max_area", 180.0))
        self.max_radius = float(cfg.get("max_radius", 12.0))
        self.max_jump_ratio = float(cfg.get("max_jump_ratio", 0.42))
        self.position_smoothing = float(cfg.get("position_smoothing", 0.72))
        self.velocity_smoothing = float(cfg.get("velocity_smoothing", 0.55))
        self.acquire_requires_motion = bool(cfg.get("acquire_requires_motion", True))
        self.require_motion_while_tracking = bool(cfg.get("require_motion_while_tracking", True))
        self.acquire_min_motion = float(cfg.get("acquire_min_motion", 0.045))
        self.track_min_motion = float(cfg.get("track_min_motion", 0.020))
        self.min_confidence = float(cfg.get("min_confidence", 0.24))
        self.reset_after_misses = max(1, int(cfg.get("reset_after_misses", 5)))
        self.court_polygon = court_polygon

        self.prev_gray: Optional[np.ndarray] = None
        self.last_pos: Optional[np.ndarray] = None
        self.velocity = np.zeros(2, dtype=np.float32)
        self.missed = 0

    def _court_mask(self, frame_shape) -> np.ndarray:
        h, w = frame_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        if self.court_polygon:
            pts = np.asarray(self.court_polygon, dtype=np.float32)
            if pts.shape == (4, 2):
                px = np.empty_like(pts, dtype=np.int32)
                px[:, 0] = np.rint(pts[:, 0] * w).astype(np.int32)
                px[:, 1] = np.rint(pts[:, 1] * h).astype(np.int32)
                cv2.fillConvexPoly(mask, px, 255)
                return mask
        mask[:] = 255
        return mask

    @staticmethod
    def _motion_fraction(mask: np.ndarray, cx: float, cy: float, radius: float) -> float:
        r = max(3, int(round(radius * 1.8)))
        x1, x2 = max(0, int(cx) - r), min(mask.shape[1], int(cx) + r + 1)
        y1, y2 = max(0, int(cy) - r), min(mask.shape[0], int(cy) + r + 1)
        patch = mask[y1:y2, x1:x2]
        if patch.size == 0:
            return 0.0
        return float(np.count_nonzero(patch)) / float(patch.size)

    def _mark_miss(self) -> BallObservation:
        self.missed += 1
        if self.missed >= self.reset_after_misses:
            self.last_pos = None
            self.velocity[:] = 0.0
        return BallObservation(None if self.last_pos is None else tuple(map(float, self.last_pos)), 0.0, False)

    def update(self, frame) -> BallObservation:
        h, w = frame.shape[:2]
        diag = math.hypot(w, h)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        colour = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        colour = cv2.bitwise_and(colour, self._court_mask(frame.shape))
        colour = cv2.morphologyEx(colour, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        if self.prev_gray is None:
            motion = np.zeros_like(gray)
        else:
            delta = cv2.absdiff(gray, self.prev_gray)
            _, motion = cv2.threshold(delta, self.motion_threshold, 255, cv2.THRESH_BINARY)
            motion = cv2.dilate(motion, np.ones((3, 3), np.uint8), iterations=1)
        self.prev_gray = gray

        predicted = None if self.last_pos is None else self.last_pos + self.velocity
        contours, _ = cv2.findContours(colour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_score = -1.0

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_area or area > self.max_area:
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            if radius < 0.8 or radius > self.max_radius:
                continue

            circle_area = math.pi * max(1.0, radius * radius)
            fill = min(1.0, area / circle_area)
            if fill < 0.10:
                continue

            moving = self._motion_fraction(motion, cx, cy, radius)
            min_motion = self.acquire_min_motion if self.last_pos is None else self.track_min_motion
            if self.last_pos is None and self.acquire_requires_motion and moving < min_motion:
                continue
            if self.last_pos is not None and self.require_motion_while_tracking and moving < min_motion:
                # Critical: prevents static yellow score graphics / branding from
                # becoming the portrait-camera target.
                continue

            proximity = 0.45
            if predicted is not None:
                dist = math.hypot(cx - float(predicted[0]), cy - float(predicted[1]))
                if dist > self.max_jump_ratio * diag:
                    continue
                proximity = max(0.0, 1.0 - dist / max(1.0, self.max_jump_ratio * diag))

            # Motion is deliberately the strongest cue. A truly moving tiny ball
            # should outrank a large/static yellow UI element.
            motion_score = min(1.0, moving * 5.0)
            score = 0.18 * fill + 0.54 * motion_score + 0.28 * proximity
            if score > best_score:
                best_score = score
                best = np.asarray([cx, cy], dtype=np.float32)

        if best is None or best_score < self.min_confidence:
            return self._mark_miss()

        if self.last_pos is None:
            new_pos = best
            new_velocity = np.zeros(2, dtype=np.float32)
        else:
            alpha = float(np.clip(self.position_smoothing, 0.0, 1.0))
            new_pos = (1.0 - alpha) * self.last_pos + alpha * best
            measured_v = new_pos - self.last_pos
            beta = float(np.clip(self.velocity_smoothing, 0.0, 1.0))
            new_velocity = (1.0 - beta) * self.velocity + beta * measured_v

        self.last_pos = new_pos.astype(np.float32)
        self.velocity = new_velocity.astype(np.float32)
        self.missed = 0
        return BallObservation(tuple(map(float, self.last_pos)), float(min(1.0, best_score)), True)
