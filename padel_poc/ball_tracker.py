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
    """Lightweight small-ball tracker for the portrait virtual camera.

    The earlier POC mostly looked for saturated yellow/green pixels. In the
    broadcast sample the ball is frequently rendered almost white, especially
    when it is far from the camera or blurred by motion. This tracker therefore
    combines two appearance masks (yellow/green + bright low-saturation), frame
    difference, compactness, predicted position, and short velocity prediction.

    It is still a POC tracker, not the final learned ball detector, but it is
    deliberately conservative about static court lines / score graphics and can
    reacquire quickly after a brief miss.
    """

    def __init__(self, cfg: Optional[dict] = None, court_polygon: Optional[Sequence[Sequence[float]]] = None):
        cfg = cfg or {}

        self.hsv_lower = np.asarray(cfg.get("hsv_lower", [12, 28, 120]), dtype=np.uint8)
        self.hsv_upper = np.asarray(cfg.get("hsv_upper", [62, 255, 255]), dtype=np.uint8)
        self.white_saturation_max = int(cfg.get("white_saturation_max", 82))
        self.white_value_min = int(cfg.get("white_value_min", 168))

        self.motion_threshold = int(cfg.get("motion_threshold", 8))
        self.min_area = float(cfg.get("min_area", 1.0))
        self.max_area = float(cfg.get("max_area", 180.0))
        self.max_radius = float(cfg.get("max_radius", 12.0))
        self.max_aspect_ratio = float(cfg.get("max_aspect_ratio", 2.8))
        self.max_jump_ratio = float(cfg.get("max_jump_ratio", 0.22))
        self.position_smoothing = float(cfg.get("position_smoothing", 0.72))
        self.velocity_smoothing = float(cfg.get("velocity_smoothing", 0.58))
        self.min_confidence = float(cfg.get("min_confidence", 0.24))
        self.acquire_motion_fraction = float(cfg.get("acquire_motion_fraction", 0.018))
        self.tracked_motion_fraction = float(cfg.get("tracked_motion_fraction", 0.004))
        self.predict_frames = max(0, int(cfg.get("predict_frames", 3)))
        self.reset_after_missed = max(1, int(cfg.get("reset_after_missed", 8)))
        self.velocity_damping = float(cfg.get("velocity_damping", 0.86))
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
        r = max(3, int(round(radius * 2.0)))
        x1, x2 = max(0, int(cx) - r), min(mask.shape[1], int(cx) + r + 1)
        y1, y2 = max(0, int(cy) - r), min(mask.shape[0], int(cy) + r + 1)
        patch = mask[y1:y2, x1:x2]
        if patch.size == 0:
            return 0.0
        return float(np.count_nonzero(patch)) / float(patch.size)

    @staticmethod
    def _bbox_aspect(contour) -> float:
        _, _, w, h = cv2.boundingRect(contour)
        short = max(1.0, float(min(w, h)))
        long = max(1.0, float(max(w, h)))
        return long / short

    def _appearance_mask(self, hsv: np.ndarray, court_mask: np.ndarray) -> np.ndarray:
        yellow = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        _, s, v = cv2.split(hsv)
        low_sat = cv2.inRange(s, 0, self.white_saturation_max)
        bright = cv2.inRange(v, self.white_value_min, 255)
        whiteish = cv2.bitwise_and(low_sat, bright)
        mask = cv2.bitwise_or(yellow, whiteish)
        return cv2.bitwise_and(mask, court_mask)

    def _motion_mask(self, gray: np.ndarray) -> np.ndarray:
        if self.prev_gray is None:
            motion = np.zeros_like(gray)
        else:
            delta = cv2.absdiff(gray, self.prev_gray)
            _, motion = cv2.threshold(delta, self.motion_threshold, 255, cv2.THRESH_BINARY)
            motion = cv2.dilate(motion, np.ones((3, 3), np.uint8), iterations=1)
        self.prev_gray = gray
        return motion

    def _predict(self) -> Optional[np.ndarray]:
        if self.last_pos is None:
            return None
        return self.last_pos + self.velocity

    def _candidate_score(self, contour, motion: np.ndarray, predicted: Optional[np.ndarray], diag: float):
        area = float(cv2.contourArea(contour))
        if area < self.min_area or area > self.max_area:
            return -1.0, None
        if self._bbox_aspect(contour) > self.max_aspect_ratio:
            return -1.0, None

        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        if radius < 0.6 or radius > self.max_radius:
            return -1.0, None

        circle_area = math.pi * max(0.8, radius * radius)
        fill = min(1.0, area / circle_area)
        if fill < 0.08:
            return -1.0, None

        moving = self._motion_fraction(motion, cx, cy, radius)
        required_motion = self.tracked_motion_fraction if predicted is not None else self.acquire_motion_fraction
        if moving < required_motion:
            return -1.0, None

        proximity = 0.45
        if predicted is not None:
            dist = math.hypot(cx - float(predicted[0]), cy - float(predicted[1]))
            max_jump = self.max_jump_ratio * diag * (1.0 + min(0.7, 0.18 * self.missed))
            if dist > max_jump:
                return -1.0, None
            proximity = max(0.0, 1.0 - dist / max(1.0, max_jump))

        compactness = min(1.0, fill * 1.35)
        motion_score = min(1.0, moving * 6.0)
        size_score = 1.0 - min(1.0, max(0.0, radius - 1.0) / max(1.0, self.max_radius - 1.0))

        if predicted is None:
            score = 0.43 * motion_score + 0.32 * compactness + 0.15 * size_score + 0.10 * proximity
        else:
            score = 0.46 * proximity + 0.25 * motion_score + 0.19 * compactness + 0.10 * size_score
        return score, np.asarray([cx, cy], dtype=np.float32)

    def update(self, frame) -> BallObservation:
        h, w = frame.shape[:2]
        diag = math.hypot(w, h)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        appearance = self._appearance_mask(hsv, self._court_mask(frame.shape))
        motion = self._motion_mask(gray)
        candidate_mask = cv2.bitwise_and(appearance, motion) if self.last_pos is None else appearance

        contours, _ = cv2.findContours(candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        predicted = self._predict()
        best = None
        best_score = -1.0

        for contour in contours:
            score, point = self._candidate_score(contour, motion, predicted, diag)
            if point is not None and score > best_score:
                best_score = score
                best = point

        if best is None or best_score < self.min_confidence:
            self.missed += 1

            # Continue a short trajectory through blur/occlusion. This is crucial
            # on fast left/right shots where the ball disappears for 1-3 frames.
            if self.last_pos is not None and self.missed <= self.predict_frames:
                self.last_pos = (self.last_pos + self.velocity).astype(np.float32)
                self.velocity = (self.velocity * self.velocity_damping).astype(np.float32)
                confidence = max(self.min_confidence, 0.42 - 0.08 * (self.missed - 1))
                return BallObservation(tuple(map(float, self.last_pos)), confidence, True)

            if self.missed >= self.reset_after_missed:
                self.last_pos = None
                self.velocity[:] = 0.0
            return BallObservation(None if self.last_pos is None else tuple(map(float, self.last_pos)), 0.0, False)

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
