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
    """Conservative small-ball tracker for the portrait virtual camera.

    v0.12 focuses on stability. The main failure in the current broadcast sample
    is a moving hand/wrist being selected as the ball. This version therefore
    adds explicit skin-colour suppression in addition to the optional pose-aware
    wrist rejection, tighter size limits, trajectory continuity, and two-frame
    confirmation before acquiring a brand-new track.

    A learned ball detector is still the right production solution. This module
    is deliberately a stronger POC fallback, not an attempt to replace one.
    """

    WRIST_INDICES = (9, 10)  # COCO left/right wrist

    def __init__(self, cfg: Optional[dict] = None, court_polygon: Optional[Sequence[Sequence[float]]] = None):
        cfg = cfg or {}

        self.hsv_lower = np.asarray(cfg.get("hsv_lower", [12, 28, 120]), dtype=np.uint8)
        self.hsv_upper = np.asarray(cfg.get("hsv_upper", [62, 255, 255]), dtype=np.uint8)
        self.white_saturation_max = int(cfg.get("white_saturation_max", 82))
        self.white_value_min = int(cfg.get("white_value_min", 168))

        self.motion_threshold = int(cfg.get("motion_threshold", 8))
        self.min_area = float(cfg.get("min_area", 1.0))
        self.max_area = float(cfg.get("max_area", 90.0))
        self.max_radius = float(cfg.get("max_radius", 7.5))
        self.max_aspect_ratio = float(cfg.get("max_aspect_ratio", 2.35))
        self.max_jump_ratio = float(cfg.get("max_jump_ratio", 0.18))
        self.position_smoothing = float(cfg.get("position_smoothing", 0.80))
        self.velocity_smoothing = float(cfg.get("velocity_smoothing", 0.64))
        self.min_confidence = float(cfg.get("min_confidence", 0.29))
        self.acquire_motion_fraction = float(cfg.get("acquire_motion_fraction", 0.022))
        self.tracked_motion_fraction = float(cfg.get("tracked_motion_fraction", 0.004))
        self.predict_frames = max(0, int(cfg.get("predict_frames", 3)))
        self.reset_after_missed = max(1, int(cfg.get("reset_after_missed", 6)))
        self.velocity_damping = float(cfg.get("velocity_damping", 0.86))

        # Skin suppression. YCrCb ranges are intentionally broad; we use the
        # fraction only as a penalty, not as a binary classifier.
        self.skin_cr_min = int(cfg.get("skin_cr_min", 132))
        self.skin_cr_max = int(cfg.get("skin_cr_max", 178))
        self.skin_cb_min = int(cfg.get("skin_cb_min", 72))
        self.skin_cb_max = int(cfg.get("skin_cb_max", 132))
        self.skin_hard_fraction = float(cfg.get("skin_hard_fraction", 0.55))
        self.skin_penalty_strength = float(cfg.get("skin_penalty_strength", 0.88))

        # Optional pose-aware rejection. The current main loop can pass these
        # arrays later without changing this API again.
        self.hand_keypoint_conf = float(cfg.get("hand_keypoint_conf", 0.28))
        self.hand_radius_ratio = float(cfg.get("hand_radius_ratio", 0.15))
        self.hand_radius_min_px = float(cfg.get("hand_radius_min_px", 14.0))
        self.hand_reject_multiplier = float(cfg.get("hand_reject_multiplier", 0.18))
        self.person_inside_multiplier = float(cfg.get("person_inside_multiplier", 0.55))
        self.impact_allow_distance_px = float(cfg.get("impact_allow_distance_px", 28.0))

        self.acquire_confirm_frames = max(1, int(cfg.get("acquire_confirm_frames", 2)))
        self.acquire_confirm_jump_ratio = float(cfg.get("acquire_confirm_jump_ratio", 0.060))

        self.court_polygon = court_polygon
        self.prev_gray: Optional[np.ndarray] = None
        self.last_pos: Optional[np.ndarray] = None
        self.velocity = np.zeros(2, dtype=np.float32)
        self.missed = 0

        self.pending_pos: Optional[np.ndarray] = None
        self.pending_count = 0

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
    def _patch_fraction(mask: np.ndarray, cx: float, cy: float, radius: float, scale: float = 1.8) -> float:
        r = max(3, int(round(radius * scale)))
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

    def _skin_mask(self, frame: np.ndarray, court_mask: np.ndarray) -> np.ndarray:
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        _, cr, cb = cv2.split(ycrcb)
        cr_ok = cv2.inRange(cr, self.skin_cr_min, self.skin_cr_max)
        cb_ok = cv2.inRange(cb, self.skin_cb_min, self.skin_cb_max)
        skin = cv2.bitwise_and(cr_ok, cb_ok)
        skin = cv2.medianBlur(skin, 3)
        return cv2.bitwise_and(skin, court_mask)

    def _motion_mask(self, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.prev_gray is None:
            raw = np.zeros_like(gray)
        else:
            delta = cv2.absdiff(gray, self.prev_gray)
            _, raw = cv2.threshold(delta, self.motion_threshold, 255, cv2.THRESH_BINARY)
        self.prev_gray = gray
        dilated = cv2.dilate(raw, np.ones((3, 3), np.uint8), iterations=1)
        return raw, dilated

    def _predict(self) -> Optional[np.ndarray]:
        if self.last_pos is None:
            return None
        return self.last_pos + self.velocity

    def _pose_context(self, player_boxes, keypoints_xy, keypoints_conf):
        boxes = []
        hands = []
        if player_boxes is None:
            return boxes, hands

        for i, bbox in enumerate(player_boxes):
            if len(bbox) < 4:
                continue
            x1, y1, x2, y2 = map(float, bbox[:4])
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append((x1, y1, x2, y2))
            person_h = max(20.0, y2 - y1)
            radius = max(self.hand_radius_min_px, self.hand_radius_ratio * person_h)

            if keypoints_xy is None or keypoints_conf is None or i >= len(keypoints_xy) or i >= len(keypoints_conf):
                continue
            for k in self.WRIST_INDICES:
                if k >= len(keypoints_xy[i]) or k >= len(keypoints_conf[i]):
                    continue
                if float(keypoints_conf[i][k]) < self.hand_keypoint_conf:
                    continue
                x, y = map(float, keypoints_xy[i][k])
                if x > 0 and y > 0:
                    hands.append((x, y, radius))
        return boxes, hands

    @staticmethod
    def _inside_box(point: np.ndarray, box) -> bool:
        x, y = float(point[0]), float(point[1])
        x1, y1, x2, y2 = box
        return x1 <= x <= x2 and y1 <= y <= y2

    def _context_multiplier(self, point: np.ndarray, predicted: Optional[np.ndarray], boxes, hands) -> float:
        trajectory_agrees = False
        if predicted is not None:
            trajectory_agrees = float(np.linalg.norm(point - predicted)) <= self.impact_allow_distance_px

        multiplier = 1.0
        if not trajectory_agrees:
            for hx, hy, radius in hands:
                if math.hypot(float(point[0]) - hx, float(point[1]) - hy) <= radius:
                    multiplier *= self.hand_reject_multiplier
                    break
            if any(self._inside_box(point, box) for box in boxes):
                multiplier *= self.person_inside_multiplier
        return float(multiplier)

    def _candidate_score(
        self,
        contour,
        motion: np.ndarray,
        appearance: np.ndarray,
        skin: np.ndarray,
        predicted: Optional[np.ndarray],
        diag: float,
        boxes,
        hands,
    ):
        area = float(cv2.contourArea(contour))
        if area < self.min_area or area > self.max_area:
            return -1.0, None
        if self._bbox_aspect(contour) > self.max_aspect_ratio:
            return -1.0, None

        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        if radius < 0.55 or radius > self.max_radius:
            return -1.0, None

        circle_area = math.pi * max(0.8, radius * radius)
        fill = min(1.0, area / circle_area)
        if fill < 0.08:
            return -1.0, None

        moving = self._patch_fraction(motion, cx, cy, radius, scale=2.0)
        required_motion = self.tracked_motion_fraction if predicted is not None else self.acquire_motion_fraction
        if moving < required_motion:
            return -1.0, None

        point = np.asarray([cx, cy], dtype=np.float32)
        proximity = 0.45
        dist_to_prediction = None
        if predicted is not None:
            dist_to_prediction = float(np.linalg.norm(point - predicted))
            max_jump = self.max_jump_ratio * diag * (1.0 + min(0.50, 0.12 * self.missed))
            if dist_to_prediction > max_jump:
                return -1.0, None
            proximity = max(0.0, 1.0 - dist_to_prediction / max(1.0, max_jump))

        appearance_score = min(1.0, self._patch_fraction(appearance, cx, cy, radius, scale=1.6) * 2.4)
        skin_fraction = self._patch_fraction(skin, cx, cy, radius, scale=1.5)

        # On first acquisition, a strongly skin-coloured tiny blob is almost
        # certainly a finger/hand highlight rather than a free-moving ball.
        if predicted is None and skin_fraction >= self.skin_hard_fraction:
            return -1.0, None

        compactness = min(1.0, fill * 1.35)
        motion_score = min(1.0, moving * 6.0)
        size_score = 1.0 - min(1.0, max(0.0, radius - 0.8) / max(1.0, self.max_radius - 0.8))

        if predicted is None:
            score = (
                0.35 * motion_score
                + 0.22 * compactness
                + 0.23 * appearance_score
                + 0.17 * size_score
                + 0.03 * proximity
            )
        else:
            score = (
                0.54 * proximity
                + 0.17 * motion_score
                + 0.11 * compactness
                + 0.10 * appearance_score
                + 0.08 * size_score
            )

        # Skin penalty remains active while tracking, but is relaxed if the
        # candidate lands very close to the predicted trajectory (possible hit).
        trajectory_agrees = dist_to_prediction is not None and dist_to_prediction <= self.impact_allow_distance_px
        if skin_fraction > 0.0:
            strength = self.skin_penalty_strength * (0.35 if trajectory_agrees else 1.0)
            score *= max(0.08, 1.0 - strength * min(1.0, skin_fraction * 1.8))

        score *= self._context_multiplier(point, predicted, boxes, hands)
        return float(score), point

    def _confirm_acquisition(self, point: np.ndarray, diag: float) -> bool:
        if self.acquire_confirm_frames <= 1:
            return True
        if self.pending_pos is None:
            self.pending_pos = point.copy()
            self.pending_count = 1
            return False

        dist = float(np.linalg.norm(point - self.pending_pos))
        if dist <= self.acquire_confirm_jump_ratio * diag:
            self.pending_count += 1
            self.pending_pos = point.copy()
        else:
            self.pending_pos = point.copy()
            self.pending_count = 1

        if self.pending_count >= self.acquire_confirm_frames:
            self.pending_pos = None
            self.pending_count = 0
            return True
        return False

    def _clear_pending(self):
        self.pending_pos = None
        self.pending_count = 0

    def update(self, frame, player_boxes=None, keypoints_xy=None, keypoints_conf=None) -> BallObservation:
        h, w = frame.shape[:2]
        diag = math.hypot(w, h)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        court_mask = self._court_mask(frame.shape)
        appearance = self._appearance_mask(hsv, court_mask)
        skin = self._skin_mask(frame, court_mask)
        raw_motion, motion = self._motion_mask(gray)
        boxes, hands = self._pose_context(player_boxes, keypoints_xy, keypoints_conf)

        candidate_mask = cv2.bitwise_and(appearance, motion) if self.last_pos is None else appearance
        contours, _ = cv2.findContours(candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        predicted = self._predict()
        best = None
        best_score = -1.0

        for contour in contours:
            score, point = self._candidate_score(
                contour,
                raw_motion,
                appearance,
                skin,
                predicted,
                diag,
                boxes,
                hands,
            )
            if point is not None and score > best_score:
                best_score = score
                best = point

        if best is None or best_score < self.min_confidence:
            self.missed += 1
            if self.last_pos is None and self.missed > 1:
                self._clear_pending()

            if self.last_pos is not None and self.missed <= self.predict_frames:
                self.last_pos = (self.last_pos + self.velocity).astype(np.float32)
                self.velocity = (self.velocity * self.velocity_damping).astype(np.float32)
                confidence = max(self.min_confidence, 0.42 - 0.08 * (self.missed - 1))
                return BallObservation(tuple(map(float, self.last_pos)), confidence, True)

            if self.missed >= self.reset_after_missed:
                self.last_pos = None
                self.velocity[:] = 0.0
                self._clear_pending()
            return BallObservation(None if self.last_pos is None else tuple(map(float, self.last_pos)), 0.0, False)

        if self.last_pos is None and not self._confirm_acquisition(best, diag):
            self.missed = 0
            return BallObservation(None, 0.0, False)

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
        self._clear_pending()
        return BallObservation(tuple(map(float, self.last_pos)), float(min(1.0, best_score)), True)
