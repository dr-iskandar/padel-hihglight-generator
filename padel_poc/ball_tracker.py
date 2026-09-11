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
    """Pose-aware small-ball tracker for the portrait virtual camera.

    v0.11 keeps the lightweight POC approach, but fixes the most common false
    positive in the current sample: hands / wrists being selected as the ball.

    The tracker now combines:
      - yellow/green + bright/white appearance;
      - frame motion;
      - small/compact-object geometry;
      - predicted-position continuity;
      - player bounding-box context;
      - wrist/hand exclusion from YOLO pose;
      - short confirmation before acquiring a completely new track.

    Important: a ball can legitimately pass near a racket/hand at impact. Hand
    rejection is therefore softened when the candidate also agrees with the
    existing predicted ball trajectory. This avoids throwing away real impact
    frames while strongly suppressing random moving hands during acquisition.
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
        self.max_area = float(cfg.get("max_area", 110.0))
        self.max_radius = float(cfg.get("max_radius", 8.5))
        self.max_aspect_ratio = float(cfg.get("max_aspect_ratio", 2.5))
        self.max_jump_ratio = float(cfg.get("max_jump_ratio", 0.20))
        self.position_smoothing = float(cfg.get("position_smoothing", 0.78))
        self.velocity_smoothing = float(cfg.get("velocity_smoothing", 0.62))
        self.min_confidence = float(cfg.get("min_confidence", 0.27))
        self.acquire_motion_fraction = float(cfg.get("acquire_motion_fraction", 0.020))
        self.tracked_motion_fraction = float(cfg.get("tracked_motion_fraction", 0.004))
        self.predict_frames = max(0, int(cfg.get("predict_frames", 3)))
        self.reset_after_missed = max(1, int(cfg.get("reset_after_missed", 7)))
        self.velocity_damping = float(cfg.get("velocity_damping", 0.86))

        # Pose-aware rejection.
        self.hand_keypoint_conf = float(cfg.get("hand_keypoint_conf", 0.28))
        self.hand_radius_ratio = float(cfg.get("hand_radius_ratio", 0.15))
        self.hand_radius_min_px = float(cfg.get("hand_radius_min_px", 14.0))
        self.hand_reject_multiplier = float(cfg.get("hand_reject_multiplier", 0.18))
        self.person_inside_multiplier = float(cfg.get("person_inside_multiplier", 0.55))
        self.impact_allow_distance_px = float(cfg.get("impact_allow_distance_px", 28.0))

        # A completely new track must be seen consistently twice. This stops a
        # single moving fingertip / wrist highlight from instantly hijacking the
        # portrait camera.
        self.acquire_confirm_frames = max(1, int(cfg.get("acquire_confirm_frames", 2)))
        self.acquire_confirm_jump_ratio = float(cfg.get("acquire_confirm_jump_ratio", 0.075))

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
        hands = []  # (x, y, radius)
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
        # A tracked ball that agrees with its predicted trajectory may be very
        # close to the racket/hand during impact, so don't suppress it there.
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

    @staticmethod
    def _appearance_fraction(mask: np.ndarray, cx: float, cy: float, radius: float) -> float:
        r = max(2, int(round(radius * 1.6)))
        x1, x2 = max(0, int(cx) - r), min(mask.shape[1], int(cx) + r + 1)
        y1, y2 = max(0, int(cy) - r), min(mask.shape[0], int(cy) + r + 1)
        patch = mask[y1:y2, x1:x2]
        if patch.size == 0:
            return 0.0
        return float(np.count_nonzero(patch)) / float(patch.size)

    def _candidate_score(
        self,
        contour,
        motion: np.ndarray,
        appearance: np.ndarray,
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
        if fill < 0.07:
            return -1.0, None

        moving = self._motion_fraction(motion, cx, cy, radius)
        required_motion = self.tracked_motion_fraction if predicted is not None else self.acquire_motion_fraction
        if moving < required_motion:
            return -1.0, None

        point = np.asarray([cx, cy], dtype=np.float32)
        proximity = 0.45
        if predicted is not None:
            dist = float(np.linalg.norm(point - predicted))
            max_jump = self.max_jump_ratio * diag * (1.0 + min(0.55, 0.14 * self.missed))
            if dist > max_jump:
                return -1.0, None
            proximity = max(0.0, 1.0 - dist / max(1.0, max_jump))

        compactness = min(1.0, fill * 1.35)
        motion_score = min(1.0, moving * 6.0)
        appearance_score = min(1.0, self._appearance_fraction(appearance, cx, cy, radius) * 2.4)
        # Padel ball is tiny in the broadcast footage. Prefer small blobs.
        size_score = 1.0 - min(1.0, max(0.0, radius - 0.8) / max(1.0, self.max_radius - 0.8))

        if predicted is None:
            score = (
                0.34 * motion_score
                + 0.22 * compactness
                + 0.22 * appearance_score
                + 0.17 * size_score
                + 0.05 * proximity
            )
        else:
            score = (
                0.50 * proximity
                + 0.18 * motion_score
                + 0.13 * compactness
                + 0.11 * appearance_score
                + 0.08 * size_score
            )

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
        raw_motion, motion = self._motion_mask(gray)
        boxes, hands = self._pose_context(player_boxes, keypoints_xy, keypoints_conf)

        # During acquisition require both appearance and motion. Once a track is
        # established, appearance is enough because trajectory continuity becomes
        # a much stronger cue than colour alone.
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
            if self.last_pos is None:
                # Do not keep a stale one-frame hand candidate around forever.
                if self.missed > 1:
                    self._clear_pending()
            # Continue a short trajectory through blur/occlusion.
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

        # New tracks require short temporal confirmation. Established tracks don't.
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
