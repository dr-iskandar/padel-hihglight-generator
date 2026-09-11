from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Sequence
import math

import cv2
import numpy as np


COURT_WIDTH_M = 10.0
COURT_LENGTH_M = 20.0


@dataclass
class TrajectoryPrediction:
    valid: bool = False
    point: Optional[tuple[float, float]] = None
    court_point: Optional[tuple[float, float]] = None
    zone: str = "UNKNOWN"
    confidence: float = 0.0
    speed_mps: float = 0.0
    horizon_seconds: float = 0.0
    event: str = "NONE"


class BallTrajectoryPredictor:
    """Short-horizon padel-ball trajectory predictor.

    This is deliberately lightweight for the POC. It converts observed image
    positions to calibrated court coordinates, estimates the recent velocity,
    predicts a short distance ahead, and classifies the destination as
    LEFT/CENTER/RIGHT. When a sharp direction change is observed (typical of a
    racket hit, wall contact, or bounce), old history is discarded so the new
    trajectory wins quickly instead of averaging both directions together.

    It is *not* a physics-accurate landing-point model yet. Its purpose is to
    give the portrait camera an early directional cue so it can glide toward
    the next area of play instead of chasing the ball reactively.
    """

    def __init__(
        self,
        frame_size,
        fps: float,
        court_polygon: Optional[Sequence[Sequence[float]]],
        cfg: Optional[dict] = None,
    ):
        cfg = cfg or {}
        self.frame_w, self.frame_h = map(int, frame_size)
        self.fps = max(1.0, float(fps))
        self.horizon_seconds = float(cfg.get("horizon_seconds", 0.32))
        self.max_history = max(4, int(cfg.get("max_history", 8)))
        self.min_points = max(3, int(cfg.get("min_points", 3)))
        self.min_span_seconds = float(cfg.get("min_span_seconds", 0.055))
        self.min_speed_mps = float(cfg.get("min_speed_mps", 1.2))
        self.max_speed_mps = float(cfg.get("max_speed_mps", 55.0))
        self.reset_angle_deg = float(cfg.get("reset_angle_deg", 52.0))
        self.reset_speed_ratio = float(cfg.get("reset_speed_ratio", 2.8))
        self.hold_seconds = float(cfg.get("hold_seconds", 0.20))
        self.min_confidence = float(cfg.get("min_confidence", 0.30))
        self.zone_margin_m = float(cfg.get("zone_margin_m", 0.25))

        self.history: Deque[tuple[int, np.ndarray, float]] = deque(maxlen=self.max_history)
        self.last_prediction = TrajectoryPrediction()
        self.last_observation_frame = -10**9
        self.last_velocity: Optional[np.ndarray] = None

        self.image_to_world = None
        self.world_to_image = None
        if court_polygon:
            pts = np.asarray(court_polygon, dtype=np.float32)
            if pts.shape == (4, 2):
                image_quad = pts.copy()
                image_quad[:, 0] *= self.frame_w
                image_quad[:, 1] *= self.frame_h
                world_quad = np.asarray(
                    [[0.0, 0.0], [COURT_WIDTH_M, 0.0],
                     [COURT_WIDTH_M, COURT_LENGTH_M], [0.0, COURT_LENGTH_M]],
                    dtype=np.float32,
                )
                self.image_to_world = cv2.getPerspectiveTransform(image_quad, world_quad)
                self.world_to_image = cv2.getPerspectiveTransform(world_quad, image_quad)

    @property
    def calibrated(self) -> bool:
        return self.image_to_world is not None and self.world_to_image is not None

    def reset(self):
        self.history.clear()
        self.last_velocity = None
        self.last_prediction = TrajectoryPrediction()

    def _to_world(self, point) -> Optional[np.ndarray]:
        if not self.calibrated or point is None:
            return None
        arr = np.asarray(point, dtype=np.float32).reshape(1, 1, 2)
        out = cv2.perspectiveTransform(arr, self.image_to_world).reshape(2)
        if not np.all(np.isfinite(out)):
            return None
        return out.astype(np.float32)

    def _to_image(self, point) -> Optional[tuple[float, float]]:
        if not self.calibrated or point is None:
            return None
        arr = np.asarray(point, dtype=np.float32).reshape(1, 1, 2)
        out = cv2.perspectiveTransform(arr, self.world_to_image).reshape(2)
        if not np.all(np.isfinite(out)):
            return None
        return float(out[0]), float(out[1])

    @staticmethod
    def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < 1e-6 or nb < 1e-6:
            return 0.0
        cosine = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
        return math.degrees(math.acos(cosine))

    def _segment_velocities(self):
        velocities = []
        if len(self.history) < 2:
            return velocities
        items = list(self.history)
        for (f0, p0, _), (f1, p1, _) in zip(items[:-1], items[1:]):
            dt = (f1 - f0) / self.fps
            if dt <= 1e-6:
                continue
            velocity = (p1 - p0) / dt
            speed = float(np.linalg.norm(velocity))
            if self.min_speed_mps * 0.20 <= speed <= self.max_speed_mps:
                velocities.append(velocity.astype(np.float32))
        return velocities

    def _direction_change_event(self, new_velocity: np.ndarray) -> bool:
        if self.last_velocity is None:
            return False
        old_speed = float(np.linalg.norm(self.last_velocity))
        new_speed = float(np.linalg.norm(new_velocity))
        if old_speed < 1e-5 or new_speed < 1e-5:
            return False
        angle = self._angle_deg(self.last_velocity, new_velocity)
        speed_ratio = max(old_speed, new_speed) / max(1e-5, min(old_speed, new_speed))
        return angle >= self.reset_angle_deg or (
            speed_ratio >= self.reset_speed_ratio and angle >= self.reset_angle_deg * 0.55
        )

    def _zone(self, x_m: float) -> str:
        one_third = COURT_WIDTH_M / 3.0
        two_thirds = 2.0 * COURT_WIDTH_M / 3.0
        margin = max(0.0, self.zone_margin_m)
        if x_m < one_third - margin:
            return "LEFT"
        if x_m > two_thirds + margin:
            return "RIGHT"
        return "CENTER"

    def _predict(self, frame_index: int, event: str = "NONE") -> TrajectoryPrediction:
        if not self.calibrated or len(self.history) < self.min_points:
            return TrajectoryPrediction(event=event)

        items = list(self.history)
        span = (items[-1][0] - items[0][0]) / self.fps
        if span < self.min_span_seconds:
            return TrajectoryPrediction(event=event)

        velocities = self._segment_velocities()
        if len(velocities) < 2:
            return TrajectoryPrediction(event=event)

        vel_stack = np.stack(velocities, axis=0)
        velocity = np.median(vel_stack, axis=0).astype(np.float32)
        speed = float(np.linalg.norm(velocity))
        if speed < self.min_speed_mps or speed > self.max_speed_mps:
            return TrajectoryPrediction(event=event)

        # Direction consistency: erratic candidate tracks should not drive camera movement.
        unit = velocity / max(1e-6, speed)
        alignments = []
        for v in velocities:
            norm = float(np.linalg.norm(v))
            if norm > 1e-6:
                alignments.append(max(0.0, float(np.dot(v / norm, unit))))
        consistency = float(np.mean(alignments)) if alignments else 0.0

        obs_conf = float(np.mean([c for _, _, c in items[-self.max_history:]]))
        count_score = min(1.0, len(velocities) / 5.0)
        span_score = min(1.0, span / 0.16)
        confidence = float(np.clip(obs_conf * consistency * (0.45 + 0.55 * count_score) * span_score, 0.0, 1.0))
        if confidence < self.min_confidence:
            return TrajectoryPrediction(event=event)

        last_world = items[-1][1]
        predicted = last_world + velocity * self.horizon_seconds
        predicted[0] = float(np.clip(predicted[0], 0.0, COURT_WIDTH_M))
        predicted[1] = float(np.clip(predicted[1], 0.0, COURT_LENGTH_M))
        image_point = self._to_image(predicted)
        if image_point is None:
            return TrajectoryPrediction(event=event)

        self.last_velocity = velocity
        return TrajectoryPrediction(
            valid=True,
            point=image_point,
            court_point=(float(predicted[0]), float(predicted[1])),
            zone=self._zone(float(predicted[0])),
            confidence=confidence,
            speed_mps=speed,
            horizon_seconds=self.horizon_seconds,
            event=event,
        )

    def update(
        self,
        point: Optional[tuple[float, float]],
        observation_confidence: float,
        frame_index: int,
        found: bool = True,
    ) -> TrajectoryPrediction:
        frame_index = int(frame_index)

        if not found or point is None or observation_confidence <= 0.0:
            hold_frames = max(1, int(round(self.hold_seconds * self.fps)))
            if frame_index - self.last_observation_frame <= hold_frames:
                return self.last_prediction
            self.last_prediction = TrajectoryPrediction()
            return self.last_prediction

        world = self._to_world(point)
        if world is None:
            self.last_prediction = TrajectoryPrediction()
            return self.last_prediction

        event = "NONE"
        if self.history:
            prev_frame, prev_world, _ = self.history[-1]
            dt = (frame_index - prev_frame) / self.fps
            if dt > 1e-6:
                new_velocity = (world - prev_world) / dt
                speed = float(np.linalg.norm(new_velocity))
                if self.min_speed_mps * 0.20 <= speed <= self.max_speed_mps and self._direction_change_event(new_velocity):
                    # Keep only the previous point so prediction restarts on the
                    # post-hit/post-bounce direction as soon as possible.
                    previous = self.history[-1]
                    self.history.clear()
                    self.history.append(previous)
                    self.last_velocity = new_velocity.astype(np.float32)
                    event = "DIRECTION_CHANGE"

        self.history.append((frame_index, world, float(np.clip(observation_confidence, 0.0, 1.0))))
        self.last_observation_frame = frame_index
        prediction = self._predict(frame_index, event=event)
        self.last_prediction = prediction
        return prediction
