from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence
import math

import cv2
import numpy as np

from padel_poc.trajectory_predictor import BallTrajectoryPredictor, TrajectoryPrediction


@dataclass
class FocusInfo:
    track_id: Optional[int]
    crop_rect: tuple[int, int, int, int]
    mode: str = "COURT"
    ball_point: Optional[tuple[float, float]] = None
    predicted_point: Optional[tuple[float, float]] = None
    predicted_zone: str = "UNKNOWN"
    prediction_confidence: float = 0.0


class SmartPortraitReframer:
    """Stable 9:16 virtual camera for padel highlights.

    v0.10 adds a short-horizon trajectory director. The portrait crop does not
    chase every ball pixel. Instead, recent ball movement is projected into the
    calibrated court, a LEFT/CENTER/RIGHT destination is predicted ~0.3 s ahead,
    and the camera begins a slow glide only when that predicted action would
    approach the portrait edge.

    This keeps the sports-highlight feel while still anticipating a serve,
    smash, volley, rebound, or other clear direction change after the ball has
    started moving on its new trajectory.
    """

    STATE_SCORES = {
        "SWING": 7.0,
        "COOLDOWN": 5.5,
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

        # Wide, fixed 9:16 crop. No dynamic zoom while rally is active.
        self.crop_height_ratio = float(cfg.get("crop_height_ratio", 1.0))
        self.vertical_bias = float(cfg.get("vertical_bias", 0.015))

        # Stable camera controls.
        self.safe_zone_ratio = float(cfg.get("safe_zone_ratio", 0.24))
        self.prediction_safe_zone_ratio = float(cfg.get("prediction_safe_zone_ratio", 0.20))
        self.pan_time_constant = float(cfg.get("pan_time_constant", 0.72))
        self.prediction_pan_time_constant = float(cfg.get("prediction_pan_time_constant", 0.82))
        self.player_pan_time_constant = float(cfg.get("player_pan_time_constant", 0.92))
        self.recenter_time_constant = float(cfg.get("recenter_time_constant", 1.80))
        self.max_pan_speed_ratio = float(cfg.get("max_pan_speed_ratio", 0.62))
        self.ball_min_confidence = float(cfg.get("ball_min_confidence", 0.28))
        self.ball_hold_seconds = float(cfg.get("ball_hold_seconds", 0.18))

        # Trajectory guidance is intentionally directional rather than exact.
        self.prediction_min_confidence = float(cfg.get("prediction_min_confidence", 0.38))
        self.prediction_lead_weight = float(cfg.get("prediction_lead_weight", 0.70))
        trajectory_cfg = dict(cfg.get("trajectory", {}) or {})
        self.trajectory = BallTrajectoryPredictor(
            frame_size=(self.frame_w, self.frame_h),
            fps=self.fps,
            court_polygon=court_polygon,
            cfg=trajectory_cfg,
        )
        self.prediction = TrajectoryPrediction()

        self.player_motion_weight = float(cfg.get("player_motion_weight", 4.0))
        self.sticky_bonus = float(cfg.get("sticky_bonus", 0.9))
        self.max_fallback_lock_seconds = float(cfg.get("max_fallback_lock_seconds", 0.35))
        self.lock_seconds = float(cfg.get("lock_seconds", 0.35))

        self.court_polygon = court_polygon
        self.tracks: Dict[int, tuple[np.ndarray, object]] = {}
        self.track_centers: Dict[int, np.ndarray] = {}
        self.track_motion: Dict[int, float] = {}
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
        ratio = float(np.clip(self.crop_height_ratio, 0.80, 1.0))
        return min(self._max_valid_crop_height(), ratio * self.frame_h)

    def update_tracks(self, track_ids, boxes, statuses: dict):
        previous_centers = self.track_centers
        new_tracks: Dict[int, tuple[np.ndarray, object]] = {}
        new_centers: Dict[int, np.ndarray] = {}
        new_motion: Dict[int, float] = {}

        for i, track_id in enumerate(track_ids):
            if i >= len(boxes):
                break
            track_id = int(track_id)
            bbox = np.asarray(boxes[i], dtype=np.float32)
            status = statuses.get(track_id)
            x1, y1, x2, y2 = map(float, bbox)
            center = np.asarray([0.5 * (x1 + x2), 0.5 * (y1 + y2)], dtype=np.float32)

            prev = previous_centers.get(track_id)
            if prev is None:
                motion = 0.0
            else:
                displacement = float(np.linalg.norm(center - prev))
                bbox_h = max(20.0, y2 - y1)
                motion = min(3.0, displacement / bbox_h)

            old_motion = self.track_motion.get(track_id, 0.0)
            motion = 0.78 * old_motion + 0.22 * motion

            new_tracks[track_id] = (bbox, status)
            new_centers[track_id] = center
            new_motion[track_id] = motion

        self.tracks = new_tracks
        self.track_centers = new_centers
        self.track_motion = new_motion

    def update_ball(
        self,
        point: Optional[tuple[float, float]],
        confidence: float,
        frame_index: int,
        found: bool = True,
    ):
        # Always update prediction state so its hold/staleness logic advances
        # even when the classical tracker temporarily loses the ball.
        self.prediction = self.trajectory.update(
            point=point,
            observation_confidence=float(confidence),
            frame_index=int(frame_index),
            found=bool(found),
        )

        if point is None:
            return
        if found and confidence >= self.ball_min_confidence:
            self.ball_point = np.asarray(point, dtype=np.float32)
            self.ball_confidence = float(confidence)
            self.ball_last_seen_frame = int(frame_index)

    def lock_target(self, track_id: int, frame_index: int, seconds: Optional[float] = None):
        self.locked_target = int(track_id)
        requested = self.lock_seconds if seconds is None else float(seconds)
        duration = min(max(0.05, requested), max(0.05, self.max_fallback_lock_seconds))
        self.lock_until_frame = int(frame_index + max(1.0, duration * self.fps))
        self.current_target = int(track_id)

    def _score(self, track_id: int, status) -> float:
        if status is None:
            score = 0.0
        else:
            state = str(getattr(status, "state", "IDLE")).upper()
            confidence = float(getattr(status, "confidence", 0.0) or 0.0)
            score = self.STATE_SCORES.get(state, 0.5) + 1.2 * confidence

        score += self.player_motion_weight * self.track_motion.get(track_id, 0.0)
        if track_id == self.current_target:
            score += self.sticky_bonus
        return score

    def _select_target(self, frame_index: int) -> Optional[int]:
        if self.locked_target is not None:
            if frame_index <= self.lock_until_frame and self.locked_target in self.tracks:
                return self.locked_target
            self.locked_target = None

        if not self.tracks:
            self.current_target = None
            return None

        best_id = max(self.tracks.keys(), key=lambda tid: self._score(tid, self.tracks[tid][1]))
        self.current_target = int(best_id)
        return self.current_target

    def _player_anchor_x(self, frame_index: int) -> tuple[float, Optional[int]]:
        target_id = self._select_target(frame_index)
        if target_id is None or target_id not in self.tracks:
            return self._fallback_center()[0], target_id
        bbox, _ = self.tracks[target_id]
        x1, _, x2, _ = map(float, bbox)
        return 0.5 * (x1 + x2), target_id

    def _ball_is_fresh(self, frame_index: int) -> bool:
        hold_frames = max(1, int(round(self.ball_hold_seconds * self.fps)))
        return self.ball_point is not None and frame_index - self.ball_last_seen_frame <= hold_frames

    def _safe_zone_target(self, target_x: float, ratio: Optional[float] = None) -> float:
        """Move only enough to keep target inside a broad horizontal safe-zone."""
        crop_w = self.crop_h * self.aspect
        half_w = 0.5 * crop_w
        chosen_ratio = self.safe_zone_ratio if ratio is None else float(ratio)
        margin = float(np.clip(chosen_ratio, 0.10, 0.42)) * crop_w
        left_safe = self.center_x - half_w + margin
        right_safe = self.center_x + half_w - margin

        if target_x < left_safe:
            return target_x + half_w - margin
        if target_x > right_safe:
            return target_x - half_w + margin
        return self.center_x

    def _prediction_target_x(self) -> Optional[float]:
        pred = self.prediction
        if not pred.valid or pred.point is None or pred.confidence < self.prediction_min_confidence:
            return None

        predicted_x = float(pred.point[0])
        if self.ball_point is not None:
            current_x = float(self.ball_point[0])
            lead = float(np.clip(self.prediction_lead_weight, 0.0, 1.0))
            predicted_x = current_x + lead * (predicted_x - current_x)
        return predicted_x

    def _desired_center_x(self, frame_index: int) -> tuple[float, str, Optional[int]]:
        fallback_x, _ = self._fallback_center()
        player_x, target_id = self._player_anchor_x(frame_index)

        prediction_x = self._prediction_target_x()
        if prediction_x is not None:
            zone = self.prediction.zone if self.prediction.zone else "?"
            # Prediction uses a slightly larger safe zone: anticipate direction,
            # but do not swing the camera unless composition really needs it.
            desired = self._safe_zone_target(prediction_x, self.prediction_safe_zone_ratio)
            return float(desired), f"PRED-{zone}", target_id

        if self._ball_is_fresh(frame_index):
            return float(self._safe_zone_target(float(self.ball_point[0]))), "BALL", target_id

        if target_id is not None:
            return float(self._safe_zone_target(player_x)), "PLAYER", target_id
        return float(fallback_x), "COURT", target_id

    def _ease_pan(self, desired_x: float, mode: str):
        error = desired_x - self.center_x
        if abs(error) < 0.75:
            return

        if mode.startswith("PRED-"):
            tau = self.prediction_pan_time_constant
            speed_scale = 0.88
        elif mode == "BALL":
            tau = self.pan_time_constant
            speed_scale = 0.82
        elif mode == "PLAYER":
            tau = self.player_pan_time_constant
            speed_scale = 0.68
        else:
            tau = self.recenter_time_constant
            speed_scale = 0.40

        dt = 1.0 / max(1.0, self.fps)
        alpha = 1.0 - math.exp(-dt / max(0.05, tau))
        raw_step = alpha * error

        crop_w = self.crop_h * self.aspect
        max_step = max(1.0, speed_scale * self.max_pan_speed_ratio * crop_w * dt)
        self.center_x += float(np.clip(raw_step, -max_step, max_step))

    def _crop_rect(self) -> tuple[int, int, int, int]:
        crop_h = min(self.crop_h, self._max_valid_crop_height())
        crop_w = crop_h * self.aspect
        half_w, half_h = crop_w * 0.5, crop_h * 0.5

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
        # Fixed zoom and fixed vertical composition. Prediction only affects the
        # horizontal camera target, and even that is constrained by a safe zone.
        self.crop_h = self._fixed_crop_height()
        _, fixed_y = self._fallback_center()
        self.center_y = fixed_y

        desired_x, mode, target_id = self._desired_center_x(frame_index)
        self._ease_pan(desired_x, mode)

        x1, y1, x2, y2 = self._crop_rect()
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            portrait = np.zeros((self.output_h, self.output_w, 3), dtype=np.uint8)
        else:
            portrait = cv2.resize(crop, (self.output_w, self.output_h), interpolation=cv2.INTER_LINEAR)

        ball = None if self.ball_point is None else tuple(map(float, self.ball_point))
        predicted = self.prediction.point if self.prediction.valid else None
        self.last_focus = FocusInfo(
            target_id,
            (x1, y1, x2, y2),
            mode,
            ball,
            predicted,
            self.prediction.zone,
            self.prediction.confidence,
        )
        return portrait

    def draw_source_crop(self, frame, color=(255, 120, 255)):
        x1, y1, x2, y2 = self.last_focus.crop_rect
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        label = f"PORTRAIT {self.last_focus.mode}"
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

        ball_xy = None
        if self.last_focus.ball_point is not None:
            ball_xy = tuple(map(int, self.last_focus.ball_point))
            cv2.circle(frame, ball_xy, 7, (0, 255, 255), 2, cv2.LINE_AA)

        if self.last_focus.predicted_point is not None:
            pred_xy = tuple(map(int, self.last_focus.predicted_point))
            if ball_xy is not None:
                cv2.arrowedLine(frame, ball_xy, pred_xy, (255, 0, 255), 2, cv2.LINE_AA, tipLength=0.12)
            cv2.circle(frame, pred_xy, 9, (255, 0, 255), 2, cv2.LINE_AA)
            text = f"{self.last_focus.predicted_zone} {self.last_focus.prediction_confidence:.0%}"
            cv2.putText(
                frame,
                text,
                (pred_xy[0] + 8, max(20, pred_xy[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 0, 255),
                1,
                cv2.LINE_AA,
            )
