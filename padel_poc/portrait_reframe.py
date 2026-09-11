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
    """9:16 virtual camera with anticipatory panning and edge protection.

    v0.11 fixes two problems seen in real POC output:
      * the portrait camera started moving too late and lagged behind rallies;
      * players near the left/right glass could be cut off even though their
        centre point was still technically inside the crop.

    The director now starts panning from the trajectory prediction earlier,
    treats the active player's *whole bounding box* as a composition constraint,
    and enters a faster catch-up mode only when action is at real risk of being
    clipped. This gives earlier movement without returning to frame-by-frame
    ball chasing.
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

        # Keep maximum available width for a 9:16 crop. No zoom pumping.
        self.crop_height_ratio = float(cfg.get("crop_height_ratio", 1.0))
        self.vertical_bias = float(cfg.get("vertical_bias", 0.015))

        # Start moving BEFORE action reaches the portrait edge.
        self.safe_zone_ratio = float(cfg.get("safe_zone_ratio", 0.27))
        self.prediction_safe_zone_ratio = float(cfg.get("prediction_safe_zone_ratio", 0.32))
        self.player_edge_guard_ratio = float(cfg.get("player_edge_guard_ratio", 0.11))
        self.emergency_edge_ratio = float(cfg.get("emergency_edge_ratio", 0.055))

        # Normal pans are smooth; emergency catch-up is only used when clipping
        # is imminent. Lower time constant = earlier/faster response.
        self.pan_time_constant = float(cfg.get("pan_time_constant", 0.46))
        self.prediction_pan_time_constant = float(cfg.get("prediction_pan_time_constant", 0.32))
        self.player_pan_time_constant = float(cfg.get("player_pan_time_constant", 0.56))
        self.recenter_time_constant = float(cfg.get("recenter_time_constant", 1.85))
        self.max_pan_speed_ratio = float(cfg.get("max_pan_speed_ratio", 1.15))
        self.emergency_pan_speed_ratio = float(cfg.get("emergency_pan_speed_ratio", 2.10))
        self.emergency_time_constant = float(cfg.get("emergency_time_constant", 0.16))

        self.ball_min_confidence = float(cfg.get("ball_min_confidence", 0.28))
        self.ball_hold_seconds = float(cfg.get("ball_hold_seconds", 0.16))

        # Trajectory cue: longer and stronger look-ahead than v0.10 so the crop
        # is already travelling before the ball reaches the outside third.
        self.prediction_min_confidence = float(cfg.get("prediction_min_confidence", 0.34))
        self.prediction_lead_weight = float(cfg.get("prediction_lead_weight", 0.95))
        trajectory_cfg = dict(cfg.get("trajectory", {}) or {})
        trajectory_cfg.setdefault("horizon_seconds", 0.46)
        trajectory_cfg.setdefault("hold_seconds", 0.18)
        trajectory_cfg.setdefault("min_confidence", 0.28)
        self.trajectory = BallTrajectoryPredictor(
            frame_size=(self.frame_w, self.frame_h),
            fps=self.fps,
            court_polygon=court_polygon,
            cfg=trajectory_cfg,
        )
        self.prediction = TrajectoryPrediction()

        self.player_motion_weight = float(cfg.get("player_motion_weight", 4.5))
        self.sticky_bonus = float(cfg.get("sticky_bonus", 0.8))
        self.max_fallback_lock_seconds = float(cfg.get("max_fallback_lock_seconds", 0.28))
        self.lock_seconds = float(cfg.get("lock_seconds", 0.28))

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

    def _crop_width(self) -> float:
        return float(self.crop_h * self.aspect)

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
            motion = 0.74 * old_motion + 0.26 * motion

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

    def _active_player(self, frame_index: int):
        target_id = self._select_target(frame_index)
        if target_id is None or target_id not in self.tracks:
            return None, None
        bbox, _ = self.tracks[target_id]
        return target_id, np.asarray(bbox, dtype=np.float32)

    def _ball_is_fresh(self, frame_index: int) -> bool:
        hold_frames = max(1, int(round(self.ball_hold_seconds * self.fps)))
        return self.ball_point is not None and frame_index - self.ball_last_seen_frame <= hold_frames

    def _point_safe_target(self, target_x: float, ratio: float) -> float:
        """Desired camera centre that keeps a point away from crop edges."""
        crop_w = self._crop_width()
        half_w = 0.5 * crop_w
        margin = float(np.clip(ratio, 0.05, 0.42)) * crop_w
        left_safe = self.center_x - half_w + margin
        right_safe = self.center_x + half_w - margin

        if target_x < left_safe:
            return target_x + half_w - margin
        if target_x > right_safe:
            return target_x - half_w + margin
        return self.center_x

    def _interval_safe_target(self, x1: float, x2: float, ratio: float) -> float:
        """Move enough to keep an entire bbox/action interval visible."""
        if x2 < x1:
            x1, x2 = x2, x1
        crop_w = self._crop_width()
        half_w = 0.5 * crop_w
        margin = float(np.clip(ratio, 0.03, 0.28)) * crop_w
        inner_half = max(10.0, half_w - margin)

        # If the requested interval is wider than the usable portrait interior,
        # centre it rather than oscillating between both edges.
        if (x2 - x1) >= 2.0 * inner_half:
            return 0.5 * (x1 + x2)

        min_center = x2 - inner_half
        max_center = x1 + inner_half
        if self.center_x < min_center:
            return min_center
        if self.center_x > max_center:
            return max_center
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

    def _desired_center_x(self, frame_index: int) -> tuple[float, str, Optional[int], bool]:
        fallback_x, _ = self._fallback_center()
        target_id, player_bbox = self._active_player(frame_index)

        prediction_x = self._prediction_target_x()
        ball_x = float(self.ball_point[0]) if self._ball_is_fresh(frame_index) else None

        # 1) Predicted trajectory leads the camera. If active player and ball are
        # in the same local action region, keep the whole region inside portrait.
        if prediction_x is not None:
            desired = self._point_safe_target(prediction_x, self.prediction_safe_zone_ratio)
            mode = f"PRED-{self.prediction.zone if self.prediction.zone else '?'}"

            if player_bbox is not None:
                px1, _, px2, _ = map(float, player_bbox)
                action_points = [px1, px2, prediction_x]
                if ball_x is not None:
                    action_points.append(ball_x)
                action_min, action_max = min(action_points), max(action_points)
                usable = self._crop_width() * (1.0 - 2.0 * self.player_edge_guard_ratio)
                if action_max - action_min <= usable:
                    desired = self._interval_safe_target(
                        action_min, action_max, self.player_edge_guard_ratio
                    )

            return float(desired), mode, target_id, self._edge_risk(prediction_x, player_bbox)

        # 2) Fresh ball is next priority. Keep hitter visible too when possible.
        if ball_x is not None:
            desired = self._point_safe_target(ball_x, self.safe_zone_ratio)
            if player_bbox is not None:
                px1, _, px2, _ = map(float, player_bbox)
                action_min, action_max = min(px1, ball_x), max(px2, ball_x)
                usable = self._crop_width() * (1.0 - 2.0 * self.player_edge_guard_ratio)
                if action_max - action_min <= usable:
                    desired = self._interval_safe_target(
                        action_min, action_max, self.player_edge_guard_ratio
                    )
            return float(desired), "BALL", target_id, self._edge_risk(ball_x, player_bbox)

        # 3) If the ball is lost, protect the complete active player bbox. This
        # specifically prevents players at the glass from being half cut off.
        if player_bbox is not None:
            px1, _, px2, _ = map(float, player_bbox)
            desired = self._interval_safe_target(px1, px2, self.player_edge_guard_ratio)
            return float(desired), "PLAYER", target_id, self._edge_risk(None, player_bbox)

        return float(fallback_x), "COURT", target_id, False

    def _edge_risk(self, point_x: Optional[float], player_bbox) -> bool:
        """True only when action is close to or outside the current crop edge."""
        crop_w = self._crop_width()
        half_w = 0.5 * crop_w
        margin = float(np.clip(self.emergency_edge_ratio, 0.02, 0.15)) * crop_w
        left = self.center_x - half_w + margin
        right = self.center_x + half_w - margin

        if point_x is not None and (point_x < left or point_x > right):
            return True
        if player_bbox is not None:
            px1, _, px2, _ = map(float, player_bbox)
            if px1 < left or px2 > right:
                return True
        return False

    def _ease_pan(self, desired_x: float, mode: str, emergency: bool = False):
        error = desired_x - self.center_x
        if abs(error) < 0.75:
            return

        if emergency:
            tau = self.emergency_time_constant
            speed_ratio = self.emergency_pan_speed_ratio
        elif mode.startswith("PRED-"):
            tau = self.prediction_pan_time_constant
            speed_ratio = self.max_pan_speed_ratio
        elif mode == "BALL":
            tau = self.pan_time_constant
            speed_ratio = self.max_pan_speed_ratio * 0.92
        elif mode == "PLAYER":
            tau = self.player_pan_time_constant
            speed_ratio = self.max_pan_speed_ratio * 0.82
        else:
            tau = self.recenter_time_constant
            speed_ratio = self.max_pan_speed_ratio * 0.36

        dt = 1.0 / max(1.0, self.fps)
        alpha = 1.0 - math.exp(-dt / max(0.04, tau))
        raw_step = alpha * error

        crop_w = self._crop_width()
        max_step = max(1.0, speed_ratio * crop_w * dt)
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
        self.crop_h = self._fixed_crop_height()
        _, fixed_y = self._fallback_center()
        self.center_y = fixed_y

        desired_x, mode, target_id, emergency = self._desired_center_x(frame_index)
        self._ease_pan(desired_x, mode, emergency=emergency)

        x1, y1, x2, y2 = self._crop_rect()
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            portrait = np.zeros((self.output_h, self.output_w, 3), dtype=np.uint8)
        else:
            portrait = cv2.resize(crop, (self.output_w, self.output_h), interpolation=cv2.INTER_LINEAR)

        ball = None if self.ball_point is None else tuple(map(float, self.ball_point))
        predicted = self.prediction.point if self.prediction.valid else None
        effective_mode = f"{mode}!" if emergency else mode
        self.last_focus = FocusInfo(
            target_id,
            (x1, y1, x2, y2),
            effective_mode,
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
                cv2.arrowedLine(
                    frame,
                    ball_xy,
                    pred_xy,
                    (255, 0, 255),
                    2,
                    cv2.LINE_AA,
                    tipLength=0.12,
                )
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
