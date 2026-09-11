from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict, deque
from typing import Deque, Dict, Iterable, Optional, Sequence
import math
import numpy as np

LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12


@dataclass
class ServeEvent:
    track_id: int
    zone: str
    wrist_speed: float
    confidence: float
    frame_index: int


@dataclass
class ServeStatus:
    track_id: int
    zone: Optional[str] = None
    state: str = "IDLE"
    confidence: float = 0.0
    wrist_speed: float = 0.0
    body_speed: float = 0.0
    stationary_progress: float = 0.0
    prep_recent: bool = False
    forward_progress: float = 0.0


@dataclass
class _MotionSample:
    frame_index: int
    center_x: float
    center_y: float
    bbox_h: float
    left_wrist: Optional[tuple[float, float]]
    right_wrist: Optional[tuple[float, float]]


@dataclass
class _TrackState:
    state: str = "IDLE"
    zone: Optional[str] = None
    zone_enter_frame: int = -1
    stationary_since: Optional[int] = None
    prep_last_seen: int = -10**9
    swing_frame: Optional[int] = None
    swing_center_y: Optional[float] = None
    swing_bbox_h: float = 1.0
    last_trigger: int = -10**9
    last_seen: int = -10**9


class ServeDetector:
    """POC serve detector based on a temporal state machine."""

    def __init__(
        self,
        zones: Dict[str, Iterable],
        cfg: dict,
        fps: float,
        court_polygon: Optional[Sequence[Sequence[float]]] = None,
    ):
        self.zones = {k: self._normalize_polygon(v) for k, v in zones.items()}
        self.court_polygon = self._normalize_polygon(court_polygon) if court_polygon is not None else None
        self.fps = float(fps)
        self.min_kp_conf = float(cfg.get("min_keypoint_conf", 0.30))
        self.stationary_seconds = float(cfg.get("stationary_seconds", 0.35))
        self.stationary_speed_threshold = float(cfg.get("stationary_speed_threshold", 0.65))
        self.prep_memory_seconds = float(cfg.get("prep_memory_seconds", 0.80))
        self.prep_wrist_torso_ratio = float(cfg.get("prep_wrist_torso_ratio", 0.35))
        self.swing_speed_threshold = float(cfg.get("swing_speed_threshold", 2.2))
        self.wrist_lookback_seconds = float(cfg.get("wrist_lookback_seconds", 0.16))
        self.body_lookback_seconds = float(cfg.get("body_lookback_seconds", 0.20))
        self.confirm_window_seconds = float(cfg.get("confirm_window_seconds", 0.65))
        self.forward_move_threshold = float(cfg.get("forward_move_threshold", 0.08))
        self.require_forward_motion = bool(cfg.get("require_forward_motion", True))
        self.cooldown_seconds = float(cfg.get("cooldown_seconds", 6.0))
        self.global_cooldown_seconds = float(cfg.get("global_cooldown_seconds", 5.0))
        self.stale_track_seconds = float(cfg.get("stale_track_seconds", 2.0))

        history_seconds = max(1.0, self.prep_memory_seconds, self.confirm_window_seconds, self.wrist_lookback_seconds * 2, self.body_lookback_seconds * 2)
        self.history_frames = max(8, int(history_seconds * self.fps) + 4)
        self.history: Dict[int, Deque[_MotionSample]] = defaultdict(lambda: deque(maxlen=self.history_frames))
        self.tracks: Dict[int, _TrackState] = defaultdict(_TrackState)
        self.statuses: Dict[int, ServeStatus] = {}
        self.global_last_trigger = -10**9

    @staticmethod
    def _normalize_polygon(values) -> tuple[tuple[float, float], ...]:
        if values is None:
            return tuple()
        vals = list(values)
        if len(vals) == 4 and all(isinstance(v, (int, float, np.integer, np.floating)) for v in vals):
            x1, y1, x2, y2 = map(float, vals)
            return ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
        polygon = []
        for point in vals:
            if len(point) != 2:
                raise ValueError("Polygon points must be [x, y]")
            polygon.append((float(point[0]), float(point[1])))
        if len(polygon) < 3:
            raise ValueError("A polygon needs at least 3 points")
        return tuple(polygon)

    @staticmethod
    def _point_in_polygon(px: float, py: float, polygon) -> bool:
        if not polygon:
            return False
        inside = False
        n = len(polygon)
        j = n - 1
        eps = 1e-9
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            dx, dy = xj - xi, yj - yi
            cross = (px - xi) * dy - (py - yi) * dx
            if abs(cross) <= 1e-8:
                dot = (px - xi) * (px - xj) + (py - yi) * (py - yj)
                if dot <= eps:
                    return True
            intersects = ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / ((yj - yi) + eps) + xi)
            if intersects:
                inside = not inside
            j = i
        return inside

    @classmethod
    def _inside_zone(cls, px: float, py: float, zones: Dict[str, tuple]) -> Optional[str]:
        for name, polygon in zones.items():
            if cls._point_in_polygon(px, py, polygon):
                return name
        return None

    @staticmethod
    def _kp(keypoints: np.ndarray, idx: int):
        if keypoints.ndim != 2 or keypoints.shape[1] < 3 or idx >= len(keypoints):
            return None
        x, y, c = map(float, keypoints[idx, :3])
        return x, y, c

    @staticmethod
    def _avg_y(points) -> Optional[float]:
        points = [p for p in points if p is not None]
        if not points:
            return None
        return sum(p[1] for p in points) / len(points)

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _frames(self, seconds: float) -> int:
        return max(1, int(round(seconds * self.fps)))

    def _recent_sample(self, track_id: int, frame_index: int, seconds: float) -> Optional[_MotionSample]:
        target = frame_index - self._frames(seconds)
        hist = self.history[track_id]
        if not hist:
            return None
        candidate = hist[0]
        for sample in hist:
            if sample.frame_index <= target:
                candidate = sample
            else:
                break
        return candidate

    def _body_speed(self, track_id: int, current: _MotionSample) -> float:
        prev = self._recent_sample(track_id, current.frame_index, self.body_lookback_seconds)
        if prev is None or prev.frame_index == current.frame_index:
            return 0.0
        dt = max(1.0 / self.fps, (current.frame_index - prev.frame_index) / self.fps)
        scale = max(1.0, 0.5 * (current.bbox_h + prev.bbox_h))
        dist = math.hypot(current.center_x - prev.center_x, current.center_y - prev.center_y)
        return (dist / scale) / dt

    def _wrist_speed(self, track_id: int, current: _MotionSample) -> float:
        prev = self._recent_sample(track_id, current.frame_index, self.wrist_lookback_seconds)
        if prev is None or prev.frame_index == current.frame_index:
            return 0.0
        dt = max(1.0 / self.fps, (current.frame_index - prev.frame_index) / self.fps)
        scale = max(1.0, 0.5 * (current.bbox_h + prev.bbox_h))
        max_speed = 0.0
        for now_pt, prev_pt in ((current.left_wrist, prev.left_wrist), (current.right_wrist, prev.right_wrist)):
            if now_pt is None or prev_pt is None:
                continue
            dist = math.hypot(now_pt[0] - prev_pt[0], now_pt[1] - prev_pt[1])
            max_speed = max(max_speed, (dist / scale) / dt)
        return max_speed

    def _prep_evidence(self, keypoints: np.ndarray, bbox_h: float) -> bool:
        ls = self._kp(keypoints, LEFT_SHOULDER)
        rs = self._kp(keypoints, RIGHT_SHOULDER)
        lh = self._kp(keypoints, LEFT_HIP)
        rh = self._kp(keypoints, RIGHT_HIP)
        lw = self._kp(keypoints, LEFT_WRIST)
        rw = self._kp(keypoints, RIGHT_WRIST)
        shoulders = [p for p in (ls, rs) if p and p[2] >= self.min_kp_conf]
        hips = [p for p in (lh, rh) if p and p[2] >= self.min_kp_conf]
        wrists = [p for p in (lw, rw) if p and p[2] >= self.min_kp_conf]
        if not wrists:
            return False
        shoulder_y = self._avg_y(shoulders)
        hip_y = self._avg_y(hips)
        if shoulder_y is None:
            return False
        if hip_y is not None and hip_y > shoulder_y:
            torso_h = max(1.0, hip_y - shoulder_y)
            low_line = shoulder_y + self.prep_wrist_torso_ratio * torso_h
        else:
            low_line = shoulder_y + 0.16 * max(1.0, bbox_h)
        return any(p[1] >= low_line for p in wrists)

    @staticmethod
    def _is_near_zone(zone: str) -> bool:
        return str(zone).lower().startswith("near")

    def _forward_progress(self, track: _TrackState, current_center_y: float) -> float:
        if track.swing_center_y is None:
            return 0.0
        scale = max(1.0, track.swing_bbox_h)
        if track.zone and self._is_near_zone(track.zone):
            return max(0.0, (track.swing_center_y - current_center_y) / scale)
        return max(0.0, (current_center_y - track.swing_center_y) / scale)

    def _reset_to_zone(self, track: _TrackState, zone: str, frame_index: int):
        track.state = "IN_ZONE"
        track.zone = zone
        track.zone_enter_frame = frame_index
        track.stationary_since = None
        track.prep_last_seen = -10**9
        track.swing_frame = None
        track.swing_center_y = None
        track.swing_bbox_h = 1.0

    def _confidence(self, track: _TrackState, frame_index: int, wrist_speed: float, stationary_progress: float, prep_recent: bool, forward_progress: float) -> float:
        score = 0.10
        score += 0.25 * self._clamp01(stationary_progress)
        if prep_recent:
            score += 0.20
        if track.state in ("SWING", "COOLDOWN"):
            score += 0.30 * self._clamp01(wrist_speed / max(1e-6, self.swing_speed_threshold))
            if self.require_forward_motion:
                score += 0.15 * self._clamp01(forward_progress / max(1e-6, self.forward_move_threshold))
            else:
                score += 0.15
        else:
            score += 0.10 * self._clamp01(wrist_speed / max(1e-6, self.swing_speed_threshold))
        return self._clamp01(score)

    def get_status(self, track_id: int) -> ServeStatus:
        return self.statuses.get(int(track_id), ServeStatus(track_id=int(track_id)))

    def prune(self, frame_index: int):
        stale_frames = self._frames(self.stale_track_seconds)
        stale = [tid for tid, track in self.tracks.items() if frame_index - track.last_seen > stale_frames]
        for tid in stale:
            self.tracks.pop(tid, None)
            self.history.pop(tid, None)
            self.statuses.pop(tid, None)

    def update(self, track_id: int, bbox_xyxy, keypoints: np.ndarray, frame_shape, frame_index: int) -> Optional[ServeEvent]:
        track_id = int(track_id)
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = map(float, bbox_xyxy)
        bh = max(1.0, y2 - y1)
        center_x = 0.5 * (x1 + x2)
        center_y = 0.5 * (y1 + y2)
        foot_x = center_x / max(1.0, w)
        foot_y = y2 / max(1.0, h)

        if self.court_polygon is not None and not self._point_in_polygon(foot_x, foot_y, self.court_polygon):
            zone = None
        else:
            zone = self._inside_zone(foot_x, foot_y, self.zones)

        track = self.tracks[track_id]
        track.last_seen = frame_index

        if zone is None:
            track.state = "IDLE"
            track.zone = None
            track.stationary_since = None
            track.swing_frame = None
            track.swing_center_y = None
            self.history[track_id].clear()
            self.statuses[track_id] = ServeStatus(track_id=track_id)
            return None

        if track.zone != zone or track.state == "IDLE":
            self._reset_to_zone(track, zone, frame_index)

        lw = self._kp(keypoints, LEFT_WRIST)
        rw = self._kp(keypoints, RIGHT_WRIST)
        left_wrist = (lw[0], lw[1]) if lw and lw[2] >= self.min_kp_conf else None
        right_wrist = (rw[0], rw[1]) if rw and rw[2] >= self.min_kp_conf else None

        sample = _MotionSample(frame_index, center_x, center_y, bh, left_wrist, right_wrist)
        self.history[track_id].append(sample)
        body_speed = self._body_speed(track_id, sample)
        wrist_speed = self._wrist_speed(track_id, sample)
        prep_now = self._prep_evidence(keypoints, bh)
        if prep_now:
            track.prep_last_seen = frame_index
        prep_recent = frame_index - track.prep_last_seen <= self._frames(self.prep_memory_seconds)

        if body_speed <= self.stationary_speed_threshold:
            if track.stationary_since is None:
                track.stationary_since = frame_index
        elif track.state not in ("SWING", "COOLDOWN"):
            track.stationary_since = None

        if track.stationary_since is None:
            stationary_progress = 0.0
        else:
            stationary_progress = (frame_index - track.stationary_since) / max(1, self._frames(self.stationary_seconds))
        stationary_progress = self._clamp01(stationary_progress)

        if track.state == "IN_ZONE" and stationary_progress >= 1.0:
            track.state = "PREPARING"

        if track.state == "PREPARING" and prep_recent and wrist_speed >= self.swing_speed_threshold:
            track.state = "SWING"
            track.swing_frame = frame_index
            track.swing_center_y = center_y
            track.swing_bbox_h = bh

        forward_progress = self._forward_progress(track, center_y) if track.state == "SWING" else 0.0
        event: Optional[ServeEvent] = None

        if track.state == "SWING":
            swing_age = frame_index - int(track.swing_frame or frame_index)
            confirm_frames = self._frames(self.confirm_window_seconds)
            per_track_ready = frame_index - track.last_trigger >= self._frames(self.cooldown_seconds)
            global_ready = frame_index - self.global_last_trigger >= self._frames(self.global_cooldown_seconds)
            motion_confirmed = forward_progress >= self.forward_move_threshold if self.require_forward_motion else swing_age >= max(1, int(0.08 * self.fps))
            confidence = self._confidence(track, frame_index, wrist_speed, stationary_progress, prep_recent, forward_progress)

            if motion_confirmed and per_track_ready and global_ready:
                track.state = "COOLDOWN"
                track.last_trigger = frame_index
                self.global_last_trigger = frame_index
                event = ServeEvent(track_id, zone, wrist_speed, max(0.90, confidence), frame_index)
            elif swing_age > confirm_frames:
                self._reset_to_zone(track, zone, frame_index)
                stationary_progress = 0.0
                forward_progress = 0.0

        if track.state == "COOLDOWN" and frame_index - track.last_trigger >= self._frames(self.cooldown_seconds):
            self._reset_to_zone(track, zone, frame_index)
            stationary_progress = 0.0
            forward_progress = 0.0

        confidence = self._confidence(track, frame_index, wrist_speed, stationary_progress, prep_recent, forward_progress)
        if event is not None:
            confidence = max(confidence, event.confidence)

        self.statuses[track_id] = ServeStatus(track_id, zone, track.state, confidence, wrist_speed, body_speed, stationary_progress, prep_recent, forward_progress)
        return event
