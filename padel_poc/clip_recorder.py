from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
import math
import cv2


@dataclass
class ActiveClip:
    writer: cv2.VideoWriter
    path: Path
    start_frame: int
    end_frame: int
    hard_end_frame: int
    last_output_slot: int = -1


class ClipRecorder:
    """Buffered clip recorder that preserves source playback speed by default.

    `fps` is always the source-video FPS. Unless `output_fps` is explicitly set
    in config, the output uses the exact source FPS and writes every source
    frame. This keeps highlights at normal speed; the system trims moments, it
    does not speed them up.
    """

    def __init__(self, output_dir: str, fps: float, frame_size, cfg: dict):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.fps = float(fps)
        requested_output_fps = float(cfg.get("output_fps", self.fps))
        self.output_fps = max(1.0, min(self.fps, requested_output_fps))
        self.frame_size = tuple(map(int, frame_size))
        self.pre_frames = max(1, int(float(cfg.get("pre_roll_seconds", 3.0)) * self.fps))
        self.post_frames = max(1, int(float(cfg.get("post_roll_seconds", 7.0)) * self.fps))
        self.min_frames = max(1, int(float(cfg.get("min_clip_seconds", 5.0)) * self.fps))
        self.max_frames = max(self.min_frames, int(float(cfg.get("max_clip_seconds", 15.0)) * self.fps))
        self.codec = str(cfg.get("codec", "mp4v"))

        self.buffer = deque(maxlen=self.pre_frames)
        self.active: Optional[ActiveClip] = None
        self.last_saved: Optional[Path] = None

    def _output_slot(self, source_frame_index: int, start_frame: int) -> int:
        elapsed_source_frames = max(0, int(source_frame_index) - int(start_frame))
        elapsed_seconds = elapsed_source_frames / max(1e-6, self.fps)
        return int(math.floor(elapsed_seconds * self.output_fps + 1e-9))

    def _write_sampled(self, frame, frame_index: int):
        if self.active is None:
            return

        # Exact-source-FPS mode: write every frame. This is the default and is
        # intentionally the safest mode for highlight clips.
        if abs(self.output_fps - self.fps) < 1e-3:
            self.active.writer.write(frame)
            self.active.last_output_slot += 1
            return

        # Optional downsampling still preserves real-world duration.
        slot = self._output_slot(frame_index, self.active.start_frame)
        if slot <= self.active.last_output_slot:
            return
        self.active.writer.write(frame)
        self.active.last_output_slot = slot

    def push(self, frame, frame_index: int):
        self.buffer.append((frame_index, frame.copy()))

        if self.active is not None:
            self._write_sampled(frame, frame_index)
            if frame_index >= self.active.end_frame or frame_index >= self.active.hard_end_frame:
                self._finish()

    def trigger(self, frame_index: int, label: str = "serve") -> Path:
        if self.active is not None:
            requested_end = frame_index + self.post_frames
            self.active.end_frame = min(self.active.hard_end_frame, max(self.active.end_frame, requested_end))
            return self.active.path

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        path = self.output_dir / f"{timestamp}_{label}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        writer = cv2.VideoWriter(str(path), fourcc, self.output_fps, self.frame_size)
        if not writer.isOpened():
            raise RuntimeError(f"Could not open VideoWriter for {path}")

        start_frame = self.buffer[0][0] if self.buffer else frame_index
        min_end = start_frame + self.min_frames
        requested_end = frame_index + self.post_frames
        hard_end = start_frame + self.max_frames
        end_frame = min(hard_end, max(min_end, requested_end))

        self.active = ActiveClip(writer, path, start_frame, end_frame, hard_end)
        for buffered_index, buffered_frame in self.buffer:
            self._write_sampled(buffered_frame, buffered_index)
        return path

    def _finish(self):
        if self.active is None:
            return
        self.active.writer.release()
        self.last_saved = self.active.path
        self.active = None

    def close(self):
        self._finish()
