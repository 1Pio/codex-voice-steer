from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .segment import AudioFrame, PreRollBuffer, write_wav


@dataclass(frozen=True)
class SegmentResult:
    wav_path: Path
    frame_count: int


class SegmentWriter:
    """Small testable bridge between capture/endpointing and file-based STT."""

    def __init__(self, config: Config) -> None:
        self.sample_rate = int(config.get("audio.sample_rate", 16000))
        self.channels = int(config.get("audio.channels", 1))
        self.pre_roll = PreRollBuffer(self.sample_rate, self.channels, int(config.get("audio.pre_roll_ms", 750)))

    def add_preroll(self, frame: AudioFrame) -> None:
        self.pre_roll.add(frame)

    def write_segment(self, path: Path, speech_frames: list[AudioFrame]) -> SegmentResult:
        frames = [*self.pre_roll.drain(), *speech_frames]
        write_wav(path, frames, self.sample_rate, self.channels)
        return SegmentResult(wav_path=path, frame_count=len(frames))
