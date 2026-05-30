from __future__ import annotations

import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class AudioFrame:
    pcm16: bytes
    sample_rate: int
    channels: int

    @property
    def samples(self) -> int:
        return len(self.pcm16) // (2 * self.channels)


class PreRollBuffer:
    def __init__(self, sample_rate: int, channels: int, max_ms: int) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.max_samples = int(sample_rate * max_ms / 1000)
        self.frames: deque[AudioFrame] = deque()
        self.samples = 0

    def add(self, frame: AudioFrame) -> None:
        if frame.sample_rate != self.sample_rate or frame.channels != self.channels:
            raise ValueError("frame format does not match pre-roll buffer")
        self.frames.append(frame)
        self.samples += frame.samples
        while self.samples > self.max_samples and self.frames:
            removed = self.frames.popleft()
            self.samples -= removed.samples

    def drain(self) -> list[AudioFrame]:
        return list(self.frames)


def write_wav(path: Path, frames: Iterable[AudioFrame], sample_rate: int, channels: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for frame in frames:
            if frame.sample_rate != sample_rate or frame.channels != channels:
                raise ValueError("cannot mix audio frame formats in one WAV")
            wav.writeframes(frame.pcm16)
    return path


def looks_fragmentary(text: str, trailing_words: list[str], min_chars: int) -> bool:
    stripped = text.strip()
    if len(stripped) < min_chars:
        return True
    last = stripped.rstrip(".,!?;:").split(" ")[-1].lower()
    return last in {word.lower() for word in trailing_words}
