from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Iterator

from .config import Config
from .segment import AudioFrame


@dataclass(frozen=True)
class AudioReadiness:
    ok: bool
    reason: str


def audio_readiness() -> AudioReadiness:
    if importlib.util.find_spec("sounddevice") is None:
        return AudioReadiness(False, "sounddevice is not installed, so microphone capture is unavailable")
    try:
        import sounddevice as sd

        device = sd.query_devices(kind="input")
    except Exception as exc:
        return AudioReadiness(False, f"default microphone input is unavailable: {exc}")
    return AudioReadiness(True, f"default microphone input available: {device.get('name', 'unknown')}")


class MicCapture:
    def __init__(self, config: Config, chunk_ms: int = 80) -> None:
        self.sample_rate = int(config.get("audio.sample_rate", 16000))
        self.channels = int(config.get("audio.channels", 1))
        self.device = None if str(config.get("audio.device", "default")) == "default" else str(config.get("audio.device"))
        self.blocksize = int(self.sample_rate * chunk_ms / 1000)

    def frames(self) -> Iterator[AudioFrame]:
        import sounddevice as sd

        with sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=self.blocksize,
            device=self.device,
            channels=self.channels,
            dtype="int16",
        ) as stream:
            while True:
                data, _overflowed = stream.read(self.blocksize)
                yield AudioFrame(pcm16=bytes(data), sample_rate=self.sample_rate, channels=self.channels)
