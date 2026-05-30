from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any

from .config import Config


@dataclass(frozen=True)
class VadReadiness:
    ok: bool
    reason: str


def vad_readiness() -> VadReadiness:
    if importlib.util.find_spec("torch") is None:
        return VadReadiness(False, "torch is not installed, so Silero VAD cannot be loaded")
    if importlib.util.find_spec("silero_vad") is None:
        return VadReadiness(False, "silero-vad is not installed")
    return VadReadiness(True, "silero-vad package is present")


class SileroVad:
    def __init__(self, config: Config) -> None:
        from silero_vad import load_silero_vad

        self.sample_rate = int(config.get("audio.sample_rate", 16000))
        self.threshold = float(config.get("vad.speech_threshold", 0.5))
        self.model = load_silero_vad()

    def speech_timestamps(self, pcm16: bytes) -> list[dict[str, Any]]:
        import numpy as np
        from silero_vad import get_speech_timestamps

        audio = np.frombuffer(pcm16, dtype=np.int16).astype("float32") / 32768.0
        return list(get_speech_timestamps(audio, self.model, sampling_rate=self.sample_rate, threshold=self.threshold))
