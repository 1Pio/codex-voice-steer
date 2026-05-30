from __future__ import annotations

import importlib.util
from dataclasses import dataclass


@dataclass(frozen=True)
class AudioReadiness:
    ok: bool
    reason: str


def audio_readiness() -> AudioReadiness:
    if importlib.util.find_spec("sounddevice") is None:
        return AudioReadiness(False, "sounddevice is not installed, so microphone capture is unavailable")
    return AudioReadiness(True, "sounddevice is present; microphone capture adapter can open the default input")
