from __future__ import annotations

import importlib.util
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .config import Config
from .segment import AudioFrame


@dataclass(frozen=True)
class AudioReadiness:
    ok: bool
    reason: str


@dataclass(frozen=True)
class AudioDevice:
    index: int
    name: str
    max_input_channels: int
    is_default: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "name": self.name,
            "max_input_channels": self.max_input_channels,
            "default": self.is_default,
        }


def audio_readiness(config: Config | None = None) -> AudioReadiness:
    if importlib.util.find_spec("sounddevice") is None:
        return AudioReadiness(False, "sounddevice is not installed, so microphone capture is unavailable")
    configured = str(config.get("audio.device", "default")) if config is not None else "default"
    device_arg = _device_arg(configured)
    try:
        import sounddevice as sd

        device = sd.query_devices(device=device_arg, kind="input") if device_arg is not None else sd.query_devices(kind="input")
    except Exception as exc:
        label = "default" if device_arg is None else configured
        return AudioReadiness(False, f"microphone input {label!r} is unavailable: {exc}")
    label = "default" if device_arg is None else configured
    return AudioReadiness(True, f"microphone input {label!r} available: {device.get('name', 'unknown')}")


def list_input_devices() -> list[AudioDevice]:
    if importlib.util.find_spec("sounddevice") is None:
        raise RuntimeError("sounddevice is not installed, so audio devices cannot be listed")
    import sounddevice as sd

    default_input = _default_input_device_index(sd)
    devices = []
    for index, device in enumerate(sd.query_devices()):
        max_input_channels = int(device.get("max_input_channels", 0))
        if max_input_channels <= 0:
            continue
        devices.append(
            AudioDevice(
                index=index,
                name=str(device.get("name", "unknown")),
                max_input_channels=max_input_channels,
                is_default=index == default_input,
            )
        )
    return devices


def _device_arg(configured: str):
    if configured == "default":
        return None
    try:
        return int(configured)
    except ValueError:
        return configured


def _default_input_device_index(sd) -> int | None:
    try:
        default = sd.default.device
        if isinstance(default, (tuple, list)) and default:
            value = default[0]
        else:
            value = default
        return None if value is None else int(value)
    except Exception:
        return None


class MicCapture:
    def __init__(self, config: Config, chunk_ms: int = 80) -> None:
        self.sample_rate = int(config.get("audio.sample_rate", 16000))
        self.channels = int(config.get("audio.channels", 1))
        self.device = _device_arg(str(config.get("audio.device", "default")))
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


def wav_frames(config: Config, wav_path: Path, chunk_ms: int = 80) -> Iterator[AudioFrame]:
    sample_rate = int(config.get("audio.sample_rate", 16000))
    channels = int(config.get("audio.channels", 1))
    blocksize = int(sample_rate * chunk_ms / 1000)
    with wave.open(str(wav_path), "rb") as wav:
        wav_channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        wav_rate = wav.getframerate()
        if wav_channels != channels or sample_width != 2 or wav_rate != sample_rate:
            raise ValueError(
                "voice test audio must be 16 kHz mono PCM16 WAV "
                f"(got {wav_rate} Hz, {wav_channels} channel(s), {sample_width * 8}-bit)"
            )
        while True:
            data = wav.readframes(blocksize)
            if not data:
                break
            expected = blocksize * channels * 2
            if len(data) < expected:
                data += b"\0" * (expected - len(data))
            yield AudioFrame(pcm16=data, sample_rate=sample_rate, channels=channels)
