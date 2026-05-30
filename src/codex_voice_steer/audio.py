from __future__ import annotations

import importlib.util
import math
import struct
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


@dataclass(frozen=True)
class AudioRecordResult:
    wav_path: Path
    sample_rate: int
    channels: int
    samples: int
    seconds: float
    device: str
    rms: float
    peak: int

    def to_dict(self) -> dict[str, object]:
        return {
            "wav_path": str(self.wav_path),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "samples": self.samples,
            "seconds": self.seconds,
            "device": self.device,
            "rms": self.rms,
            "peak": self.peak,
        }


@dataclass(frozen=True)
class AudioLevel:
    elapsed_sec: float
    rms: float
    peak: int
    samples: int
    device: str

    def to_dict(self) -> dict[str, object]:
        return {
            "elapsed_sec": self.elapsed_sec,
            "rms": self.rms,
            "peak": self.peak,
            "samples": self.samples,
            "device": self.device,
        }


def audio_readiness(config: Config | None = None, probe_stream: bool = False) -> AudioReadiness:
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
    if probe_stream:
        try:
            _probe_input_stream(config, device_arg)
        except Exception as exc:
            return AudioReadiness(False, f"microphone input {label!r} cannot be opened: {exc}")
    return AudioReadiness(True, f"microphone input {label!r} available: {device.get('name', 'unknown')}")


def list_input_devices() -> list[AudioDevice]:
    if importlib.util.find_spec("sounddevice") is None:
        raise RuntimeError("sounddevice is not installed, so audio devices cannot be listed")
    import sounddevice as sd

    default_input = _default_input_device_index(sd)
    default_input_name = _default_input_device_name(sd) if default_input is None else ""
    devices = []
    for index, device in enumerate(sd.query_devices()):
        max_input_channels = int(device.get("max_input_channels", 0))
        if max_input_channels <= 0:
            continue
        name = str(device.get("name", "unknown"))
        devices.append(
            AudioDevice(
                index=index,
                name=name,
                max_input_channels=max_input_channels,
                is_default=index == default_input or bool(default_input_name and name == default_input_name),
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
        index = None if value is None else int(value)
        return index if index is not None and index >= 0 else None
    except Exception:
        return None


def _default_input_device_name(sd) -> str:
    try:
        device = sd.query_devices(kind="input")
        return str(device.get("name", ""))
    except Exception:
        return ""


def _probe_input_stream(config: Config | None, device) -> None:
    import sounddevice as sd

    sample_rate = int(config.get("audio.sample_rate", 16000)) if config is not None else 16000
    channels = int(config.get("audio.channels", 1)) if config is not None else 1
    blocksize = int(sample_rate * 80 / 1000)
    with sd.RawInputStream(
        samplerate=sample_rate,
        blocksize=blocksize,
        device=device,
        channels=channels,
        dtype="int16",
    ):
        return


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


def record_input_wav(config: Config, wav_path: Path, seconds: float) -> AudioRecordResult:
    if seconds <= 0:
        raise ValueError("record duration must be greater than zero")
    sample_rate = int(config.get("audio.sample_rate", 16000))
    channels = int(config.get("audio.channels", 1))
    target_samples = int(sample_rate * seconds)
    captured_samples = 0
    level_samples = 0
    sum_squares = 0.0
    peak = 0
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    capture = MicCapture(config)
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for frame in capture.frames():
            remaining = target_samples - captured_samples
            if remaining <= 0:
                break
            take_samples = min(frame.samples, remaining)
            chunk = frame.pcm16[: take_samples * channels * 2]
            stats = pcm16_level_stats(chunk)
            level_samples += int(stats["samples"])
            sum_squares += stats["sum_squares"]
            peak = max(peak, int(stats["peak"]))
            wav.writeframes(chunk)
            captured_samples += take_samples
    rms = math.sqrt(sum_squares / level_samples) if level_samples else 0.0
    return AudioRecordResult(
        wav_path=wav_path,
        sample_rate=sample_rate,
        channels=channels,
        samples=captured_samples,
        seconds=captured_samples / sample_rate,
        device=str(config.get("audio.device", "default")),
        rms=rms,
        peak=peak,
    )


def input_levels(config: Config, seconds: float, interval_ms: int = 500) -> Iterator[AudioLevel]:
    if seconds <= 0:
        raise ValueError("meter duration must be greater than zero")
    if interval_ms <= 0:
        raise ValueError("meter interval must be greater than zero")
    sample_rate = int(config.get("audio.sample_rate", 16000))
    channels = int(config.get("audio.channels", 1))
    target_samples = int(sample_rate * seconds)
    interval_samples = max(1, int(sample_rate * interval_ms / 1000))
    captured_samples = 0
    window_samples = 0
    window_level_samples = 0
    sum_squares = 0.0
    peak = 0
    device = str(config.get("audio.device", "default"))
    for frame in MicCapture(config).frames():
        remaining = target_samples - captured_samples
        if remaining <= 0:
            break
        take_samples = min(frame.samples, remaining)
        chunk = frame.pcm16[: take_samples * channels * 2]
        stats = pcm16_level_stats(chunk)
        window_level_samples += int(stats["samples"])
        window_samples += take_samples
        sum_squares += float(stats["sum_squares"])
        peak = max(peak, int(stats["peak"]))
        captured_samples += take_samples
        if window_samples >= interval_samples or captured_samples >= target_samples:
            rms = math.sqrt(sum_squares / window_level_samples) if window_level_samples else 0.0
            yield AudioLevel(
                elapsed_sec=captured_samples / sample_rate,
                rms=rms,
                peak=peak,
                samples=window_samples,
                device=device,
            )
            window_samples = 0
            window_level_samples = 0
            sum_squares = 0.0
            peak = 0


def pcm16_level_stats(pcm16: bytes) -> dict[str, float | int]:
    sample_count = 0
    sum_squares = 0.0
    peak = 0
    for (sample,) in struct.iter_unpack("<h", pcm16[: len(pcm16) - (len(pcm16) % 2)]):
        value = int(sample)
        magnitude = abs(value)
        peak = max(peak, magnitude)
        sum_squares += float(value * value)
        sample_count += 1
    rms = math.sqrt(sum_squares / sample_count) if sample_count else 0.0
    return {"samples": sample_count, "sum_squares": sum_squares, "rms": rms, "peak": peak}


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
