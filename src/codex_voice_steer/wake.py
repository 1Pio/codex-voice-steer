from __future__ import annotations

import importlib.util
import math
import time
import wave
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .audio import pcm16_level_stats


@dataclass(frozen=True)
class WakeReadiness:
    ok: bool
    reason: str
    model_path: Path


@dataclass(frozen=True)
class WakeAudioTest:
    wav_path: Path
    hit: bool
    max_score: float
    threshold: float
    frame_count: int
    sample_rate: int
    channels: int
    max_score_time_sec: float
    rms: float
    peak: int

    def to_dict(self) -> dict[str, object]:
        return {
            "wav_path": str(self.wav_path),
            "hit": self.hit,
            "max_score": self.max_score,
            "threshold": self.threshold,
            "frame_count": self.frame_count,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "max_score_time_sec": self.max_score_time_sec,
            "rms": self.rms,
            "peak": self.peak,
        }


def wake_readiness(config: Config, repo_root: Path | None = None) -> WakeReadiness:
    model_path = Path(str(config.get("wake.model_path", "models/wake/scarlett.onnx")))
    if not model_path.is_absolute():
        model_path = (repo_root or _default_repo_root()) / model_path
        if not model_path.exists():
            packaged = _packaged_wake_model_path()
            if packaged.exists():
                model_path = packaged
    if importlib.util.find_spec("openwakeword") is None:
        return WakeReadiness(False, "openwakeword Python package is not installed", model_path)
    if not model_path.exists():
        return WakeReadiness(False, f"custom wake model missing: {model_path}", model_path)
    try:
        from openwakeword.model import Model

        Model(wakeword_models=[str(model_path)], **_openwakeword_feature_kwargs())
    except Exception as exc:
        return WakeReadiness(False, f"custom wake model failed to load: {exc}", model_path)
    return WakeReadiness(True, "openwakeword and custom scarlett model are present", model_path)


def _default_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def _packaged_wake_model_path() -> Path:
    return Path(__file__).resolve().parent / "resources" / "wake" / "scarlett.onnx"


class OpenWakeWordDetector:
    def __init__(self, config: Config, repo_root: Path | None = None) -> None:
        readiness = wake_readiness(config, repo_root)
        if not readiness.ok:
            raise RuntimeError(readiness.reason)
        from openwakeword.model import Model

        self.word = str(config.get("wake.word", "scarlett"))
        self.sensitivity = float(config.get("wake.sensitivity", 0.5))
        self.refractory_sec = max(0.0, float(config.get("wake.refractory_ms", 1200)) / 1000)
        self.last_wake_monotonic = 0.0
        self.model = Model(wakeword_models=[str(readiness.model_path)], **_openwakeword_feature_kwargs())

    def predict(self, pcm16_frame) -> bool:
        score = self.score(pcm16_frame)
        if score < self.sensitivity:
            return False
        now = time.monotonic()
        if self.last_wake_monotonic and now - self.last_wake_monotonic < self.refractory_sec:
            return False
        self.last_wake_monotonic = now
        return True

    def score(self, pcm16_frame) -> float:
        if isinstance(pcm16_frame, bytes):
            import numpy as np

            pcm16_frame = np.frombuffer(pcm16_frame, dtype=np.int16)
        scores = self.model.predict(pcm16_frame)
        return float(scores.get(self.word, 0.0))


def _openwakeword_feature_kwargs() -> dict[str, str]:
    melspec = _packaged_openwakeword_model_path("melspectrogram.onnx")
    embedding = _packaged_openwakeword_model_path("embedding_model.onnx")
    if melspec.exists() and embedding.exists():
        return {
            "inference_framework": "onnx",
            "melspec_model_path": str(melspec),
            "embedding_model_path": str(embedding),
        }
    return {}


def _packaged_openwakeword_model_path(name: str) -> Path:
    return Path(__file__).resolve().parent / "resources" / "openwakeword" / "models" / name


def score_wake_audio(config: Config, wav_path: Path, threshold: float | None = None) -> WakeAudioTest:
    detector = OpenWakeWordDetector(config)
    threshold = detector.sensitivity if threshold is None else threshold
    max_score = 0.0
    max_score_time_sec = 0.0
    frame_count = 0
    level_samples = 0
    sum_squares = 0.0
    peak = 0
    target_samples = int(int(config.get("audio.sample_rate", 16000)) * 80 / 1000)

    with wave.open(str(wav_path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        if channels != 1 or sample_width != 2 or sample_rate != int(config.get("audio.sample_rate", 16000)):
            raise ValueError(
                "wake test audio must be 16 kHz mono PCM16 WAV "
                f"(got {sample_rate} Hz, {channels} channel(s), {sample_width * 8}-bit)"
            )
        while True:
            chunk = wav.readframes(target_samples)
            if not chunk:
                break
            if len(chunk) < target_samples * 2:
                chunk += b"\0" * (target_samples * 2 - len(chunk))
            stats = pcm16_level_stats(chunk)
            level_samples += int(stats["samples"])
            sum_squares += float(stats["sum_squares"])
            peak = max(peak, int(stats["peak"]))
            score = detector.score(chunk)
            if score > max_score:
                max_score = score
                max_score_time_sec = frame_count * target_samples / sample_rate
            frame_count += 1
    rms = math.sqrt(sum_squares / level_samples) if level_samples else 0.0

    return WakeAudioTest(
        wav_path=wav_path,
        hit=max_score >= threshold,
        max_score=max_score,
        threshold=threshold,
        frame_count=frame_count,
        sample_rate=sample_rate,
        channels=channels,
        max_score_time_sec=max_score_time_sec,
        rms=rms,
        peak=peak,
    )
