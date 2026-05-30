from __future__ import annotations

import importlib.util
import wave
from dataclasses import dataclass
from pathlib import Path

from .config import Config


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

    def to_dict(self) -> dict[str, object]:
        return {
            "wav_path": str(self.wav_path),
            "hit": self.hit,
            "max_score": self.max_score,
            "threshold": self.threshold,
            "frame_count": self.frame_count,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
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

        Model(wakeword_models=[str(model_path)])
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
        self.model = Model(wakeword_models=[str(readiness.model_path)])

    def predict(self, pcm16_frame) -> bool:
        score = self.score(pcm16_frame)
        return score >= self.sensitivity

    def score(self, pcm16_frame) -> float:
        if isinstance(pcm16_frame, bytes):
            import numpy as np

            pcm16_frame = np.frombuffer(pcm16_frame, dtype=np.int16)
        scores = self.model.predict(pcm16_frame)
        return float(scores.get(self.word, 0.0))


def score_wake_audio(config: Config, wav_path: Path, threshold: float | None = None) -> WakeAudioTest:
    detector = OpenWakeWordDetector(config)
    threshold = detector.sensitivity if threshold is None else threshold
    max_score = 0.0
    frame_count = 0
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
            frame_count += 1
            if len(chunk) < target_samples * 2:
                chunk += b"\0" * (target_samples * 2 - len(chunk))
            max_score = max(max_score, detector.score(chunk))

    return WakeAudioTest(
        wav_path=wav_path,
        hit=max_score >= threshold,
        max_score=max_score,
        threshold=threshold,
        frame_count=frame_count,
        sample_rate=sample_rate,
        channels=channels,
    )
