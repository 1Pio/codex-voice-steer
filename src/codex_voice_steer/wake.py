from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

from .config import Config


@dataclass(frozen=True)
class WakeReadiness:
    ok: bool
    reason: str
    model_path: Path


def wake_readiness(config: Config, repo_root: Path | None = None) -> WakeReadiness:
    model_path = Path(str(config.get("wake.model_path", "models/wake/scarlett.onnx")))
    if not model_path.is_absolute():
        model_path = (repo_root or _default_repo_root()) / model_path
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


class OpenWakeWordDetector:
    def __init__(self, config: Config, repo_root: Path | None = None) -> None:
        readiness = wake_readiness(config, repo_root)
        if not readiness.ok:
            raise RuntimeError(readiness.reason)
        from openwakeword.model import Model

        self.word = str(config.get("wake.word", "scarlett"))
        self.sensitivity = float(config.get("wake.sensitivity", 0.55))
        self.model = Model(wakeword_models=[str(readiness.model_path)])

    def predict(self, pcm16_frame) -> bool:
        scores = self.model.predict(pcm16_frame)
        score = float(scores.get(self.word, 0.0))
        return score >= self.sensitivity
