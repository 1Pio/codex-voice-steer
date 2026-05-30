from __future__ import annotations

from codex_voice_steer.config import load_config
from codex_voice_steer.wake import wake_readiness


def test_wake_readiness_points_to_required_scarlett_model(tmp_path) -> None:
    cfg = load_config(path=tmp_path / "missing.toml")
    readiness = wake_readiness(cfg, repo_root=tmp_path)
    assert readiness.ok is False
    assert readiness.model_path.name == "scarlett.onnx"
