from __future__ import annotations

from codex_voice_steer.config import load_config
from codex_voice_steer.wake import wake_readiness


def test_wake_readiness_uses_packaged_scarlett_model_when_repo_model_is_missing(tmp_path) -> None:
    cfg = load_config(path=tmp_path / "missing.toml")
    readiness = wake_readiness(cfg, repo_root=tmp_path)
    assert readiness.model_path.name == "scarlett.onnx"
    assert "resources/wake" in readiness.model_path.as_posix()
