from __future__ import annotations

import sys
import types
from importlib.machinery import ModuleSpec

from codex_voice_steer.config import load_config
from codex_voice_steer.wake import wake_readiness


def test_wake_readiness_rejects_unloadable_model(tmp_path, monkeypatch) -> None:
    model_path = tmp_path / "scarlett.onnx"
    model_path.write_bytes(b"not an onnx model")
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(f'[wake]\nmodel_path = "{model_path}"\n')

    openwakeword = types.ModuleType("openwakeword")
    openwakeword.__path__ = []
    openwakeword.__spec__ = ModuleSpec("openwakeword", loader=None, is_package=True)
    model_module = types.ModuleType("openwakeword.model")
    model_module.__spec__ = ModuleSpec("openwakeword.model", loader=None)

    class Model:
        def __init__(self, wakeword_models):
            raise RuntimeError("bad model")

    model_module.Model = Model
    monkeypatch.setitem(sys.modules, "openwakeword", openwakeword)
    monkeypatch.setitem(sys.modules, "openwakeword.model", model_module)

    readiness = wake_readiness(load_config(path=cfg_path), repo_root=tmp_path)
    assert readiness.ok is False
    assert "failed to load" in readiness.reason
