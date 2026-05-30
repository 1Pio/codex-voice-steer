from __future__ import annotations

import sys
import types
import wave
from importlib.machinery import ModuleSpec

from codex_voice_steer.config import load_config
from codex_voice_steer import wake
from codex_voice_steer.wake import OpenWakeWordDetector, _openwakeword_feature_kwargs, score_wake_audio, wake_readiness


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
        def __init__(self, wakeword_models, **_kwargs):
            raise RuntimeError("bad model")

    model_module.Model = Model
    monkeypatch.setitem(sys.modules, "openwakeword", openwakeword)
    monkeypatch.setitem(sys.modules, "openwakeword.model", model_module)

    readiness = wake_readiness(load_config(path=cfg_path), repo_root=tmp_path)
    assert readiness.ok is False
    assert "failed to load" in readiness.reason


def test_wake_readiness_rejects_disabled_wake_in_v1(tmp_path) -> None:
    model_path = tmp_path / "scarlett.onnx"
    model_path.write_bytes(b"fake")
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(f'[wake]\nmodel_path = "{model_path}"\nenabled = false\n')

    readiness = wake_readiness(load_config(path=cfg_path), repo_root=tmp_path)

    assert readiness.ok is False
    assert "unsupported in V1" in readiness.reason


def test_wake_detector_converts_pcm_bytes_to_int16_array(tmp_path, monkeypatch) -> None:
    model_path = tmp_path / "scarlett.onnx"
    model_path.write_bytes(b"fake")
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(f'[wake]\nmodel_path = "{model_path}"\n')

    openwakeword = types.ModuleType("openwakeword")
    openwakeword.__path__ = []
    openwakeword.__spec__ = ModuleSpec("openwakeword", loader=None, is_package=True)
    model_module = types.ModuleType("openwakeword.model")
    model_module.__spec__ = ModuleSpec("openwakeword.model", loader=None)
    seen = {}

    class Model:
        def __init__(self, wakeword_models, **kwargs):
            seen["kwargs"] = kwargs

        def predict(self, frame):
            seen["type"] = type(frame).__name__
            seen["dtype"] = str(frame.dtype)
            seen["shape"] = frame.shape
            return {"scarlett": 0.99}

    model_module.Model = Model
    monkeypatch.setitem(sys.modules, "openwakeword", openwakeword)
    monkeypatch.setitem(sys.modules, "openwakeword.model", model_module)

    detector = OpenWakeWordDetector(load_config(path=cfg_path), repo_root=tmp_path)
    assert detector.predict(b"\x01\x00\x02\x00") is True
    assert seen["type"] == "ndarray"
    assert seen["dtype"] == "int16"
    assert seen["shape"] == (2,)
    assert seen["kwargs"]["inference_framework"] == "onnx"
    assert seen["kwargs"]["melspec_model_path"].endswith("melspectrogram.onnx")
    assert seen["kwargs"]["embedding_model_path"].endswith("embedding_model.onnx")


def test_wake_detector_honors_refractory_window(tmp_path, monkeypatch) -> None:
    model_path = tmp_path / "scarlett.onnx"
    model_path.write_bytes(b"fake")
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(f'[wake]\nmodel_path = "{model_path}"\nsensitivity = 0.5\nrefractory_ms = 1200\n')

    openwakeword = types.ModuleType("openwakeword")
    openwakeword.__path__ = []
    openwakeword.__spec__ = ModuleSpec("openwakeword", loader=None, is_package=True)
    model_module = types.ModuleType("openwakeword.model")
    model_module.__spec__ = ModuleSpec("openwakeword.model", loader=None)

    class Model:
        def __init__(self, wakeword_models, **_kwargs):
            pass

        def predict(self, frame):
            return {"scarlett": 0.99}

    times = iter([10.0, 10.5, 11.3])

    model_module.Model = Model
    monkeypatch.setitem(sys.modules, "openwakeword", openwakeword)
    monkeypatch.setitem(sys.modules, "openwakeword.model", model_module)
    monkeypatch.setattr(wake.time, "monotonic", lambda: next(times))

    detector = OpenWakeWordDetector(load_config(path=cfg_path), repo_root=tmp_path)
    assert detector.predict(b"\0" * 1280 * 2) is True
    assert detector.predict(b"\0" * 1280 * 2) is False
    assert detector.predict(b"\0" * 1280 * 2) is True


def test_wake_readiness_falls_back_to_packaged_model(tmp_path, monkeypatch) -> None:
    packaged = tmp_path / "packaged.onnx"
    packaged.write_bytes(b"fake")
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text('[wake]\nmodel_path = "models/wake/scarlett.onnx"\n')

    openwakeword = types.ModuleType("openwakeword")
    openwakeword.__path__ = []
    openwakeword.__spec__ = ModuleSpec("openwakeword", loader=None, is_package=True)
    model_module = types.ModuleType("openwakeword.model")
    model_module.__spec__ = ModuleSpec("openwakeword.model", loader=None)

    class Model:
        def __init__(self, wakeword_models, **_kwargs):
            assert wakeword_models == [str(packaged)]

    model_module.Model = Model
    monkeypatch.setitem(sys.modules, "openwakeword", openwakeword)
    monkeypatch.setitem(sys.modules, "openwakeword.model", model_module)
    monkeypatch.setattr(wake, "_packaged_wake_model_path", lambda: packaged)

    readiness = wake_readiness(load_config(path=cfg_path), repo_root=tmp_path)
    assert readiness.ok is True
    assert readiness.model_path == packaged


def test_wake_audio_scores_wav_frames(tmp_path, monkeypatch) -> None:
    model_path = tmp_path / "scarlett.onnx"
    model_path.write_bytes(b"fake")
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(f'[wake]\nmodel_path = "{model_path}"\nsensitivity = 0.55\n')
    wav_path = tmp_path / "input.wav"
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\0" * 1280 * 2)
        wav.writeframes(b"\1\0" * 1280)

    openwakeword = types.ModuleType("openwakeword")
    openwakeword.__path__ = []
    openwakeword.__spec__ = ModuleSpec("openwakeword", loader=None, is_package=True)
    model_module = types.ModuleType("openwakeword.model")
    model_module.__spec__ = ModuleSpec("openwakeword.model", loader=None)

    class Model:
        calls = 0

        def __init__(self, wakeword_models, **_kwargs):
            pass

        def predict(self, frame):
            self.__class__.calls += 1
            return {"scarlett": 0.4 if self.__class__.calls == 1 else 0.7}

    model_module.Model = Model
    monkeypatch.setitem(sys.modules, "openwakeword", openwakeword)
    monkeypatch.setitem(sys.modules, "openwakeword.model", model_module)

    result = score_wake_audio(load_config(path=cfg_path), wav_path)
    assert result.hit is True
    assert result.max_score == 0.7
    assert result.frame_count == 2
    assert result.max_score_time_sec == 0.08
    assert result.rms > 0
    assert result.peak == 1
    assert result.to_dict()["max_score_time_sec"] == 0.08


def test_openwakeword_feature_kwargs_use_packaged_onnx_resources() -> None:
    kwargs = _openwakeword_feature_kwargs()
    assert kwargs["inference_framework"] == "onnx"
    assert kwargs["melspec_model_path"].endswith("melspectrogram.onnx")
    assert kwargs["embedding_model_path"].endswith("embedding_model.onnx")
