from __future__ import annotations

import sys
import types

from codex_voice_steer import audio
from codex_voice_steer.audio import MicCapture, audio_readiness, list_input_devices
from codex_voice_steer.config import load_config


def test_audio_readiness_checks_configured_device_name(tmp_path, monkeypatch) -> None:
    seen = {}
    fake_sd = types.ModuleType("sounddevice")

    def query_devices(device=None, kind=None):
        seen["device"] = device
        seen["kind"] = kind
        return {"name": "Loopback Input"}

    fake_sd.query_devices = query_devices
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    monkeypatch.setattr(audio.importlib.util, "find_spec", lambda _name: object())

    cfg = load_config(overrides={"audio": {"device": "Loopback Input"}}, path=tmp_path / "missing.toml")
    result = audio_readiness(cfg)
    assert result.ok is True
    assert seen == {"device": "Loopback Input", "kind": "input"}
    assert "Loopback Input" in result.reason


def test_audio_readiness_checks_configured_device_index(tmp_path, monkeypatch) -> None:
    seen = {}
    fake_sd = types.ModuleType("sounddevice")

    def query_devices(device=None, kind=None):
        seen["device"] = device
        seen["kind"] = kind
        return {"name": "MacBook Pro Microphone"}

    fake_sd.query_devices = query_devices
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    monkeypatch.setattr(audio.importlib.util, "find_spec", lambda _name: object())

    cfg = load_config(overrides={"audio": {"device": "3"}}, path=tmp_path / "missing.toml")
    result = audio_readiness(cfg)
    assert result.ok is True
    assert seen == {"device": 3, "kind": "input"}


def test_mic_capture_uses_configured_numeric_device(tmp_path) -> None:
    cfg = load_config(overrides={"audio": {"device": "3"}}, path=tmp_path / "missing.toml")
    capture = MicCapture(cfg)
    assert capture.device == 3


def test_list_input_devices_filters_outputs_and_marks_default(monkeypatch) -> None:
    default = types.SimpleNamespace(device=(2, 0))
    fake_sd = types.ModuleType("sounddevice")
    fake_sd.default = default
    fake_sd.query_devices = lambda: [
        {"name": "Built-in Output", "max_input_channels": 0},
        {"name": "Loopback Input", "max_input_channels": 2},
        {"name": "MacBook Pro Microphone", "max_input_channels": 1},
    ]
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    monkeypatch.setattr(audio.importlib.util, "find_spec", lambda _name: object())

    devices = list_input_devices()
    assert [device.name for device in devices] == ["Loopback Input", "MacBook Pro Microphone"]
    assert [device.index for device in devices] == [1, 2]
    assert devices[1].is_default is True
