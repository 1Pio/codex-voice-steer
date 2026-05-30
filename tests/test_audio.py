from __future__ import annotations

import sys
import types

from codex_voice_steer import audio
from codex_voice_steer.audio import audio_readiness
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
