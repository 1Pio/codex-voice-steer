from __future__ import annotations

import sys
import struct
import types
import wave

import pytest

from codex_voice_steer import audio
from codex_voice_steer.audio import apply_gain_pcm16, MicCapture, audio_readiness, input_levels, list_input_devices, play_and_record_input_wav, record_input_wav
from codex_voice_steer.config import load_config
from codex_voice_steer.segment import AudioFrame


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


def test_audio_readiness_can_probe_configured_stream(tmp_path, monkeypatch) -> None:
    seen = {}
    fake_sd = types.ModuleType("sounddevice")

    class RawInputStream:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def __enter__(self):
            seen["opened"] = True
            return self

        def __exit__(self, *_exc):
            seen["closed"] = True

    fake_sd.RawInputStream = RawInputStream
    fake_sd.query_devices = lambda device=None, kind=None: {"name": "Loopback Input"}
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    monkeypatch.setattr(audio.importlib.util, "find_spec", lambda _name: object())

    cfg = load_config(overrides={"audio": {"device": "2"}}, path=tmp_path / "missing.toml")
    result = audio_readiness(cfg, probe_stream=True)
    assert result.ok is True
    assert seen["device"] == 2
    assert seen["samplerate"] == 16000
    assert seen["channels"] == 1
    assert seen["opened"] is True
    assert seen["closed"] is True


def test_audio_readiness_reports_stream_open_failure(tmp_path, monkeypatch) -> None:
    fake_sd = types.ModuleType("sounddevice")

    class RawInputStream:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            raise RuntimeError("permission denied")

        def __exit__(self, *_exc):
            pass

    fake_sd.RawInputStream = RawInputStream
    fake_sd.query_devices = lambda device=None, kind=None: {"name": "MacBook Pro Microphone"}
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    monkeypatch.setattr(audio.importlib.util, "find_spec", lambda _name: object())

    cfg = load_config(path=tmp_path / "missing.toml")
    result = audio_readiness(cfg, probe_stream=True)
    assert result.ok is False
    assert "cannot be opened" in result.reason
    assert "permission denied" in result.reason


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


def test_list_input_devices_marks_implicit_default_by_name(monkeypatch) -> None:
    fake_sd = types.ModuleType("sounddevice")
    fake_sd.default = types.SimpleNamespace(device=(-1, -1))

    def query_devices(device=None, kind=None):
        if kind == "input":
            return {"name": "MacBook Pro Microphone"}
        return [
            {"name": "MacBook Pro Microphone", "max_input_channels": 1},
            {"name": "ZoomAudioDevice", "max_input_channels": 2},
        ]

    fake_sd.query_devices = query_devices
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    monkeypatch.setattr(audio.importlib.util, "find_spec", lambda _name: object())

    devices = list_input_devices()
    assert devices[0].is_default is True
    assert devices[1].is_default is False


def test_record_input_wav_writes_fixed_duration_capture(tmp_path, monkeypatch) -> None:
    class FakeMicCapture:
        def __init__(self, config):
            self.config = config

        def frames(self):
            yield AudioFrame(pcm16=b"\1\0" * 1280, sample_rate=16000, channels=1)
            yield AudioFrame(pcm16=b"\2\0" * 1280, sample_rate=16000, channels=1)

    monkeypatch.setattr(audio, "MicCapture", FakeMicCapture)
    cfg = load_config(overrides={"audio": {"device": "Loopback Input"}}, path=tmp_path / "missing.toml")
    wav_path = tmp_path / "capture.wav"

    result = record_input_wav(cfg, wav_path, seconds=0.12)

    assert result.wav_path == wav_path
    assert result.samples == 1920
    assert result.device == "Loopback Input"
    assert result.rms == pytest.approx(2**0.5)
    assert result.peak == 2
    assert result.gain_db == 0.0
    assert result.clipped_samples == 0
    assert result.clipped_ratio == 0.0
    assert result.to_dict()["rms"] == pytest.approx(2**0.5)
    with wave.open(str(wav_path), "rb") as wav:
        assert wav.getframerate() == 16000
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getnframes() == 1920


def test_play_and_record_input_wav_captures_loopback_route(tmp_path, monkeypatch) -> None:
    source_path = tmp_path / "source.wav"
    with wave.open(str(source_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\1\0" * 1280)

    seen = {"writes": []}
    fake_sd = types.ModuleType("sounddevice")

    class RawInputStream:
        def __init__(self, **kwargs):
            seen["input"] = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            pass

        def read(self, samples):
            return b"\2\0" * samples, False

    class RawOutputStream:
        def __init__(self, **kwargs):
            seen["output"] = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            pass

        def write(self, data):
            seen["writes"].append(data)

    fake_sd.RawInputStream = RawInputStream
    fake_sd.RawOutputStream = RawOutputStream
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)

    cfg = load_config(overrides={"audio": {"device": "2"}}, path=tmp_path / "missing.toml")
    result = play_and_record_input_wav(cfg, source_path, tmp_path / "captured.wav", output_device="3")

    assert seen["input"]["device"] == 2
    assert seen["output"]["device"] == 3
    assert seen["writes"] == [b"\1\0" * 1280]
    assert result.samples == 1280
    assert result.peak == 2
    assert result.rms == pytest.approx(2.0)
    assert result.source_wav_path == source_path
    assert result.output_device == "3"
    assert result.to_dict()["source_wav_path"] == str(source_path)
    with wave.open(str(result.wav_path), "rb") as wav:
        assert wav.getframerate() == 16000
        assert wav.getnframes() == 1280


def test_input_levels_reports_windowed_rms_and_peak(tmp_path, monkeypatch) -> None:
    class FakeMicCapture:
        def __init__(self, config):
            self.config = config

        def frames(self):
            yield AudioFrame(pcm16=b"\1\0" * 1280, sample_rate=16000, channels=1)
            yield AudioFrame(pcm16=b"\2\0" * 1280, sample_rate=16000, channels=1)
            yield AudioFrame(pcm16=b"\3\0" * 1280, sample_rate=16000, channels=1)

    monkeypatch.setattr(audio, "MicCapture", FakeMicCapture)
    cfg = load_config(overrides={"audio": {"device": "0"}}, path=tmp_path / "missing.toml")

    levels = list(input_levels(cfg, seconds=0.24, interval_ms=160))

    assert len(levels) == 2
    assert levels[0].elapsed_sec == pytest.approx(0.16)
    assert levels[0].rms == pytest.approx(2.5**0.5)
    assert levels[0].peak == 2
    assert levels[0].device == "0"
    assert levels[0].gain_db == 0.0
    assert levels[0].clipped_samples == 0
    assert levels[0].clipped_ratio == 0.0
    assert levels[1].elapsed_sec == pytest.approx(0.24)
    assert levels[1].rms == pytest.approx(3)
    assert levels[1].peak == 3


def test_apply_gain_pcm16_amplifies_and_reports_clipping() -> None:
    result = apply_gain_pcm16(struct.pack("<hh", 20000, -20000), 6.020599913279624)
    assert result["pcm16"] == struct.pack("<hh", 32767, -32768)
    assert result["clipped_samples"] == 2
    assert result["clipped_ratio"] == 1.0
