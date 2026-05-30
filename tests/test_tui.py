from __future__ import annotations

import types

from codex_voice_steer.config import load_config
from codex_voice_steer import tui
from codex_voice_steer.tui import _events_after, _last_event_ts, render_event, write_ui


def test_tui_renders_voice_events() -> None:
    assert render_event({"event": "wake_detected"}) == "wake detected"
    assert render_event({"event": "vad_final", "wav_path": "/tmp/cxv.wav"}) == "vad final: /tmp/cxv.wav"
    assert render_event({"event": "stt_final", "transcript": "check status now"}) == "user: check status now"
    assert render_event({"event": "sent", "action": "turn/start"}) == "sent: turn/start"


def test_tui_honors_visibility_toggles(tmp_path) -> None:
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("[ui]\nshow_wake_events = false\nshow_final_transcripts = false\n")
    cfg = load_config(path=cfg_path)
    assert render_event({"event": "wake_detected"}, cfg) == ""
    assert render_event({"event": "stt_final", "transcript": "hidden"}, cfg) == ""


def test_tui_jsonl_and_quiet_modes(capsys, tmp_path) -> None:
    jsonl_path = tmp_path / "jsonl.toml"
    jsonl_path.write_text('[ui]\nmode = "jsonl"\n')
    write_ui(load_config(path=jsonl_path), "listening", "listening")
    assert '"event": "listening"' in capsys.readouterr().out

    quiet_path = tmp_path / "quiet.toml"
    quiet_path.write_text('[ui]\nmode = "quiet"\n')
    write_ui(load_config(path=quiet_path), "listening", "listening")
    assert capsys.readouterr().out == ""


def test_foreground_preflight_uses_configured_audio_device(tmp_path, monkeypatch) -> None:
    seen = {}
    cfg = load_config(overrides={"audio": {"device": "Loopback Input"}}, path=tmp_path / "missing.toml")

    def fake_audio_readiness(config, probe_stream=False):
        seen["device"] = config.get("audio.device")
        seen["probe_stream"] = probe_stream
        return types.SimpleNamespace(ok=False, reason="blocked for test")

    monkeypatch.setattr(tui, "audio_readiness", fake_audio_readiness)
    monkeypatch.setattr(tui, "vad_readiness", lambda: types.SimpleNamespace(ok=True, reason="ok"))
    monkeypatch.setattr(tui, "wake_readiness", lambda _config: types.SimpleNamespace(ok=True, reason="ok"))

    assert tui.run_foreground_tui(cfg) == 2
    assert seen == {"device": "Loopback Input", "probe_stream": True}


def test_tui_event_cursor_survives_capped_history() -> None:
    before = [{"event": "old", "ts": float(index)} for index in range(200)]
    after = [{"event": "old", "ts": float(index)} for index in range(1, 200)]
    after.append({"event": "wake_detected", "ts": 200.0})

    cursor = _last_event_ts(before)
    assert cursor == 199.0
    assert _events_after(after, cursor) == [{"event": "wake_detected", "ts": 200.0}]
