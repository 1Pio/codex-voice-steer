from __future__ import annotations

from codex_voice_steer.config import load_config
from codex_voice_steer.tui import render_event, write_ui


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
