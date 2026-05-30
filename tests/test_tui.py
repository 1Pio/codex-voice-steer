from __future__ import annotations

from codex_voice_steer.tui import render_event


def test_tui_renders_voice_events() -> None:
    assert render_event({"event": "wake_detected"}) == "wake detected"
    assert render_event({"event": "vad_final", "wav_path": "/tmp/cxv.wav"}) == "vad final: /tmp/cxv.wav"
    assert render_event({"event": "stt_final", "transcript": "check status now"}) == "user: check status now"
    assert render_event({"event": "sent", "action": "turn/start"}) == "sent: turn/start"
