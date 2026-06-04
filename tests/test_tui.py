from __future__ import annotations

import asyncio
import types

from codex_voice_steer.config import load_config
from codex_voice_steer import tui
from codex_voice_steer.tui import DisplayState, _events_after, _last_event_ts, render_event, render_events, write_ui


def test_tui_renders_voice_events() -> None:
    assert render_event({"event": "wake_detected"}) == "wake detected"
    assert render_event({"event": "vad_final", "wav_path": "/tmp/cxv.wav"}) == "vad final: /tmp/cxv.wav"
    assert render_event({"event": "stt_final", "transcript": "check status now"}) == "user: check status now"
    assert render_event({"event": "sent", "action": "turn/start"}) == "sent: turn/start"


def test_tui_suppresses_duplicate_stt_final_when_user_final_is_present() -> None:
    rendered = render_events(
        [
            {"event": "stt_final", "transcript": " Scarlet. "},
            {"event": "user_final", "text": "Scarlet."},
            {"event": "sent", "action": "turn/start"},
        ]
    )

    assert [line for _event, line in rendered] == ["user: Scarlet.", "sent: turn/start"]


def test_tui_suppresses_duplicate_user_text_across_poll_batches() -> None:
    state = DisplayState()
    first = render_events([{"event": "stt_final", "transcript": " Scarlet. "}], display_state=state)
    second = render_events([{"event": "user_final", "text": "Scarlet."}], display_state=state)

    assert [line for _event, line in first] == ["user: Scarlet."]
    assert second == []


def test_tui_honors_visibility_toggles(tmp_path) -> None:
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("[ui]\nshow_wake_events = false\nshow_final_transcripts = false\n")
    cfg = load_config(path=cfg_path)
    assert render_event({"event": "wake_detected"}, cfg) == ""
    assert render_event({"event": "stt_final", "transcript": "hidden"}, cfg) == ""


def test_tui_renders_codex_tool_progress_by_default(tmp_path) -> None:
    cfg = load_config(path=tmp_path / "missing.toml")
    assert render_event({"event": "codex_tool_started", "summary": "command: git status"}, cfg) == "codex action: command: git status"
    assert render_event({"event": "codex_tool_progress", "message": "Downloading"}, cfg) == ""


def test_tui_renders_auto_compaction_status_by_default(tmp_path) -> None:
    cfg = load_config(path=tmp_path / "missing.toml")

    assert (
        render_event({"event": "auto_compact_started", "usage_ratio": 0.57}, cfg)
        == "automatically compacting context (57.0%)"
    )
    assert render_event({"event": "auto_compact_completed"}, cfg) == "compacted context"


def test_tui_can_show_and_hide_auto_compaction_status_events(tmp_path) -> None:
    cfg = load_config(
        overrides={
            "ui": {
                "visible_events": ["auto_compact_started", "auto_compact_completed"],
                "hidden_events": ["auto_compact_completed"],
            }
        },
        path=tmp_path / "missing.toml",
    )

    rendered = render_events(
        [
            {"event": "auto_compact_started"},
            {"event": "auto_compact_completed"},
            {"event": "wake_detected"},
        ],
        cfg,
    )

    assert [line for _event, line in rendered] == ["automatically compacting context"]


def test_tui_can_hide_codex_tool_traces(tmp_path) -> None:
    cfg = load_config(overrides={"ui": {"show_codex_tool_traces": False}}, path=tmp_path / "missing.toml")
    assert render_event({"event": "codex_tool_started", "summary": "command: git status"}, cfg) == ""


def test_tui_renders_codex_msd_separately_from_tool_traces(tmp_path) -> None:
    cfg = load_config(overrides={"ui": {"show_codex_tool_traces": False}}, path=tmp_path / "missing.toml")
    assert (
        render_event({"event": "codex_msd_started", "summary": "command: /bin/zsh -lc 'msd say --text hello'", "msd_args": "--text hello"}, cfg)
        == "\x1b[1mcodex msd:\x1b[0m --text hello"
    )


def test_tui_extracts_msd_say_args_from_legacy_summary(tmp_path) -> None:
    cfg = load_config(path=tmp_path / "missing.toml")
    assert (
        render_event({"event": "codex_msd_started", "summary": "command: /bin/zsh -lc \"msd say --text 'Yes.' --instruct fast\""}, cfg)
        == "\x1b[1mcodex msd:\x1b[0m --text 'Yes.' --instruct fast"
    )


def test_tui_renders_codex_final_answer_with_answer_limit(tmp_path) -> None:
    cfg = load_config(overrides={"ui": {"max_codex_answer_lines": 2}}, path=tmp_path / "missing.toml")
    rendered = render_event({"event": "codex_final_answer", "text": "one\ntwo\nthree"}, cfg)
    assert rendered == "\x1b[1mcodex:\x1b[0m one\ntwo\n... truncated 1 line(s)"


def test_tui_filters_visible_and_hidden_events(tmp_path) -> None:
    cfg = load_config(
        overrides={"ui": {"visible_events": ["wake_detected", "codex_msd_started"], "hidden_events": ["wake_detected"]}},
        path=tmp_path / "missing.toml",
    )
    rendered = render_events(
        [
            {"event": "wake_detected"},
            {"event": "sent", "action": "turn/start"},
            {"event": "codex_msd_started", "summary": "command: msd say hello", "msd_args": "hello"},
        ],
        cfg,
    )
    assert [line for _event, line in rendered] == ["\x1b[1mcodex msd:\x1b[0m hello"]


def test_event_line_can_dim_timestamp_with_opacity() -> None:
    line = tui.event_line("wake detected", timestamp_opacity=0.45)
    assert line.startswith("\x1b[38;2;115;115;115m")
    assert "\x1b[0m  wake detected" in line


def test_event_line_can_dim_secondary_status_after_timestamp() -> None:
    line = tui.event_line("sent: turn/start", timestamp_opacity=0.45, secondary_status_opacity=0.7)
    assert line.startswith("\x1b[38;2;115;115;115m")
    assert "\x1b[0m  \x1b[38;2;178;178;178msent: turn/start\x1b[0m" in line


def test_event_line_keeps_primary_status_at_normal_opacity() -> None:
    assert tui.event_line("user: hello", timestamps=False, secondary_status_opacity=0.7) == "user: hello"
    assert tui.event_line("\x1b[1mcodex msd:\x1b[0m --text hello", timestamps=False, secondary_status_opacity=0.7) == "\x1b[1mcodex msd:\x1b[0m --text hello"
    assert tui.event_line("\x1b[1mcodex:\x1b[0m done", timestamps=False, secondary_status_opacity=0.7) == "\x1b[1mcodex:\x1b[0m done"


def test_event_line_dims_secondary_status_without_timestamps() -> None:
    assert tui.event_line("sent: turn/start", timestamps=False, secondary_status_opacity=0.7) == "\x1b[38;2;178;178;178msent: turn/start\x1b[0m"


def test_tui_can_disable_bold_labels(tmp_path) -> None:
    cfg = load_config(overrides={"ui": {"bold_labels": False}}, path=tmp_path / "missing.toml")
    assert render_event({"event": "user_final", "text": "plain"}, cfg) == "user: plain"


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


def test_foreground_listener_handles_daemon_loss(tmp_path, monkeypatch, capsys) -> None:
    cfg = load_config(overrides={"ui": {"mode": "jsonl"}}, path=tmp_path / "missing.toml")
    calls = []

    async def fake_ensure_daemon(_config):
        return None

    async def fake_send_request(_config, payload):
        calls.append(payload["command"])
        if calls == ["status"]:
            return {"ok": True, "state": {"events": []}}
        if calls == ["status", "listen"]:
            return {"ok": True}
        raise FileNotFoundError("daemon socket disappeared")

    monkeypatch.setattr(tui, "ensure_daemon", fake_ensure_daemon)
    monkeypatch.setattr(tui, "send_request", fake_send_request)

    result = asyncio.run(tui._run_foreground_listener(cfg, poll_interval=0, max_polls=None))

    assert result == 1
    assert '"event": "daemon_lost"' in capsys.readouterr().out


def test_foreground_listener_sends_listen_overrides(tmp_path, monkeypatch) -> None:
    cfg = load_config(path=tmp_path / "missing.toml")
    payloads = []

    async def fake_ensure_daemon(_config):
        return None

    async def fake_send_request(_config, payload):
        payloads.append(payload)
        if payload["command"] == "status":
            return {"ok": True, "state": {"events": []}}
        return {"ok": True}

    monkeypatch.setattr(tui, "ensure_daemon", fake_ensure_daemon)
    monkeypatch.setattr(tui, "send_request", fake_send_request)

    result = asyncio.run(
        tui._run_foreground_listener(
            cfg,
            poll_interval=0,
            max_polls=1,
            listen_overrides={"codex": {"fast": True, "effort": "minimal"}},
        )
    )

    assert result == 0
    assert payloads[1] == {"command": "listen", "overrides": {"codex": {"fast": True, "effort": "minimal"}}}


def test_foreground_ctrl_c_stops_background_daemon(tmp_path, monkeypatch, capsys) -> None:
    cfg = load_config(overrides={"server": {"state_db": str(tmp_path / "state.json")}}, path=tmp_path / "missing.toml")
    stopped = {}

    def fake_audio_readiness(_config, probe_stream=False):
        return types.SimpleNamespace(ok=True, reason="ok")

    def fake_asyncio_run(coro):
        coro.close()
        raise KeyboardInterrupt

    monkeypatch.setattr(tui, "audio_readiness", fake_audio_readiness)
    monkeypatch.setattr(tui, "vad_readiness", lambda: types.SimpleNamespace(ok=True, reason="ok"))
    monkeypatch.setattr(tui, "wake_readiness", lambda _config: types.SimpleNamespace(ok=True, reason="ok"))
    monkeypatch.setattr(tui.asyncio, "run", fake_asyncio_run)
    monkeypatch.setattr(tui, "stop_background", lambda _config: stopped.setdefault("called", True))
    monkeypatch.setattr(tui, "send_request", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("pause should not be sent")))

    assert tui.run_foreground_tui(cfg) == 0
    assert stopped["called"] is True
    out = capsys.readouterr().out
    assert "session: -" in out
    assert "resume: new" in out
    assert "stopped" in out
