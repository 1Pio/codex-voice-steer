from __future__ import annotations

from codex_voice_steer.config import load_config
from codex_voice_steer.session import render_session_header, render_session_status, session_status_info
from codex_voice_steer.state import StateStore


def test_session_status_uses_saved_thread_when_config_is_unpinned(tmp_path) -> None:
    state_path = tmp_path / "state.json"
    config = load_config(overrides={"server": {"state_db": str(state_path)}}, path=tmp_path / "missing.toml")
    StateStore(state_path).update(thread_id="thread_saved", session_id="session_saved", cwd="/tmp/cxv")

    info = session_status_info(config)

    assert info["effective_resume_thread_id"] == "thread_saved"
    assert info["effective_resume_source"] == "saved state"
    assert "resume saved state thread" in info["behavior"]
    assert "saved session: session_saved" in render_session_status(info)
    assert render_session_header(config) == "session: session_saved     resume: thread_saved"


def test_session_status_config_thread_overrides_saved_state(tmp_path) -> None:
    state_path = tmp_path / "state.json"
    config = load_config(
        overrides={"server": {"state_db": str(state_path)}, "codex": {"thread_id": "thread_config"}},
        path=tmp_path / "missing.toml",
    )
    StateStore(state_path).update(thread_id="thread_saved", session_id="session_saved")

    info = session_status_info(config)

    assert info["effective_resume_thread_id"] == "thread_config"
    assert info["effective_resume_source"] == "codex.thread_id"
    assert "resume codex.thread_id" in info["behavior"]


def test_session_status_reports_new_thread_when_no_saved_state(tmp_path) -> None:
    config = load_config(overrides={"server": {"state_db": str(tmp_path / "state.json")}}, path=tmp_path / "missing.toml")

    info = session_status_info(config)

    assert info["effective_resume_thread_id"] == ""
    assert info["effective_resume_source"] == "new thread"
    assert info["behavior"].startswith("start a new thread on next send")
