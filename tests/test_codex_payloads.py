from __future__ import annotations

from codex_voice_steer.codex_app_server import CodexAppServer
from codex_voice_steer.config import load_config
from codex_voice_steer.state import StateStore


def test_thread_start_injects_developer_instructions(tmp_path) -> None:
    cfg = load_config(path=tmp_path / "missing.toml")
    bridge = CodexAppServer(cfg)
    params = bridge._thread_start_params()
    assert params["developerInstructions"]
    assert params["approvalPolicy"] == "on-request"
    assert params["approvalsReviewer"] == "auto_review"
    assert params["config"]["default_permissions"] == ":workspace"


def test_text_input_contains_voice_metadata(tmp_path) -> None:
    cfg = load_config(path=tmp_path / "missing.toml")
    bridge = CodexAppServer(cfg)
    payload = bridge._text_input("check status")
    assert payload["type"] == "text"
    assert "wake=scarlett" in payload["text"]
    assert "check status" in payload["text"]


def test_turn_started_notification_does_not_clear_active_turn_when_id_missing(tmp_path) -> None:
    cfg = load_config(path=tmp_path / "missing.toml")
    store = StateStore(tmp_path / "state.json")
    store.update(active_turn_id="turn_from_request")
    bridge = CodexAppServer(cfg, state_store=store)
    bridge._handle_notification("turn/started", {})
    assert store.load().active_turn_id == "turn_from_request"


def test_turn_notifications_accept_nested_turn_ids(tmp_path) -> None:
    cfg = load_config(path=tmp_path / "missing.toml")
    store = StateStore(tmp_path / "state.json")
    bridge = CodexAppServer(cfg, state_store=store)
    bridge._handle_notification("turn/started", {"turn": {"id": "turn_nested"}})
    assert store.load().active_turn_id == "turn_nested"
    bridge._handle_notification("turn/completed", {"turn": {"id": "turn_nested"}})
    events = store.load().events or []
    assert events[-1]["turn_id"] == "turn_nested"
    assert store.load().active_turn_id == ""
