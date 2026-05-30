from __future__ import annotations

from typing import Any

from codex_voice_steer.codex_app_server import CodexAppServer, JsonRpcError
from codex_voice_steer.config import load_config
from codex_voice_steer.state import StateStore


class FakeBridge(CodexAppServer):
    def __init__(self, *args, fail_steer: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fail_steer = fail_steer
        self.requests: list[tuple[str, dict[str, Any]]] = []

    def ensure_thread(self, config=None) -> str:
        state = self.state_store.load()
        thread_id = state.thread_id or "thread_1"
        self.state_store.update(thread_id=thread_id)
        return thread_id

    def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self.requests.append((method, params))
        if method == "turn/steer" and self.fail_steer:
            raise JsonRpcError("not steerable")
        return {"turn": {"id": params.get("expectedTurnId", "turn_2")}}


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


def test_stale_turn_completed_notification_does_not_clear_new_active_turn(tmp_path) -> None:
    cfg = load_config(path=tmp_path / "missing.toml")
    store = StateStore(tmp_path / "state.json")
    store.update(active_turn_id="turn_new")
    bridge = CodexAppServer(cfg, state_store=store)
    bridge._handle_notification("turn/completed", {"turn": {"id": "turn_old"}})
    assert store.load().active_turn_id == "turn_new"


def test_deliver_text_steers_active_turn_with_expected_turn_id(tmp_path) -> None:
    cfg = load_config(path=tmp_path / "missing.toml")
    store = StateStore(tmp_path / "state.json")
    store.update(thread_id="thread_1", active_turn_id="turn_active")
    bridge = FakeBridge(cfg, state_store=store)
    result = bridge.deliver_text("add this", force_steer=True)
    method, params = bridge.requests[-1]
    assert method == "turn/steer"
    assert params["expectedTurnId"] == "turn_active"
    assert result.action == "turn/steer"


def test_deliver_text_queues_when_active_turn_is_not_steerable(tmp_path) -> None:
    cfg = load_config(path=tmp_path / "missing.toml")
    store = StateStore(tmp_path / "state.json")
    store.update(thread_id="thread_1", active_turn_id="turn_active")
    bridge = FakeBridge(cfg, state_store=store, fail_steer=True)
    result = bridge.deliver_text("queue this", force_steer=True)
    assert result.action == "queue"
    assert store.load().queued_inputs == ["queue this"]


def test_deliver_text_uses_invocation_config_overrides(tmp_path) -> None:
    cfg = load_config(path=tmp_path / "missing.toml")
    override = cfg.with_overrides({"codex": {"cwd": str(tmp_path), "model": "gpt-test"}})
    store = StateStore(tmp_path / "state.json")
    bridge = FakeBridge(cfg, state_store=store)
    result = bridge.deliver_text("use override", config=override)
    method, params = bridge.requests[-1]
    assert result.action == "turn/start"
    assert method == "turn/start"
    assert params["cwd"] == str(tmp_path.resolve())
    assert params["model"] == "gpt-test"


def test_interrupt_sends_active_turn_id(tmp_path) -> None:
    cfg = load_config(path=tmp_path / "missing.toml")
    store = StateStore(tmp_path / "state.json")
    store.update(thread_id="thread_1", active_turn_id="turn_active")
    bridge = FakeBridge(cfg, state_store=store)
    result = bridge.interrupt()
    method, params = bridge.requests[-1]
    assert method == "turn/interrupt"
    assert params == {"threadId": "thread_1", "turnId": "turn_active"}
    assert result.action == "turn/interrupt"
    assert store.load().active_turn_id == ""
    events = store.load().events or []
    assert events[-1]["action"] == "turn/interrupt"


def test_interrupt_noop_is_recorded(tmp_path) -> None:
    cfg = load_config(path=tmp_path / "missing.toml")
    store = StateStore(tmp_path / "state.json")
    bridge = FakeBridge(cfg, state_store=store)
    result = bridge.interrupt()
    assert result.action == "noop"
    events = store.load().events or []
    assert events[-1]["action"] == "turn/interrupt"
    assert events[-1]["noop"] is True
