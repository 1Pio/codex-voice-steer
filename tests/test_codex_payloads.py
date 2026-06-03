from __future__ import annotations

import json
import threading
from types import SimpleNamespace
from typing import Any

import pytest

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


def test_app_server_rejects_unmanaged_mode(tmp_path) -> None:
    cfg = load_config(overrides={"codex": {"app_server": "external"}}, path=tmp_path / "missing.toml")
    bridge = CodexAppServer(cfg)

    with pytest.raises(ValueError, match="codex.app_server"):
        bridge._app_server_listen()


def test_permission_profile_is_used_for_app_server_config(tmp_path) -> None:
    cfg = load_config(overrides={"codex": {"permission_profile": ":read-only"}}, path=tmp_path / "missing.toml")
    bridge = CodexAppServer(cfg)
    assert bridge._thread_start_params()["config"]["default_permissions"] == ":read-only"
    assert bridge._thread_resume_params("thread_1")["config"]["default_permissions"] == ":read-only"


def test_legacy_permissions_alias_still_works(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[codex]\npermissions = ":danger-full-access"\n')
    bridge = CodexAppServer(load_config(path=path))
    assert bridge._thread_start_params()["config"]["default_permissions"] == ":danger-full-access"


def test_native_agent_mode_falls_back_to_injected_instructions(tmp_path) -> None:
    cfg = load_config(overrides={"instructions": {"mode": "native_agent"}}, path=tmp_path / "missing.toml")
    bridge = CodexAppServer(cfg)
    params = bridge._thread_start_params()
    assert "controlled through codex-voice-steer" in params["developerInstructions"]


def test_unknown_instruction_mode_disables_injected_instructions(tmp_path) -> None:
    cfg = load_config(overrides={"instructions": {"mode": "off"}}, path=tmp_path / "missing.toml")
    bridge = CodexAppServer(cfg)
    params = bridge._thread_start_params()
    assert params["developerInstructions"] == ""


def test_selected_bundled_agent_instructions_are_injected(tmp_path) -> None:
    cfg = load_config(overrides={"codex": {"agent": "cxv-voice-msd"}}, path=tmp_path / "missing.toml")
    bridge = CodexAppServer(cfg)
    params = bridge._thread_start_params()
    assert "msd say" in params["developerInstructions"]
    assert "controlled through codex-voice-steer" in params["developerInstructions"]


def test_text_input_contains_voice_metadata(tmp_path) -> None:
    cfg = load_config(path=tmp_path / "missing.toml")
    bridge = CodexAppServer(cfg)
    payload = bridge._text_input("check status")
    assert payload["type"] == "text"
    assert "wake=scarlett" in payload["text"]
    assert "check status" in payload["text"]


def test_text_input_honors_voice_metadata_toggles(tmp_path) -> None:
    cfg = load_config(
        overrides={
            "delivery": {
                "include_wake_word": False,
                "include_stt_diagnostics": True,
            }
        },
        path=tmp_path / "missing.toml",
    )
    bridge = CodexAppServer(cfg)
    payload = bridge._text_input("check status")
    assert "wake=" not in payload["text"]
    assert "stt=macparakeet mode=clean" in payload["text"]


def test_text_input_can_disable_voice_metadata(tmp_path) -> None:
    cfg = load_config(overrides={"delivery": {"include_voice_metadata": False}}, path=tmp_path / "missing.toml")
    bridge = CodexAppServer(cfg)
    assert bridge._text_input("check status")["text"] == "check status"


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


def test_agent_message_delta_accepts_current_item_notification(tmp_path) -> None:
    cfg = load_config(path=tmp_path / "missing.toml")
    store = StateStore(tmp_path / "state.json")
    bridge = CodexAppServer(cfg, state_store=store)

    bridge._handle_notification(
        "item/agentMessage/delta",
        {"threadId": "thread_1", "turnId": "turn_1", "itemId": "item_1", "delta": "working"},
    )

    events = store.load().events or []
    assert events[-1]["event"] == "codex_visible_delta"
    assert events[-1]["thread_id"] == "thread_1"
    assert events[-1]["turn_id"] == "turn_1"
    assert events[-1]["delta"] == "working"


def test_item_started_records_codex_tool_action(tmp_path) -> None:
    cfg = load_config(path=tmp_path / "missing.toml")
    store = StateStore(tmp_path / "state.json")
    bridge = CodexAppServer(cfg, state_store=store)

    bridge._handle_notification(
        "item/started",
        {
            "threadId": "thread_1",
            "turnId": "turn_1",
            "item": {"id": "item_1", "type": "commandExecution", "command": "git status --short"},
        },
    )

    events = store.load().events or []
    assert events[-1]["event"] == "codex_tool_started"
    assert events[-1]["summary"] == "command: git status --short"


def test_mcp_progress_records_codex_tool_progress(tmp_path) -> None:
    cfg = load_config(path=tmp_path / "missing.toml")
    store = StateStore(tmp_path / "state.json")
    bridge = CodexAppServer(cfg, state_store=store)

    bridge._handle_notification(
        "item/mcpToolCall/progress",
        {"threadId": "thread_1", "turnId": "turn_1", "itemId": "item_1", "message": "Working"},
    )

    events = store.load().events or []
    assert events[-1]["event"] == "codex_tool_progress"
    assert events[-1]["message"] == "Working"


def test_request_waits_on_condition_until_response_arrives(tmp_path) -> None:
    class FakeStdin:
        def __init__(self) -> None:
            self.lines: list[str] = []
            self.written = threading.Event()

        def write(self, line: str) -> None:
            self.lines.append(line)
            self.written.set()

        def flush(self) -> None:
            return None

    cfg = load_config(path=tmp_path / "missing.toml")
    bridge = CodexAppServer(cfg)
    stdin = FakeStdin()
    bridge.proc = SimpleNamespace(stdin=stdin)
    result: dict[str, Any] = {}

    thread = threading.Thread(target=lambda: result.update(bridge.request("ping", {"ok": True})))
    thread.start()
    assert stdin.written.wait(1.0)
    assert thread.is_alive()
    request_id = json.loads(stdin.lines[0])["id"]
    with bridge._condition:
        bridge._pending[request_id] = {"id": request_id, "result": {"pong": True}}
        bridge._condition.notify_all()
    thread.join(1.0)

    assert result == {"pong": True}


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


def test_deliver_text_queues_when_active_policy_is_queue(tmp_path) -> None:
    cfg = load_config(overrides={"delivery": {"when_active": "queue"}}, path=tmp_path / "missing.toml")
    store = StateStore(tmp_path / "state.json")
    store.update(thread_id="thread_1", active_turn_id="turn_active")
    bridge = FakeBridge(cfg, state_store=store)
    result = bridge.deliver_text("queue this")
    assert result.action == "queue"
    assert bridge.requests == []
    assert store.load().queued_inputs == ["queue this"]


def test_deliver_text_rejects_unsupported_idle_policy_before_thread_work(tmp_path) -> None:
    cfg = load_config(overrides={"delivery": {"when_idle": "queue"}}, path=tmp_path / "missing.toml")
    store = StateStore(tmp_path / "state.json")
    bridge = FakeBridge(cfg, state_store=store)

    with pytest.raises(ValueError, match="delivery.when_idle"):
        bridge.deliver_text("do not send")

    assert bridge.requests == []
    assert store.load().thread_id == ""


def test_deliver_text_rejects_unsupported_active_policy_before_thread_work(tmp_path) -> None:
    cfg = load_config(overrides={"delivery": {"when_active": "interrupt"}}, path=tmp_path / "missing.toml")
    store = StateStore(tmp_path / "state.json")
    bridge = FakeBridge(cfg, state_store=store)

    with pytest.raises(ValueError, match="delivery.when_active"):
        bridge.deliver_text("do not send")

    assert bridge.requests == []
    assert store.load().thread_id == ""


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
