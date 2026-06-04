from __future__ import annotations

import asyncio
import os
import threading
import time
from types import SimpleNamespace

from codex_voice_steer import daemon
from codex_voice_steer.config import load_config
from codex_voice_steer.daemon import CxvDaemon, stop_background
from codex_voice_steer.state import StateStore


class FakeCodexBridge:
    def __init__(self, store: StateStore) -> None:
        self.store = store
        self.interrupted = False

    def interrupt(self):
        self.interrupted = True
        self.store.update(active_turn_id="")
        return SimpleNamespace(action="turn/interrupt", thread_id="thread_old", turn_id="turn_active")

    def start_new_thread(self, config):
        return self.store.update(
            thread_id="thread_new",
            session_id="session_new",
            cwd=str(config.get("codex.cwd", ".")),
            active_turn_id="",
            queued_inputs=[],
        )


def test_stop_clears_volatile_state_when_already_stopped(tmp_path) -> None:
    state_path = tmp_path / "state.json"
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        f"""
[server]
socket_path = "{tmp_path / 'cxv.sock'}"
pid_path = "{tmp_path / 'cxv.pid'}"
state_db = "{state_path}"
"""
    )
    store = StateStore(state_path)
    store.update(active_turn_id="turn_1", listening=True)
    config = load_config(path=cfg_path)
    assert stop_background(config) is False
    state = store.load()
    assert state.active_turn_id == ""
    assert state.listening is False


def test_stop_cleans_stale_serve_processes_when_already_stopped(tmp_path, monkeypatch) -> None:
    called = {}
    state_path = tmp_path / "state.json"
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        f"""
[server]
socket_path = "{tmp_path / 'cxv.sock'}"
pid_path = "{tmp_path / 'cxv.pid'}"
state_db = "{state_path}"
"""
    )
    monkeypatch.setattr(daemon, "_terminate_stale_serve_processes", lambda: called.setdefault("cleanup", True))

    assert stop_background(load_config(path=cfg_path)) is False
    assert called["cleanup"] is True


def test_stale_serve_pids_parses_pgrep_output(monkeypatch) -> None:
    current = os.getpid()

    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(returncode=0, stdout=f"{current}\n123\nnot-a-pid\n456\n")

    monkeypatch.setattr(daemon.subprocess, "run", fake_run)

    assert daemon._stale_serve_pids(exclude={current, 456}) == [123]


def test_voice_delivery_config_queues_when_barge_in_is_disabled(tmp_path) -> None:
    config = load_config(overrides={"wake": {"allow_barge_in": False}}, path=tmp_path / "missing.toml")
    delivery = CxvDaemon._voice_delivery_config(config)
    assert delivery.get("delivery.when_active") == "queue"


def test_voice_delivery_config_preserves_active_policy_when_barge_in_is_enabled(tmp_path) -> None:
    config = load_config(overrides={"delivery": {"when_active": "steer"}}, path=tmp_path / "missing.toml")
    delivery = CxvDaemon._voice_delivery_config(config)
    assert delivery.get("delivery.when_active") == "steer"


def test_daemon_records_listen_and_pause_events(tmp_path, monkeypatch) -> None:
    config = load_config(
        overrides={
            "server": {
                "socket_path": str(tmp_path / "cxv.sock"),
                "pid_path": str(tmp_path / "cxv.pid"),
                "state_db": str(tmp_path / "state.json"),
            },
            "audio": {"device": "MacBook Pro Microphone"},
        },
        path=tmp_path / "missing.toml",
    )
    cxv = CxvDaemon(config)
    monkeypatch.setattr(cxv, "_listen_blockers", lambda: [])
    monkeypatch.setattr(cxv, "_listen_loop", lambda *_args: asyncio.sleep(0))

    async def run() -> list[str]:
        await cxv._dispatch({"command": "listen"})
        await cxv._dispatch({"command": "pause"})
        return [str(event["event"]) for event in cxv.state_store.load().events or []]

    assert asyncio.run(run())[-2:] == ["listening_started", "listening_paused"]


def test_daemon_session_new_refuses_pinned_config_thread(tmp_path) -> None:
    config = load_config(
        overrides={"server": {"state_db": str(tmp_path / "state.json")}, "codex": {"thread_id": "thread_config"}},
        path=tmp_path / "missing.toml",
    )
    cxv = CxvDaemon(config)

    response = asyncio.run(cxv._dispatch({"command": "session-new"}))

    assert response["ok"] is False
    assert response["configured_thread_id"] == "thread_config"
    assert "codex.thread_id" in response["error"]


def test_daemon_session_new_refuses_active_turn_without_force(tmp_path) -> None:
    config = load_config(overrides={"server": {"state_db": str(tmp_path / "state.json")}}, path=tmp_path / "missing.toml")
    cxv = CxvDaemon(config)
    cxv.state_store.update(thread_id="thread_old", session_id="session_old", active_turn_id="turn_active")

    response = asyncio.run(cxv._dispatch({"command": "session-new"}))

    assert response["ok"] is False
    assert response["active_turn_id"] == "turn_active"
    assert cxv.state_store.load().thread_id == "thread_old"


def test_daemon_session_new_force_interrupts_and_saves_new_thread(tmp_path) -> None:
    config = load_config(
        overrides={"server": {"state_db": str(tmp_path / "state.json")}, "codex": {"cwd": "/tmp/new-cwd"}},
        path=tmp_path / "missing.toml",
    )
    cxv = CxvDaemon(config)
    fake = FakeCodexBridge(cxv.state_store)
    cxv.codex = fake  # type: ignore[assignment]
    cxv.state_store.update(thread_id="thread_old", session_id="session_old", active_turn_id="turn_active", queued_inputs=["old"])

    response = asyncio.run(cxv._dispatch({"command": "session-new", "force": True}))

    assert response["ok"] is True
    assert fake.interrupted is True
    state = cxv.state_store.load()
    assert state.thread_id == "thread_new"
    assert state.session_id == "session_new"
    assert state.active_turn_id == ""
    assert state.queued_inputs == []


def test_daemon_pause_stops_listener_worker_cooperatively(tmp_path, monkeypatch) -> None:
    config = load_config(
        overrides={
            "server": {
                "socket_path": str(tmp_path / "cxv.sock"),
                "pid_path": str(tmp_path / "cxv.pid"),
                "state_db": str(tmp_path / "state.json"),
            },
        },
        path=tmp_path / "missing.toml",
    )
    cxv = CxvDaemon(config)
    started = threading.Event()
    stopped = threading.Event()
    monkeypatch.setattr(cxv, "_listen_blockers", lambda: [])

    def fake_run_voice_turn(stop_event, _overrides):
        started.set()
        while not stop_event.is_set():
            time.sleep(0.001)
        stopped.set()
        return SimpleNamespace(status="no_input", transcript="", wav_path="", reason="stopped")

    monkeypatch.setattr(cxv, "_run_voice_turn", fake_run_voice_turn)

    async def run() -> list[str]:
        await cxv._dispatch({"command": "listen"})
        assert await asyncio.to_thread(started.wait, 1.0)
        await cxv._dispatch({"command": "pause"})
        assert await asyncio.to_thread(stopped.wait, 1.0)
        return [str(event["event"]) for event in cxv.state_store.load().events or []]

    events = asyncio.run(run())
    assert events[-1] == "listening_paused"
    assert "voice_turn" not in events
    assert cxv.listen_task is None
    assert cxv.listen_stop_event is None


def test_daemon_refuses_new_listener_while_previous_worker_is_stopping(tmp_path, monkeypatch) -> None:
    config = load_config(overrides={"server": {"state_db": str(tmp_path / "state.json")}}, path=tmp_path / "missing.toml")
    cxv = CxvDaemon(config)
    stop_event = threading.Event()

    async def never_done() -> None:
        await asyncio.sleep(3600)

    async def run() -> dict:
        cxv.listen_task = asyncio.create_task(never_done())
        cxv.listen_stop_event = stop_event
        stop_event.set()
        monkeypatch.setattr(cxv, "_listen_blockers", lambda: [])
        try:
            return await cxv._dispatch({"command": "listen"})
        finally:
            cxv.listen_task.cancel()

    response = asyncio.run(run())
    assert response["ok"] is False
    assert "still stopping" in response["blockers"][0]
    assert cxv.state_store.load().listening is False


def test_daemon_reuses_live_wake_detector_for_same_config(tmp_path, monkeypatch) -> None:
    created = []

    class FakeWakeDetector:
        def __init__(self, config):
            created.append(config.get("wake.refractory_ms"))

    monkeypatch.setattr(daemon, "OpenWakeWordDetector", FakeWakeDetector)
    config = load_config(path=tmp_path / "missing.toml")
    cxv = CxvDaemon(config)

    first = cxv._wake_detector(config)
    second = cxv._wake_detector(config)

    assert first is second
    assert created == [1200]


def test_daemon_recreates_live_wake_detector_when_refractory_changes(tmp_path, monkeypatch) -> None:
    created = []

    class FakeWakeDetector:
        def __init__(self, config):
            self.refractory_ms = config.get("wake.refractory_ms")
            created.append(self.refractory_ms)

    monkeypatch.setattr(daemon, "OpenWakeWordDetector", FakeWakeDetector)
    config = load_config(path=tmp_path / "missing.toml")
    cxv = CxvDaemon(config)

    first = cxv._wake_detector(config)
    second = cxv._wake_detector(config.with_overrides({"wake": {"refractory_ms": 2400}}))

    assert first is not second
    assert created == [1200, 2400]


def test_daemon_idle_timeout_is_disabled_by_default(tmp_path) -> None:
    config = load_config(overrides={"server": {"state_db": str(tmp_path / "state.json")}}, path=tmp_path / "missing.toml")
    cxv = CxvDaemon(config)
    cxv.last_activity_monotonic = 100.0

    assert cxv._should_exit_for_idle(100000.0) is False


def test_daemon_idle_timeout_waits_for_no_active_work(tmp_path) -> None:
    config = load_config(
        overrides={"server": {"state_db": str(tmp_path / "state.json"), "idle_timeout_minutes": 1}},
        path=tmp_path / "missing.toml",
    )
    cxv = CxvDaemon(config)
    cxv.last_activity_monotonic = 100.0

    cxv.state_store.update(listening=True)
    assert cxv._should_exit_for_idle(200.0) is False

    cxv.state_store.update(listening=False, active_turn_id="turn_1")
    assert cxv._should_exit_for_idle(300.0) is False

    cxv.state_store.update(active_turn_id="", queued_inputs=["queued"])
    assert cxv._should_exit_for_idle(400.0) is False

    cxv.state_store.update(queued_inputs=[])
    assert cxv._should_exit_for_idle(cxv.last_activity_monotonic + 61.0) is True


def test_idle_monitor_records_timeout_and_sets_stop_event(tmp_path) -> None:
    async def run() -> list[dict]:
        config = load_config(
            overrides={"server": {"state_db": str(tmp_path / "state.json"), "idle_timeout_minutes": 1}},
            path=tmp_path / "missing.toml",
        )
        cxv = CxvDaemon(config)
        cxv._idle_timeout_sec = lambda: 0.001  # type: ignore[method-assign]
        cxv.last_activity_monotonic = daemon.time.monotonic() - 1
        stop_event = asyncio.Event()

        await asyncio.wait_for(cxv._idle_monitor(stop_event, poll_interval=0.001), timeout=0.1)

        assert stop_event.is_set()
        return cxv.state_store.load().events or []

    events = asyncio.run(run())
    assert events[-1]["event"] == "daemon_idle_timeout"
    assert events[-1]["idle_timeout_minutes"] == 1.0


def _token_usage_params(total_tokens: int = 560, context_window: int = 1000, thread_id: str = "thread_1", turn_id: str = "turn_1") -> dict:
    return {
        "threadId": thread_id,
        "turnId": turn_id,
        "tokenUsage": {
            "total": {
                "totalTokens": total_tokens,
                "inputTokens": total_tokens,
                "cachedInputTokens": 0,
                "outputTokens": 0,
                "reasoningOutputTokens": 0,
            },
            "last": {
                "totalTokens": 0,
                "inputTokens": 0,
                "cachedInputTokens": 0,
                "outputTokens": 0,
                "reasoningOutputTokens": 0,
            },
            "modelContextWindow": context_window,
        },
    }


def test_auto_compact_waits_for_completed_turn_and_threshold(tmp_path) -> None:
    compacted: list[str] = []

    async def run() -> list[dict]:
        config = load_config(
            overrides={"server": {"state_db": str(tmp_path / "state.json")}, "auto_compact": {"idle_delay_sec": 0.001, "cooldown_sec": 0}},
            path=tmp_path / "missing.toml",
        )
        cxv = CxvDaemon(config)
        cxv.codex = SimpleNamespace(compact_thread=lambda thread_id: compacted.append(thread_id))
        cxv.state_store.update(thread_id="thread_1", active_turn_id="turn_1")

        cxv._handle_codex_event("thread/tokenUsage/updated", _token_usage_params(total_tokens=560))
        assert cxv.auto_compact_task is None

        cxv.state_store.update(active_turn_id="")
        cxv._handle_codex_event("turn/completed", {"threadId": "thread_1", "turn": {"id": "turn_1"}})
        await asyncio.sleep(0.02)
        return cxv.state_store.load().events or []

    events = asyncio.run(run())
    assert compacted == ["thread_1"]
    assert [event["event"] for event in events[-2:]] == ["auto_compact_scheduled", "auto_compact_started"]


def test_auto_compact_ignores_usage_below_threshold(tmp_path) -> None:
    async def run() -> tuple[object, list[dict]]:
        config = load_config(
            overrides={"server": {"state_db": str(tmp_path / "state.json")}, "auto_compact": {"idle_delay_sec": 0.001}},
            path=tmp_path / "missing.toml",
        )
        cxv = CxvDaemon(config)
        cxv.state_store.update(thread_id="thread_1", active_turn_id="")
        cxv._handle_codex_event("thread/tokenUsage/updated", _token_usage_params(total_tokens=540))
        cxv._handle_codex_event("turn/completed", {"threadId": "thread_1", "turn": {"id": "turn_1"}})
        await asyncio.sleep(0.01)
        return cxv.auto_compact_task, cxv.state_store.load().events or []

    task, events = asyncio.run(run())
    assert task is None
    assert "auto_compact_scheduled" not in [event["event"] for event in events]


def test_auto_compact_cancels_when_bound_thread_starts_new_turn(tmp_path) -> None:
    compacted: list[str] = []

    async def run() -> list[dict]:
        config = load_config(
            overrides={"server": {"state_db": str(tmp_path / "state.json")}, "auto_compact": {"idle_delay_sec": 0.05, "cooldown_sec": 0}},
            path=tmp_path / "missing.toml",
        )
        cxv = CxvDaemon(config)
        cxv.codex = SimpleNamespace(compact_thread=lambda thread_id: compacted.append(thread_id))
        cxv.state_store.update(thread_id="thread_1", active_turn_id="")
        cxv._handle_codex_event("turn/completed", {"threadId": "thread_1", "turn": {"id": "turn_1"}})
        cxv._handle_codex_event("thread/tokenUsage/updated", _token_usage_params(total_tokens=560))
        cxv._handle_codex_event("turn/started", {"threadId": "thread_1", "turn": {"id": "turn_2"}})
        await asyncio.sleep(0.08)
        return cxv.state_store.load().events or []

    events = asyncio.run(run())
    assert compacted == []
    assert events[-1]["event"] == "auto_compact_cancelled"
    assert events[-1]["source"] == "turn_started"


def test_auto_compact_rechecks_queued_input_before_compacting(tmp_path) -> None:
    compacted: list[str] = []

    async def run() -> list[dict]:
        config = load_config(
            overrides={"server": {"state_db": str(tmp_path / "state.json")}, "auto_compact": {"idle_delay_sec": 0.001, "cooldown_sec": 0}},
            path=tmp_path / "missing.toml",
        )
        cxv = CxvDaemon(config)
        cxv.codex = SimpleNamespace(compact_thread=lambda thread_id: compacted.append(thread_id))
        cxv.state_store.update(thread_id="thread_1", active_turn_id="")
        cxv._handle_codex_event("turn/completed", {"threadId": "thread_1", "turn": {"id": "turn_1"}})
        cxv._handle_codex_event("thread/tokenUsage/updated", _token_usage_params(total_tokens=560))
        cxv.state_store.update(queued_inputs=["next"])
        await asyncio.sleep(0.02)
        return cxv.state_store.load().events or []

    events = asyncio.run(run())
    assert compacted == []
    assert events[-1]["event"] == "auto_compact_skipped"
    assert events[-1]["reason"] == "queued_input"


def test_auto_compact_honors_cooldown(tmp_path) -> None:
    config = load_config(
        overrides={"server": {"state_db": str(tmp_path / "state.json")}, "auto_compact": {"cooldown_sec": 300}},
        path=tmp_path / "missing.toml",
    )
    cxv = CxvDaemon(config)
    cxv.state_store.update(thread_id="thread_1", active_turn_id="")
    cxv.last_auto_compact_monotonic["thread_1"] = daemon.time.monotonic()
    cxv._handle_codex_event("turn/completed", {"threadId": "thread_1", "turn": {"id": "turn_1"}})
    cxv._handle_codex_event("thread/tokenUsage/updated", _token_usage_params(total_tokens=560))

    events = cxv.state_store.load().events or []
    assert cxv.auto_compact_task is None
    assert events[-1]["event"] == "auto_compact_skipped"
    assert events[-1]["reason"] == "cooldown"


def test_auto_compact_runtime_overrides_are_applied(tmp_path) -> None:
    config = load_config(overrides={"server": {"state_db": str(tmp_path / "state.json")}}, path=tmp_path / "missing.toml")
    cxv = CxvDaemon(config)
    cxv._apply_runtime_auto_compact_overrides({"overrides": {"auto_compact": {"enabled": False, "idle_delay_sec": 12}}})

    effective = cxv._auto_compact_config()
    assert effective.get("auto_compact.enabled") is False
    assert effective.get("auto_compact.idle_delay_sec") == 12
