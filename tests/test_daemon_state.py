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
