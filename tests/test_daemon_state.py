from __future__ import annotations

import asyncio

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


def test_voice_delivery_config_queues_when_barge_in_is_disabled(tmp_path) -> None:
    config = load_config(overrides={"wake": {"allow_barge_in": False}}, path=tmp_path / "missing.toml")
    delivery = CxvDaemon._voice_delivery_config(config)
    assert delivery.get("delivery.when_active") == "queue"


def test_voice_delivery_config_preserves_active_policy_when_barge_in_is_enabled(tmp_path) -> None:
    config = load_config(overrides={"delivery": {"when_active": "steer"}}, path=tmp_path / "missing.toml")
    delivery = CxvDaemon._voice_delivery_config(config)
    assert delivery.get("delivery.when_active") == "steer"


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
