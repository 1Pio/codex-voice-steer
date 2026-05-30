from __future__ import annotations

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
