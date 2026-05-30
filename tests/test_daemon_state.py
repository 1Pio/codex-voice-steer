from __future__ import annotations

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
