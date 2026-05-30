from __future__ import annotations

from codex_voice_steer.config import load_config
from codex_voice_steer.daemon import stop_background
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
