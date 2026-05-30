from __future__ import annotations

from codex_voice_steer.state import StateStore


def test_state_store_keeps_recent_events(tmp_path) -> None:
    store = StateStore(tmp_path / "state.json")
    for i in range(205):
        store.append_event("event", n=i)
    state = store.load()
    assert state.events is not None
    assert len(state.events) == 200
    assert state.events[0]["n"] == 5


def test_state_store_save_uses_atomic_replace(tmp_path) -> None:
    path = tmp_path / "state.json"
    store = StateStore(path)
    store.update(active_turn_id="turn_1")
    assert store.load().active_turn_id == "turn_1"
    assert not path.with_name(path.name + ".tmp").exists()
