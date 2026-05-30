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
