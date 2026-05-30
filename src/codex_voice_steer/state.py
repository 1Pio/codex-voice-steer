from __future__ import annotations

import json
from dataclasses import dataclass
import time
from pathlib import Path
from threading import RLock
from typing import Any

from .paths import state_db_path


@dataclass
class CxvState:
    thread_id: str = ""
    session_id: str = ""
    cwd: str = "."
    active_turn_id: str = ""
    listening: bool = False
    queued_inputs: list[str] | None = None
    events: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "session_id": self.session_id,
            "cwd": self.cwd,
            "active_turn_id": self.active_turn_id,
            "listening": self.listening,
            "queued_inputs": self.queued_inputs or [],
            "events": self.events or [],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CxvState":
        return cls(
            thread_id=str(data.get("thread_id", "")),
            session_id=str(data.get("session_id", "")),
            cwd=str(data.get("cwd", ".")),
            active_turn_id=str(data.get("active_turn_id", "")),
            listening=bool(data.get("listening", False)),
            queued_inputs=list(data.get("queued_inputs", [])),
            events=list(data.get("events", [])),
        )


class StateStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or state_db_path()
        self._lock = RLock()

    def load(self) -> CxvState:
        with self._lock:
            if not self.path.exists():
                return CxvState()
            return CxvState.from_dict(json.loads(self.path.read_text()))

    def save(self, state: CxvState) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.path.with_name(self.path.name + ".tmp")
            tmp_path.write_text(json.dumps(state.to_dict(), indent=2, sort_keys=True) + "\n")
            tmp_path.replace(self.path)

    def update(self, **kwargs: Any) -> CxvState:
        with self._lock:
            state = self.load()
            for key, value in kwargs.items():
                setattr(state, key, value)
            self.save(state)
            return state

    def append_event(self, event: str, **fields: Any) -> CxvState:
        with self._lock:
            state = self.load()
            events = state.events or []
            events.append({"ts": time.time(), "event": event, **fields})
            state.events = events[-200:]
            self.save(state)
            return state
