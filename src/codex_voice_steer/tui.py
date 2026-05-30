from __future__ import annotations

import sys
import time

from .config import Config
from .audio import audio_readiness
from .vad import vad_readiness
from .wake import wake_readiness


def event_line(message: str, timestamps: bool = True) -> str:
    if not timestamps:
        return message
    return f"{time.strftime('%H:%M:%S')}  {message}"


def run_foreground_tui(config: Config) -> int:
    print("cxv 0.1.0  codex-voice-steer")
    print(f"wake: {config.get('wake.word', 'scarlett')}     stt: {config.get('stt.engine', 'macparakeet')} {config.get('stt.mode', 'clean')}     codex: app-server")
    blockers = []
    for readiness in (audio_readiness(), vad_readiness(), wake_readiness(config)):
        if not readiness.ok:
            blockers.append(readiness.reason)
    if blockers:
        for blocker in blockers:
            print(event_line(f"blocked: {blocker}", bool(config.get("ui.show_timestamps", True))))
        print("Run `cxv doctor` for details.")
        return 2
    print(event_line("listening", bool(config.get("ui.show_timestamps", True))))
    print("Press Ctrl-C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print()
        print(event_line("stopped", bool(config.get("ui.show_timestamps", True))))
        return 0


def emit_jsonl(event: str, **fields: object) -> None:
    import json

    sys.stdout.write(json.dumps({"event": event, **fields}) + "\n")
    sys.stdout.flush()
