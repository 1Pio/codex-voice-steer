from __future__ import annotations

import asyncio
import sys
import time
from typing import Any

from .config import Config
from .audio import audio_readiness
from .daemon import ensure_daemon, send_request
from .vad import vad_readiness
from .wake import wake_readiness


def event_line(message: str, timestamps: bool = True) -> str:
    if not timestamps:
        return message
    return f"{time.strftime('%H:%M:%S')}  {message}"


def run_foreground_tui(config: Config, poll_interval: float = 0.5, max_polls: int | None = None) -> int:
    mode = str(config.get("ui.mode", "interactive"))
    if mode == "jsonl":
        emit_jsonl(
            "ready",
            wake=config.get("wake.word", "scarlett"),
            stt=f"{config.get('stt.engine', 'macparakeet')} {config.get('stt.mode', 'clean')}",
            codex="app-server",
        )
    elif mode != "quiet":
        print("cxv 0.1.0  codex-voice-steer")
        print(f"wake: {config.get('wake.word', 'scarlett')}     stt: {config.get('stt.engine', 'macparakeet')} {config.get('stt.mode', 'clean')}     codex: app-server")
    blockers = []
    for readiness in (audio_readiness(config, probe_stream=True), vad_readiness(), wake_readiness(config)):
        if not readiness.ok:
            blockers.append(readiness.reason)
    if blockers:
        for blocker in blockers:
            write_ui(config, "blocked", f"blocked: {blocker}", blocker=blocker)
        if mode != "jsonl" and mode != "quiet":
            print("Run `cxv doctor` for details.")
        return 2
    if mode != "jsonl" and mode != "quiet":
        print("Press Ctrl-C to stop.")
    try:
        return asyncio.run(_run_foreground_listener(config, poll_interval=poll_interval, max_polls=max_polls))
    except KeyboardInterrupt:
        if mode != "jsonl" and mode != "quiet":
            print()
        try:
            asyncio.run(send_request(config, {"command": "pause"}))
        except Exception as exc:
            write_ui(config, "pause_failed", f"pause failed: {exc}", error=str(exc))
        write_ui(config, "stopped", "stopped")
        return 0


async def _run_foreground_listener(config: Config, poll_interval: float, max_polls: int | None) -> int:
    await ensure_daemon(config)
    before = await send_request(config, {"command": "status"})
    last_seen_ts = _last_event_ts(before.get("state", {}).get("events", []))
    response = await send_request(config, {"command": "listen"})
    if not response.get("ok"):
        for blocker in response.get("blockers", []):
            write_ui(config, "blocked", f"blocked: {blocker}", blocker=blocker)
        return 1
    write_ui(config, "listening", "listening")
    polls = 0
    try:
        while True:
            try:
                status = await send_request(config, {"command": "status"})
            except Exception as exc:
                write_ui(config, "daemon_lost", f"daemon lost: {exc}", error=str(exc))
                return 1
            events = list(status.get("state", {}).get("events", []))
            new_events = _events_after(events, last_seen_ts)
            for event in new_events:
                rendered = render_event(event, config)
                if rendered:
                    write_ui(config, str(event.get("event", "event")), rendered, source_event=event)
            if new_events:
                last_seen_ts = _last_event_ts(new_events)
            polls += 1
            if max_polls is not None and polls >= max_polls:
                return 0
            await asyncio.sleep(poll_interval)
    finally:
        if max_polls is not None:
            await send_request(config, {"command": "pause"})


def _events_after(events: list[dict[str, Any]], timestamp: float) -> list[dict[str, Any]]:
    return [event for event in events if _event_ts(event) > timestamp]


def _last_event_ts(events: list[dict[str, Any]]) -> float:
    if not events:
        return 0.0
    return max((_event_ts(event) for event in events), default=0.0)


def _event_ts(event: dict[str, Any]) -> float:
    try:
        return float(event.get("ts", 0.0))
    except (TypeError, ValueError):
        return 0.0


def render_event(event: dict[str, Any], config: Config | None = None) -> str:
    name = str(event.get("event", ""))
    if name == "wake_detected":
        if config is not None and not config.get("ui.show_wake_events", True):
            return ""
        return "wake detected"
    if name == "vad_final":
        if config is not None and not config.get("ui.show_vad_events", True):
            return ""
        return f"vad final: {event.get('wav_path', '')}"
    if name == "stt_final":
        if config is not None and not config.get("ui.show_final_transcripts", True):
            return ""
        return "user: " + str(event.get("transcript", "")).strip()
    if name == "user_final":
        if config is not None and not config.get("ui.show_final_transcripts", True):
            return ""
        return "user: " + str(event.get("text", "")).strip()
    if name == "sent":
        return f"sent: {event.get('action', '')}"
    if name == "turn_started":
        return f"turn started: {event.get('turn_id', '')}"
    if name == "turn_completed":
        return f"turn completed: {event.get('turn_id', '')}"
    if name == "voice_turn":
        return f"voice turn: {event.get('status', '')}"
    if name == "voice_test_audio":
        return f"voice test: {event.get('status', '')}"
    if name == "voice_error":
        return f"voice error: {event.get('error', '')}"
    if name == "codex_visible_delta":
        if config is not None and not config.get("ui.show_codex_visible_messages", True):
            return ""
        return "codex: " + str(event.get("delta", "")).strip()
    return ""


def write_ui(config: Config, event: str, message: str, **fields: object) -> None:
    mode = str(config.get("ui.mode", "interactive"))
    if mode == "quiet":
        return
    if mode == "jsonl":
        emit_jsonl(event, message=message, **fields)
        return
    print(event_line(message, bool(config.get("ui.show_timestamps", True))))


def emit_jsonl(event: str, **fields: object) -> None:
    import json

    sys.stdout.write(json.dumps({"event": event, **fields}) + "\n")
    sys.stdout.flush()
