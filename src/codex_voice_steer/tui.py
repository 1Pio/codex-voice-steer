from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from .config import Config
from .audio import audio_readiness
from .daemon import ensure_daemon, send_request, stop_background
from .session import render_session_header, session_status_info
from .vad import vad_readiness
from .wake import wake_readiness


BOLD_LABELS = {"user:", "codex:", "codex msd:"}


def event_line(message: str, timestamps: bool = True, timestamp_opacity: float = 1.0) -> str:
    if not timestamps:
        return message
    timestamp = time.strftime("%H:%M:%S")
    return f"{_timestamp_label(timestamp, timestamp_opacity)}  {message}"


@dataclass
class DisplayState:
    shown_user_texts: set[str] = field(default_factory=set)


def run_foreground_tui(
    config: Config,
    poll_interval: float = 0.5,
    max_polls: int | None = None,
    listen_overrides: dict[str, Any] | None = None,
) -> int:
    mode = str(config.get("ui.mode", "interactive"))
    if mode == "jsonl":
        session_info = session_status_info(config)
        emit_jsonl(
            "ready",
            wake=config.get("wake.word", "scarlett"),
            stt=f"{config.get('stt.engine', 'macparakeet')} {config.get('stt.mode', 'clean')}",
            codex="app-server",
            session_id=session_info["saved_session_id"],
            resume_thread_id=session_info["effective_resume_thread_id"],
            resume_source=session_info["effective_resume_source"],
        )
    elif mode != "quiet":
        print("cxv 0.1.0  codex-voice-steer")
        print(
            f"wake: {config.get('wake.word', 'scarlett')}     "
            f"stt: {config.get('stt.engine', 'macparakeet')} {config.get('stt.mode', 'clean')}     "
            f"codex: app-server     {render_session_header(config)}"
        )
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
        return asyncio.run(_run_foreground_listener(config, poll_interval=poll_interval, max_polls=max_polls, listen_overrides=listen_overrides))
    except KeyboardInterrupt:
        if mode != "jsonl" and mode != "quiet":
            print()
        try:
            stop_background(config)
        except Exception as exc:
            write_ui(config, "stop_failed", f"stop failed: {exc}", error=str(exc))
        write_ui(config, "stopped", "stopped")
        return 0


async def _run_foreground_listener(
    config: Config,
    poll_interval: float,
    max_polls: int | None,
    listen_overrides: dict[str, Any] | None = None,
) -> int:
    await ensure_daemon(config)
    before = await send_request(config, {"command": "status"})
    last_seen_ts = _last_event_ts(before.get("state", {}).get("events", []))
    listen_payload: dict[str, Any] = {"command": "listen"}
    if listen_overrides:
        listen_payload["overrides"] = listen_overrides
    response = await send_request(config, listen_payload)
    if not response.get("ok"):
        for blocker in response.get("blockers", []):
            write_ui(config, "blocked", f"blocked: {blocker}", blocker=blocker)
        return 1
    write_ui(config, "listening", "listening")
    display_state = DisplayState()
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
            for event, rendered in render_events(new_events, config, display_state=display_state):
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
    if not _event_visible(name, config):
        return ""
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
        return _labeled(config, "user:", _limit_lines(str(event.get("transcript", "")).strip(), _line_limit(config, "ui.max_transcript_lines", 200)))
    if name == "user_final":
        if config is not None and not config.get("ui.show_final_transcripts", True):
            return ""
        return _labeled(config, "user:", _limit_lines(str(event.get("text", "")).strip(), _line_limit(config, "ui.max_transcript_lines", 200)))
    if name == "sent":
        return _labeled(config, "sent:", str(event.get("action", "")))
    if name == "turn_started":
        return _labeled(config, "turn started:", str(event.get("turn_id", "")))
    if name == "turn_completed":
        return _labeled(config, "turn completed:", str(event.get("turn_id", "")))
    if name == "voice_turn":
        return _labeled(config, "voice turn:", str(event.get("status", "")))
    if name == "voice_test_audio":
        return _labeled(config, "voice test:", str(event.get("status", "")))
    if name == "voice_error":
        return _labeled(config, "voice error:", str(event.get("error", "")))
    if name == "codex_visible_delta":
        if config is not None and not config.get("ui.show_codex_visible_messages", True):
            return ""
        return _labeled(config, "codex:", str(event.get("delta", "")).strip())
    if name == "codex_final_answer":
        if config is not None and not config.get("ui.show_codex_final_answers", True):
            return ""
        text = _limit_lines(str(event.get("text", "")).strip(), _line_limit(config, "ui.max_codex_answer_lines", 200))
        return _labeled(config, "codex:", text)
    if name == "codex_tool_started":
        if config is not None and not config.get("ui.show_codex_tool_traces", True):
            return ""
        text = _limit_lines(str(event.get("summary", "")).strip(), _line_limit(config, "ui.max_codex_action_lines", 1))
        return _labeled(config, "codex action:", text)
    if name == "codex_msd_started":
        if config is not None and not config.get("ui.show_codex_msd_traces", True):
            return ""
        text = _limit_lines(_codex_msd_text(event), _line_limit(config, "ui.max_codex_msd_lines", 20))
        return _labeled(config, "codex msd:", text)
    if name == "codex_tool_progress":
        if config is not None and not config.get("ui.show_codex_tool_traces", True):
            return ""
        return _labeled(config, "codex progress:", str(event.get("message", "")).strip())
    if name == "codex_token_usage":
        return _labeled(config, "context:", _context_usage_text(event))
    if name == "auto_compact_scheduled":
        return _labeled(config, "auto compact:", f"scheduled in {float(event.get('idle_delay_sec', 0.0)):g}s at {_ratio_label(event)}")
    if name == "auto_compact_cancelled":
        return _labeled(config, "auto compact:", f"cancelled: {event.get('source', 'activity')}")
    if name == "auto_compact_started":
        return _labeled(config, "auto compact:", f"started at {_ratio_label(event)}")
    if name == "auto_compact_completed":
        return _labeled(config, "auto compact:", "completed")
    if name == "auto_compact_skipped":
        return _labeled(config, "auto compact:", f"skipped: {event.get('reason', '')}")
    if name == "auto_compact_failed":
        return _labeled(config, "auto compact:", f"failed: {event.get('error', '')}")
    return ""


def render_events(
    events: list[dict[str, Any]],
    config: Config | None = None,
    display_state: DisplayState | None = None,
) -> list[tuple[dict[str, Any], str]]:
    display_state = display_state or DisplayState()
    user_final_texts = {
        _normalized_event_text(event.get("text", ""))
        for event in events
        if str(event.get("event", "")) == "user_final"
    }
    rendered_events: list[tuple[dict[str, Any], str]] = []
    for event in events:
        name = str(event.get("event", ""))
        user_text = _user_event_text(event)
        normalized_user_text = _normalized_event_text(user_text)
        if name == "stt_final" and normalized_user_text in user_final_texts:
            continue
        if user_text and normalized_user_text in display_state.shown_user_texts:
            continue
        rendered = render_event(event, config)
        if rendered:
            rendered_events.append((event, rendered))
            if user_text:
                display_state.shown_user_texts.add(normalized_user_text)
    return rendered_events


def _normalized_event_text(value: object) -> str:
    return " ".join(str(value).strip().split())


def _user_event_text(event: dict[str, Any]) -> str:
    name = str(event.get("event", ""))
    if name == "stt_final":
        return str(event.get("transcript", ""))
    if name == "user_final":
        return str(event.get("text", ""))
    return ""


def write_ui(config: Config, event: str, message: str, **fields: object) -> None:
    mode = str(config.get("ui.mode", "interactive"))
    if mode == "quiet":
        return
    if mode == "jsonl":
        emit_jsonl(event, message=message, **fields)
        return
    print(event_line(message, bool(config.get("ui.show_timestamps", True)), float(config.get("ui.timestamp_opacity", 1.0))))


def emit_jsonl(event: str, **fields: object) -> None:
    import json

    sys.stdout.write(json.dumps({"event": event, **fields}) + "\n")
    sys.stdout.flush()


def _event_visible(name: str, config: Config | None) -> bool:
    if config is None:
        return True
    visible = _string_list(config.get("ui.visible_events", []))
    hidden = set(_string_list(config.get("ui.hidden_events", [])))
    if name in hidden:
        return False
    return not visible or name in visible


def _string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


def _labeled(config: Config | None, label: str, text: str) -> str:
    rendered_label = _label(config, label)
    return rendered_label if not text else f"{rendered_label} {text}"


def _label(config: Config | None, label: str) -> str:
    if config is None:
        return label
    if str(config.get("ui.mode", "interactive")) == "jsonl":
        return label
    if label not in BOLD_LABELS:
        return label
    if not config.get("ui.bold_labels", True):
        return label
    return f"\x1b[1m{label}\x1b[0m"


def _limit_lines(text: str, max_lines: int) -> str:
    if max_lines < 1:
        return ""
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    remaining = len(lines) - max_lines
    return "\n".join([*lines[:max_lines], f"... truncated {remaining} line(s)"])


def _line_limit(config: Config | None, key: str, default: int) -> int:
    if config is None:
        return default
    try:
        return int(config.get(key, default))
    except (TypeError, ValueError):
        return default


def _timestamp_label(timestamp: str, opacity: float) -> str:
    opacity = max(0.0, min(1.0, opacity))
    if opacity >= 0.999:
        return timestamp
    gray = int(round(255 * opacity))
    return f"\x1b[38;2;{gray};{gray};{gray}m{timestamp}\x1b[0m"


def _codex_msd_text(event: dict[str, Any]) -> str:
    msd_args = str(event.get("msd_args", "")).strip()
    if msd_args:
        return msd_args
    summary = str(event.get("summary", "")).strip()
    marker = "msd say"
    index = summary.find(marker)
    if index < 0:
        return summary
    return summary[index + len(marker) :].strip().strip("'\"")


def _context_usage_text(event: dict[str, Any]) -> str:
    total = int(event.get("total_tokens", 0) or 0)
    window = int(event.get("model_context_window", 0) or 0)
    if window <= 0:
        return f"{total} tokens"
    return f"{_ratio_label(event)} ({total}/{window})"


def _ratio_label(event: dict[str, Any]) -> str:
    try:
        ratio = float(event.get("usage_ratio", 0.0) or 0.0)
    except (TypeError, ValueError):
        ratio = 0.0
    return f"{ratio * 100:.1f}%"
