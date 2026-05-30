from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .agents import install_agent, list_agents, print_agent
from .audio import list_input_devices, record_input_wav
from .config import Config, load_config, set_config_value, write_default_config
from .daemon import ensure_daemon, is_running, run_serve, send_request, start_background, stop_background
from .doctor import render_doctor, run_doctor
from .models import render_models
from .tui import run_foreground_tui
from .wake import score_wake_audio
from .wake_training import render_wake_training_checks, wake_training_checks


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    overrides = _overrides_from_args(args)
    config = load_config(overrides)
    command = args.command
    if command is None:
        return run_foreground_tui(config)
    try:
        return dispatch(args, config)
    except Exception as exc:
        print(f"cxv: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cxv", description="Local voice-to-Codex bridge.")
    parser.add_argument("--cwd", help="Override Codex working directory for this cxv invocation.")
    parser.add_argument("--agent", help="Override configured Codex custom agent name when native support exists.")
    parser.add_argument("--model", help="Override Codex model.")
    parser.add_argument("--no-start", action="store_true", help="Do not autostart the daemon for commands that need it.")
    parser.add_argument("--jsonl", action="store_true", help="Emit foreground cxv events as JSON Lines.")
    parser.add_argument("--quiet", action="store_true", help="Suppress foreground cxv status/event output.")
    parser.add_argument("--show-partials", action="store_true", help="Show partial transcript events when available.")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("up", help="Start the background cxv daemon.")
    sub.add_parser("down", help="Stop the background cxv daemon.")
    status = sub.add_parser("status", help="Show daemon and binding status.")
    status.add_argument("--json", action="store_true", help="Print the full raw daemon status payload.")
    status.add_argument("--events", type=int, default=5, help="Number of recent events to show in compact status.")
    sub.add_parser("serve", help="Run the cxv daemon in the foreground.")
    sub.add_parser("listen", help="Start daemon if needed and enable listening.")
    sub.add_parser("pause", help="Pause listening while keeping daemon state warm.")

    text = sub.add_parser("text", help="Send typed text through the same route as finalized speech.")
    text.add_argument("text", nargs=argparse.REMAINDER)
    ttt = sub.add_parser("ttt", help="Alias for cxv text.")
    ttt.add_argument("text", nargs=argparse.REMAINDER)
    steer = sub.add_parser("steer", help="Force steer behavior if a Codex turn is active.")
    steer.add_argument("text", nargs=argparse.REMAINDER)
    sub.add_parser("interrupt", help="Interrupt the active Codex turn.")
    sub.add_parser("stop", help="Alias for cxv interrupt, not daemon down.")

    bind = sub.add_parser("bind", help="Bind cxv to a Codex thread/session target.")
    bind.add_argument("--thread", default="", help="Codex thread id to resume/use.")
    bind.add_argument("--cwd", default=".", help="Working directory for future turns.")

    cfg = sub.add_parser("config", help="Manage ~/.config/codex-voice-steer/config.toml.")
    cfg_sub = cfg.add_subparsers(dest="config_command", required=True)
    init = cfg_sub.add_parser("init", help="Create the default config.")
    init.add_argument("--force", action="store_true")
    cfg_sub.add_parser("show", help="Print the resolved config.")
    cfg_sub.add_parser("edit", help="Open config in $EDITOR.")
    cfg_set = cfg_sub.add_parser("set", help="Set a dotted config value.")
    cfg_set.add_argument("key")
    cfg_set.add_argument("value")

    sub.add_parser("models", help="List built-in compatible STT models.")
    sub.add_parser("doctor", help="Check local cxv dependencies and blockers.")

    audio = sub.add_parser("audio", help="Inspect local audio input devices.")
    audio_sub = audio.add_subparsers(dest="audio_command", required=True)
    audio_devices = audio_sub.add_parser("devices", help="List available input devices for audio.device.")
    audio_devices.add_argument("--json", action="store_true", help="Print device list as JSON.")
    audio_record = audio_sub.add_parser("record", help="Record configured input to a 16 kHz mono PCM16 WAV.")
    audio_record.add_argument("wav", help="Output WAV path.")
    audio_record.add_argument("--seconds", type=float, default=5.0, help="Duration to record.")
    audio_record.add_argument("--device", help="Temporary input device override by index or name.")
    audio_record.add_argument("--json", action="store_true", help="Print capture details as JSON.")

    wake = sub.add_parser("wake", help="Wake-model utilities and readiness checks.")
    wake_sub = wake.add_subparsers(dest="wake_command", required=True)
    wake_sub.add_parser("training-status", help="Check local Scarlett wake-model training prerequisites.")
    test_audio = wake_sub.add_parser("test-audio", help="Score a 16 kHz mono PCM16 WAV through the wake detector.")
    test_audio.add_argument("wav", help="Path to the WAV file to score.")
    test_audio.add_argument("--threshold", type=float, default=None, help="Override wake threshold for this test.")

    voice = sub.add_parser("voice", help="Controlled full-pipeline voice test utilities.")
    voice_sub = voice.add_subparsers(dest="voice_command", required=True)
    voice_test = voice_sub.add_parser("test-audio", help="Run a 16 kHz mono PCM16 WAV through wake, VAD, STT, and optional Codex delivery.")
    voice_test.add_argument("wav", help="Path to the WAV file to process.")
    voice_test.add_argument("--send", action="store_true", help="Send the finalized transcript to Codex after STT.")

    agents = sub.add_parser("agents", help="List, install, or print bundled Codex agents.")
    agents_sub = agents.add_subparsers(dest="agents_command", required=True)
    agents_sub.add_parser("list")
    install = agents_sub.add_parser("install")
    install.add_argument("kind", choices=["slim", "msd"])
    install.add_argument("--force", action="store_true")
    prn = agents_sub.add_parser("print")
    prn.add_argument("kind", choices=["slim", "msd"])
    return parser


def dispatch(args: argparse.Namespace, config: Config) -> int:
    if args.command == "serve":
        run_serve()
        return 0
    if args.command == "up":
        pid = start_background(config)
        print(f"cxv daemon running pid={pid}")
        return 0
    if args.command == "down":
        stopped = stop_background(config)
        print("cxv daemon stopped" if stopped else "cxv daemon already stopped")
        return 0
    if args.command == "status":
        return _status(args, config)
    if args.command in {"listen", "pause", "text", "ttt", "steer", "interrupt", "stop", "bind", "voice"}:
        return asyncio.run(_daemon_command(args, config))
    if args.command == "config":
        return _config_command(args, config)
    if args.command == "models":
        print(render_models())
        return 0
    if args.command == "doctor":
        print(render_doctor(run_doctor(config)))
        return 0
    if args.command == "audio":
        return _audio_command(args, config)
    if args.command == "wake":
        return _wake_command(args, config)
    if args.command == "agents":
        return _agents_command(args)
    raise ValueError(f"unknown command: {args.command}")


async def _daemon_command(args: argparse.Namespace, config: Config) -> int:
    await ensure_daemon(config, no_start=args.no_start)
    if args.command == "listen":
        response = await send_request(config, _payload(args, "listen"))
    elif args.command == "pause":
        response = await send_request(config, {"command": "pause"})
    elif args.command in {"text", "ttt", "steer"}:
        text = " ".join(args.text).strip()
        if not text:
            raise ValueError(f"cxv {args.command} requires text")
        command = "steer" if args.command == "steer" else "text"
        response = await send_request(config, _payload(args, command, text=text))
    elif args.command in {"interrupt", "stop"}:
        response = await send_request(config, {"command": "interrupt"})
    elif args.command == "bind":
        response = await send_request(config, {"command": "bind", "thread_id": args.thread, "cwd": args.cwd})
    elif args.command == "voice":
        if args.voice_command != "test-audio":
            raise ValueError(f"unsupported voice command: {args.voice_command}")
        response = await send_request(config, _payload(args, "voice-test-audio", wav=args.wav, send=args.send))
    else:
        raise ValueError(f"unsupported daemon command: {args.command}")
    print(json.dumps(response, indent=2, sort_keys=True))
    return 0 if response.get("ok") else 1


def _payload(args: argparse.Namespace, command: str, **fields: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"command": command, **fields}
    overrides = _overrides_from_args(args)
    if overrides:
        payload["overrides"] = overrides
    return payload


def _status(args: argparse.Namespace, config: Config) -> int:
    if not is_running(config):
        print("cxv daemon: stopped")
        return 0
    response = asyncio.run(send_request(config, {"command": "status"}))
    if args.json:
        print(json.dumps(response, indent=2, sort_keys=True))
    else:
        print(_render_compact_status(response, event_limit=max(args.events, 0)))
    return 0 if response.get("ok") else 1


def _render_compact_status(response: dict[str, Any], event_limit: int = 5) -> str:
    state = dict(response.get("state") or {})
    queued = state.get("queued_inputs") or []
    events = state.get("events") or []
    lines = [
        "cxv daemon: running",
        f"listening: {_yes_no(state.get('listening', False))}",
        f"thread: {state.get('thread_id') or '-'}",
        f"session: {state.get('session_id') or '-'}",
        f"active turn: {state.get('active_turn_id') or '-'}",
        f"queued inputs: {len(queued)}",
        f"cwd: {state.get('cwd') or '.'}",
    ]
    if event_limit and events:
        lines.append("recent events:")
        for event in events[-event_limit:]:
            lines.append(f"- {_event_summary(event)}")
    return "\n".join(lines)


def _yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"


def _event_summary(event: dict[str, Any]) -> str:
    name = str(event.get("event", "event"))
    details: list[str] = []
    for key in ("action", "status", "turn_id", "transcript", "reason", "error"):
        value = event.get(key)
        if value:
            details.append(f"{key}={_clip(str(value))}")
    return name if not details else f"{name} " + " ".join(details)


def _clip(value: str, limit: int = 80) -> str:
    clean = " ".join(value.split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1] + "..."


def _config_command(args: argparse.Namespace, config: Config) -> int:
    if args.config_command == "init":
        path = write_default_config(force=args.force)
        print(path)
        return 0
    if args.config_command == "show":
        print(config.path)
        print(json.dumps(config.data, indent=2, sort_keys=True))
        return 0
    if args.config_command == "edit":
        write_default_config()
        editor = os.environ.get("EDITOR", "vi")
        return subprocess.call([editor, str(config.path)])
    if args.config_command == "set":
        path = set_config_value(args.key, args.value)
        print(path)
        return 0
    raise ValueError(f"unknown config command: {args.config_command}")


def _agents_command(args: argparse.Namespace) -> int:
    if args.agents_command == "list":
        print(list_agents())
        return 0
    if args.agents_command == "print":
        print(print_agent(args.kind))
        return 0
    if args.agents_command == "install":
        target = install_agent(args.kind, force=args.force)
        print(f"Installed: {target}")
        print("Select it with: cxv config set codex.agent " + ("cxv-voice-msd" if args.kind == "msd" else "cxv-voice-slim"))
        print("Until native app-server agent selection is proven, cxv injects the selected agent instructions.")
        return 0
    raise ValueError(f"unknown agents command: {args.agents_command}")


def _audio_command(args: argparse.Namespace, config: Config | None = None) -> int:
    if args.audio_command == "devices":
        devices = list_input_devices()
        if args.json:
            print(json.dumps([device.to_dict() for device in devices], indent=2, sort_keys=True))
        else:
            print(_render_audio_devices(devices))
        return 0
    if args.audio_command == "record":
        cfg = config or load_config()
        if args.device:
            cfg = cfg.with_overrides({"audio": {"device": args.device}})
        result = record_input_wav(cfg, Path(args.wav), seconds=float(args.seconds))
        if args.json:
            print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        else:
            print(f"recorded {result.seconds:.2f}s from {result.device}: {result.wav_path}")
            print("Verify with: cxv wake test-audio " + str(result.wav_path))
        return 0
    raise ValueError(f"unknown audio command: {args.audio_command}")


def _render_audio_devices(devices) -> str:
    lines = ["cxv audio input devices"]
    if not devices:
        lines.append("(none)")
        return "\n".join(lines)
    for device in devices:
        marker = " *" if device.is_default else ""
        lines.append(f"{device.index}: {device.name} ({device.max_input_channels} input channel(s)){marker}")
    lines.append("Use one with: cxv config set audio.device <index-or-name>")
    return "\n".join(lines)


def _wake_command(args: argparse.Namespace, config: Config) -> int:
    if args.wake_command == "training-status":
        checks = wake_training_checks()
        print(render_wake_training_checks(checks))
        return 0 if all(check.ok for check in checks) else 1
    if args.wake_command == "test-audio":
        result = score_wake_audio(config, Path(args.wav), threshold=args.threshold)
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0 if result.hit else 1
    raise ValueError(f"unknown wake command: {args.wake_command}")


def _overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    codex: dict[str, Any] = {}
    ui: dict[str, Any] = {}
    if getattr(args, "cwd", None):
        codex["cwd"] = args.cwd
    if getattr(args, "agent", None):
        codex["agent"] = args.agent
    if getattr(args, "model", None):
        codex["model"] = args.model
    if getattr(args, "jsonl", False):
        ui["mode"] = "jsonl"
    if getattr(args, "quiet", False):
        ui["mode"] = "quiet"
    if getattr(args, "show_partials", False):
        ui["show_partial_transcripts"] = True
    if codex:
        overrides["codex"] = codex
    if ui:
        overrides["ui"] = ui
    return overrides


if __name__ == "__main__":
    raise SystemExit(main())
