from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .agents import install_agent, list_agents, print_agent
from .audio import input_levels, list_input_devices, list_output_devices, play_and_record_input_wav, play_wav, record_input_wav
from .calibration import calibrate_wake
from .config import Config, load_config, set_config_value, unset_config_value, write_default_config
from .daemon import ensure_daemon, is_running, run_serve, send_request, start_background, stop_background
from .doctor import render_doctor, run_doctor
from .models import render_models
from .session import render_session_status, session_status_info
from .tui import run_foreground_tui
from .wake import score_wake_audio
from .wake_samples import (
    DEFAULT_SYNTHETIC_MSD_INSTRUCTS,
    DEFAULT_SYNTHETIC_MSD_LANGUAGES,
    DEFAULT_SYNTHETIC_MSD_VOICES,
    HARD_NEGATIVE_PROMPTS,
    ENVIRONMENTAL_NEGATIVE_PROMPTS,
    capture_take,
    generate_synthetic_msd_samples,
    init_dataset,
    next_take_index,
    prompts_for_args,
    read_prompts_file,
    render_dataset_summary,
    render_score_summary,
    render_synthetic_msd_summary,
    score_dataset,
    summarize_dataset,
    split_csv_values,
)
from .wake_training import render_wake_training_checks, wake_training_checks


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    overrides = _overrides_from_args(args)
    config = load_config(overrides)
    command = args.command
    if command is None:
        return run_foreground_tui(config, listen_overrides=overrides)
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
    parser.add_argument("--fast", action="store_true", help="Request Codex fast service tier for this invocation.")
    parser.add_argument(
        "--effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        help="Override Codex reasoning effort for this invocation.",
    )
    parser.add_argument("--no-start", action="store_true", help="Do not autostart the daemon for commands that need it.")
    parser.add_argument("--jsonl", action="store_true", help="Emit foreground cxv events as JSON Lines.")
    parser.add_argument("--quiet", action="store_true", help="Suppress foreground cxv status/event output.")
    parser.add_argument("--show-partials", action="store_true", help="Show partial transcript events when available.")
    parser.add_argument("--timestamp-opacity", type=float, help="Dim foreground TUI timestamps with an ANSI opacity approximation from 0.0 to 1.0.")
    parser.add_argument("--plain-labels", action="store_true", help="Disable bold labels such as user:, codex:, and codex msd: in the foreground TUI.")
    parser.add_argument("--show-events", help="Comma-separated foreground event names to show; when set, other event-history lines are hidden.")
    parser.add_argument("--hide-events", help="Comma-separated foreground event names to hide.")
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

    session = sub.add_parser("session", help="Inspect or refresh the saved Codex session.")
    session_sub = session.add_subparsers(dest="session_command", required=True)
    session_status = session_sub.add_parser("status", help="Show the session/thread cxv will resume and related behavior.")
    session_status.add_argument("--json", action="store_true", help="Print session status as JSON.")
    session_new = session_sub.add_parser("new", help="Start a fresh Codex thread and save it for future turns.")
    session_new.add_argument("--force", action="store_true", help="Interrupt an active turn if needed before starting the new session.")
    session_new.add_argument("--json", action="store_true", help="Print the raw daemon response as JSON.")

    cfg = sub.add_parser("config", help="Manage ~/.config/codex-voice-steer/config.toml.")
    cfg_sub = cfg.add_subparsers(dest="config_command", required=True)
    init = cfg_sub.add_parser("init", help="Create the default config.")
    init.add_argument("--force", action="store_true")
    cfg_sub.add_parser("show", help="Print the resolved config.")
    cfg_sub.add_parser("edit", help="Open config in $EDITOR.")
    cfg_set = cfg_sub.add_parser("set", help="Set a dotted config value.")
    cfg_set.add_argument("key")
    cfg_set.add_argument("value")
    cfg_unset = cfg_sub.add_parser("unset", help="Remove a key from the user config.")
    cfg_unset.add_argument("key")

    sub.add_parser("models", help="List built-in compatible STT models.")
    sub.add_parser("doctor", help="Check local cxv dependencies and blockers.")

    audio = sub.add_parser("audio", help="Inspect local audio input devices.")
    audio_sub = audio.add_subparsers(dest="audio_command", required=True)
    audio_devices = audio_sub.add_parser("devices", help="List available input devices for audio.device.")
    audio_devices.add_argument("--json", action="store_true", help="Print device list as JSON.")
    audio_devices.add_argument("--kind", choices=["input", "output", "all"], default="input", help="Device kind to list.")
    audio_record = audio_sub.add_parser("record", help="Record configured input to a 16 kHz mono PCM16 WAV.")
    audio_record.add_argument("wav", help="Output WAV path.")
    audio_record.add_argument("--seconds", type=float, default=5.0, help="Duration to record.")
    audio_record.add_argument("--device", help="Temporary input device override by index or name.")
    audio_record.add_argument("--gain-db", type=float, help="Temporary input gain in decibels.")
    audio_record.add_argument("--json", action="store_true", help="Print capture details as JSON.")
    audio_loopback = audio_sub.add_parser("loopback-test", help="Play a WAV while recording the configured input for loopback verification.")
    audio_loopback.add_argument("source_wav", help="16 kHz mono PCM16 WAV to play to the output device.")
    audio_loopback.add_argument("captured_wav", help="Output WAV captured from the configured input.")
    audio_loopback.add_argument("--seconds", type=float, help="Override capture/play duration; defaults to source length.")
    audio_loopback.add_argument("--device", help="Temporary input device override by index or name.")
    audio_loopback.add_argument("--output-device", default="default", help="Temporary output device override by index or name.")
    audio_loopback.add_argument("--gain-db", type=float, help="Temporary input gain in decibels.")
    audio_loopback.add_argument("--json", action="store_true", help="Print loopback capture details as JSON.")
    audio_play = audio_sub.add_parser("play", help="Play a WAV to an output device for controlled loopback tests.")
    audio_play.add_argument("source_wav", help="PCM16 WAV to play to the output device.")
    audio_play.add_argument("--output-device", default="default", help="Temporary output device override by index or name.")
    audio_play.add_argument("--json", action="store_true", help="Print playback details as JSON.")
    audio_meter = audio_sub.add_parser("meter", help="Print live input RMS/peak levels for the configured device.")
    audio_meter.add_argument("--seconds", type=float, default=5.0, help="Duration to monitor.")
    audio_meter.add_argument("--interval-ms", type=int, default=500, help="Level reporting interval.")
    audio_meter.add_argument("--device", help="Temporary input device override by index or name.")
    audio_meter.add_argument("--gain-db", type=float, help="Temporary input gain in decibels.")
    audio_meter.add_argument("--jsonl", action="store_true", help="Print each level sample as JSON Lines.")

    wake = sub.add_parser("wake", help="Wake-model utilities and readiness checks.")
    wake_sub = wake.add_subparsers(dest="wake_command", required=True)
    training_status = wake_sub.add_parser("training-status", help="Check local Scarlett wake-model training prerequisites.")
    training_status.add_argument(
        "--python",
        dest="training_python",
        help="Training venv Python; defaults to CXV_WAKE_TRAINING_PYTHON or the durable wakeword cache venv.",
    )
    test_audio = wake_sub.add_parser("test-audio", help="Score a 16 kHz mono PCM16 WAV through the wake detector.")
    test_audio.add_argument("wav", help="Path to the WAV file to score.")
    test_audio.add_argument("--threshold", type=float, default=None, help="Override wake threshold for this test.")
    calibrate = wake_sub.add_parser("calibrate", help="Record a live sample and score it through the wake detector.")
    calibrate.add_argument("wav", help="Output WAV path for the captured calibration sample.")
    calibrate.add_argument("--seconds", type=float, default=5.0, help="Duration to record.")
    calibrate.add_argument("--device", help="Temporary input device override by index or name.")
    calibrate.add_argument("--gain-db", type=float, help="Temporary input gain in decibels.")
    calibrate.add_argument("--threshold", type=float, default=None, help="Override wake threshold for this calibration.")
    calibrate.add_argument("--min-rms", type=float, default=1000.0, help="Minimum RMS for a strong enough live proof.")
    calibrate.add_argument("--min-peak", type=int, default=4000, help="Minimum peak amplitude for a strong enough live proof.")
    samples = wake_sub.add_parser("samples", help="Capture and score real-user wake-word training samples.")
    samples_sub = samples.add_subparsers(dest="samples_command", required=True)
    samples_init = samples_sub.add_parser("init", help="Create a wake sample dataset directory.")
    samples_init.add_argument("dir", help="Dataset directory.")
    samples_record = samples_sub.add_parser("record", help="Record one take; press Enter to stop and save.")
    _add_wake_samples_capture_args(samples_record)
    samples_session = samples_sub.add_parser("session", help="Record many takes with Space/Enter/q controls.")
    _add_wake_samples_capture_args(samples_session, include_prompt=False)
    prompt_group = samples_session.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt", default="", help="Prompt text for this take.")
    prompt_group.add_argument("--prompts", help="File with one prompt per line.")
    prompt_group.add_argument("--preset", choices=["scarlett"], help="Use built-in prompt sequence.")
    samples_list = samples_sub.add_parser("list", help="Summarize wake sample counts, duration, levels, tags, and prompts.")
    samples_list.add_argument("dir", help="Dataset directory.")
    samples_list.add_argument("--json", action="store_true", help="Print summary as JSON.")
    samples_score = samples_sub.add_parser("score", help="Score sample WAVs through the current OpenWakeWord adapter.")
    samples_score.add_argument("dir", help="Dataset directory.")
    samples_score.add_argument(
        "--model",
        default=None,
        help="Wake model path; defaults to configured wake.model_path, falling back to models/wake/scarlett.onnx.",
    )
    samples_score.add_argument("--threshold", type=float, default=None, help="Override wake threshold.")
    samples_score.add_argument("--label", choices=["positive", "negative", "noise"], default=None, help="Score only one label folder.")
    samples_score.add_argument("--no-receipt", action="store_true", help="Do not append scores.jsonl; useful for read-only evaluation of manual datasets.")
    samples_score.add_argument("--scores-path", default=None, help="Append score receipts to this JSONL path instead of the dataset scores.jsonl.")
    samples_score.add_argument("--json", action="store_true", help="Print score summary as JSON.")
    samples_synthetic = samples_sub.add_parser("synthetic-msd", help="Generate optional synthetic negative wake samples with msd render.")
    samples_synthetic.add_argument("dir", help="Synthetic dataset directory.")
    samples_synthetic.add_argument("--prompts", action="append", required=True, help="Prompt file with one negative prompt per line; may be repeated.")
    samples_synthetic.add_argument("--tag", default="synthetic-negative", help="Tag written to filenames and metadata.")
    samples_synthetic.add_argument("--count", type=int, required=True, help="Number of synthetic negative samples to generate.")
    samples_synthetic.add_argument("--voices", default=",".join(DEFAULT_SYNTHETIC_MSD_VOICES), help="Comma-separated msd voices.")
    samples_synthetic.add_argument("--languages", default=",".join(DEFAULT_SYNTHETIC_MSD_LANGUAGES), help="Comma-separated language/style labels passed to msd.")
    samples_synthetic.add_argument("--instructs", nargs="+", default=DEFAULT_SYNTHETIC_MSD_INSTRUCTS, help="One or more msd --instruct strings.")
    samples_synthetic.add_argument("--model", default="", help="Optional msd model alias or Hugging Face model id.")
    samples_synthetic.add_argument("--speed", type=float, default=None, help="Optional msd speed hint.")
    samples_synthetic.add_argument("--ttl", type=float, default=None, help="Optional msd daemon ttl hint.")
    samples_synthetic.add_argument("--msd-bin", default="msd", help="msd executable path.")
    samples_synthetic.add_argument("--converter", default=None, help="Optional ffmpeg or sox executable for WAV normalization.")
    samples_synthetic.add_argument("--progress-every", type=int, default=25, help="Print progress every N generated samples; 0 disables progress.")
    samples_synthetic.add_argument("--dry-run", action="store_true", help="Validate and summarize the generation plan without writing files.")
    samples_synthetic.add_argument("--json", action="store_true", help="Print summary as JSON.")

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
    if args.command == "session":
        return _session_command(args, config)
    if args.command in {"listen", "pause", "text", "ttt", "steer", "interrupt", "stop", "bind", "voice"}:
        return asyncio.run(_daemon_command(args, config))
    if args.command == "config":
        return _config_command(args, config)
    if args.command == "models":
        print(render_models())
        return 0
    if args.command == "doctor":
        checks = run_doctor(config)
        print(render_doctor(checks))
        return 0 if all(check.ok for check in checks) else 1
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


def _session_command(args: argparse.Namespace, config: Config) -> int:
    if args.session_command == "status":
        info = session_status_info(config)
        if args.json:
            print(json.dumps(info, indent=2, sort_keys=True))
        else:
            print(render_session_status(info))
        return 0
    if args.session_command == "new":
        return asyncio.run(_session_new(args, config))
    raise ValueError(f"unsupported session command: {args.session_command}")


async def _session_new(args: argparse.Namespace, config: Config) -> int:
    await ensure_daemon(config, no_start=args.no_start)
    response = await send_request(config, _payload(args, "session-new", force=args.force))
    if args.json:
        print(json.dumps(response, indent=2, sort_keys=True))
    elif response.get("ok"):
        state = dict(response.get("state") or {})
        print("cxv session refreshed")
        print(f"thread: {state.get('thread_id') or '-'}")
        print(f"session: {state.get('session_id') or '-'}")
        print(f"cwd: {state.get('cwd') or '.'}")
    else:
        print(f"cxv: {response.get('error', 'session refresh failed')}", file=sys.stderr)
        configured = response.get("configured_thread_id")
        source = response.get("configured_source")
        if configured and source:
            print(f"configured {source}: {configured}", file=sys.stderr)
            print(f"unset it with: cxv config unset {source}", file=sys.stderr)
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
    event_time = _event_time_label(event)
    if event_time:
        details.append(f"at={event_time}")
    for key in ("action", "status", "turn_id", "transcript", "reason", "error", "device", "noop"):
        value = event.get(key)
        if value is not None and value != "":
            details.append(f"{key}={_clip(str(value))}")
    return name if not details else f"{name} " + " ".join(details)


def _event_time_label(event: dict[str, Any]) -> str:
    try:
        ts = float(event.get("ts", 0.0))
    except (TypeError, ValueError):
        return ""
    if ts <= 0:
        return ""
    return time.strftime("%H:%M:%S", time.localtime(ts))


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
    if args.config_command == "unset":
        path = unset_config_value(args.key)
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
        devices = _audio_devices(str(args.kind))
        if args.json:
            print(json.dumps([device.to_dict() for device in devices], indent=2, sort_keys=True))
        else:
            print(_render_audio_devices(devices, kind=str(args.kind)))
        return 0
    if args.audio_command == "record":
        cfg = config or load_config()
        cfg = _audio_override_config(cfg, device=args.device, gain_db=args.gain_db)
        result = record_input_wav(cfg, Path(args.wav), seconds=float(args.seconds))
        if args.json:
            print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        else:
            print(f"recorded {result.seconds:.2f}s from {result.device}: {result.wav_path}")
            print("Verify with: cxv wake test-audio " + str(result.wav_path))
        return 0
    if args.audio_command == "loopback-test":
        cfg = config or load_config()
        cfg = _audio_override_config(cfg, device=args.device, gain_db=args.gain_db)
        result = play_and_record_input_wav(
            cfg,
            Path(args.source_wav),
            Path(args.captured_wav),
            seconds=args.seconds,
            output_device=str(args.output_device),
        )
        if args.json:
            print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        else:
            print(f"played {result.source_wav_path} to output {result.output_device}; captured {result.seconds:.2f}s from {result.device}: {result.wav_path}")
            print("Verify route with: cxv wake test-audio " + str(result.wav_path))
        return 0
    if args.audio_command == "play":
        result = play_wav(Path(args.source_wav), output_device=str(args.output_device))
        if args.json:
            print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        else:
            print(f"played {result.seconds:.2f}s to output {result.output_device}: {result.source_wav_path}")
        return 0
    if args.audio_command == "meter":
        cfg = config or load_config()
        cfg = _audio_override_config(cfg, device=args.device, gain_db=args.gain_db)
        max_peak = 0
        last_device = str(cfg.get("audio.device", "default"))
        last_device_name = ""
        samples_seen = 0
        for level in input_levels(cfg, seconds=float(args.seconds), interval_ms=int(args.interval_ms)):
            max_peak = max(max_peak, int(level.peak))
            last_device = level.device
            last_device_name = level.device_name
            samples_seen += int(level.samples)
            if args.jsonl:
                print(json.dumps(level.to_dict(), sort_keys=True))
            else:
                device_label = level.device if not level.device_name else f"{level.device} ({level.device_name})"
                print(
                    f"{level.elapsed_sec:5.2f}s  rms={level.rms:8.2f}  peak={level.peak:5d}  "
                    f"clipped={level.clipped_ratio:.3f}  gain={level.gain_db:g}dB  device={device_label}"
                )
        if samples_seen and max_peak == 0 and not args.jsonl:
            device_label = last_device if not last_device_name else f"{last_device} ({last_device_name})"
            print(
                "warning: input stream opened but captured digital silence from "
                f"{device_label}; try `cxv audio meter --device \"MacBook Pro Microphone\"`, "
                "check macOS microphone permission for your terminal app, and avoid numeric device indexes if they change."
            )
        return 0
    raise ValueError(f"unknown audio command: {args.audio_command}")


def _audio_devices(kind: str):
    if kind == "input":
        return list_input_devices()
    if kind == "output":
        return list_output_devices()
    if kind == "all":
        seen: dict[int, object] = {}
        for device in [*list_input_devices(), *list_output_devices()]:
            seen[device.index] = device
        return list(seen.values())
    raise ValueError(f"unknown audio device kind: {kind}")


def _render_audio_devices(devices, kind: str = "input") -> str:
    lines = [f"cxv audio {kind} devices"]
    if not devices:
        lines.append("(none)")
        return "\n".join(lines)
    for device in devices:
        marker = " *" if device.is_default else ""
        if kind == "input":
            capability = f"{device.max_input_channels} input channel(s)"
        elif kind == "output":
            capability = f"{device.max_output_channels} output channel(s)"
        else:
            capability = f"{device.max_input_channels} input / {device.max_output_channels} output channel(s)"
        lines.append(f"{device.index}: {device.name} ({capability}){marker}")
    if kind == "output":
        lines.append("Use one with: cxv audio loopback-test --output-device <index-or-name> ...")
    elif kind == "all":
        lines.append("Use input with: cxv config set audio.device <index-or-name>")
        lines.append('Prefer a quoted input name if indexes change, for example: cxv config set audio.device "MacBook Pro Microphone"')
        lines.append("Use output with: cxv audio loopback-test --output-device <index-or-name> ...")
    else:
        lines.append("Use one with: cxv config set audio.device <index-or-name>")
        lines.append('Prefer a quoted name if indexes change, for example: cxv config set audio.device "MacBook Pro Microphone"')
    return "\n".join(lines)


def _wake_command(args: argparse.Namespace, config: Config) -> int:
    if args.wake_command == "training-status":
        checks = wake_training_checks(python=args.training_python)
        print(render_wake_training_checks(checks))
        return 0 if all(check.ok for check in checks) else 1
    if args.wake_command == "test-audio":
        result = score_wake_audio(config, Path(args.wav), threshold=args.threshold)
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0 if result.hit else 1
    if args.wake_command == "calibrate":
        cfg = _audio_override_config(config, device=args.device, gain_db=args.gain_db)
        result = calibrate_wake(
            cfg,
            Path(args.wav),
            seconds=float(args.seconds),
            threshold=args.threshold,
            min_rms=float(args.min_rms),
            min_peak=int(args.min_peak),
        )
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0 if result.ok else 1
    if args.wake_command == "samples":
        return _wake_samples_command(args, config)
    raise ValueError(f"unknown wake command: {args.wake_command}")


def _add_wake_samples_capture_args(parser: argparse.ArgumentParser, include_prompt: bool = True) -> None:
    parser.add_argument("dir", help="Dataset directory.")
    parser.add_argument("--label", required=True, choices=["positive", "negative", "noise"], help="Sample label.")
    if include_prompt:
        parser.add_argument("--prompt", default="", help="Prompt text for this take.")
    parser.add_argument("--tag", default="", help="Short tag added to filenames and metadata.")
    parser.add_argument("--device", help="Temporary input device override by index or name.")
    parser.add_argument("--gain-db", type=float, help="Temporary input gain in decibels.")
    parser.add_argument("--min-rms", type=float, default=100.0, help="Minimum RMS before saving without --keep-weak.")
    parser.add_argument("--min-peak", type=int, default=500, help="Minimum peak before saving without --keep-weak.")
    parser.add_argument("--keep-weak", action="store_true", help="Save weak or silent takes instead of discarding them.")


def _wake_samples_command(args: argparse.Namespace, config: Config) -> int:
    dataset_dir = Path(args.dir)
    if args.samples_command == "init":
        init_dataset(dataset_dir)
        print(f"wake sample dataset: {dataset_dir}")
        print("created/preserved: positive/, negative/, noise/, metadata.jsonl")
        return 0
    if args.samples_command == "record":
        cfg = _audio_override_config(config, device=args.device, gain_db=args.gain_db)
        print(f"recording {args.label}; press Enter to save")
        try:
            result = capture_take(
                cfg,
                dataset_dir,
                label=args.label,
                prompt=args.prompt,
                tag=args.tag,
                mode="record",
                min_rms=float(args.min_rms),
                min_peak=int(args.min_peak),
                keep_weak=bool(args.keep_weak),
                stop_keys=("enter",),
                discard_keys=(),
            )
        except KeyboardInterrupt:
            print("\naborted; current take discarded")
            return 130
        return _print_capture_result(dataset_dir, result)
    if args.samples_command == "session":
        cfg = _audio_override_config(config, device=args.device, gain_db=args.gain_db)
        prompts = prompts_for_args(prompt=args.prompt, prompts_file=Path(args.prompts) if args.prompts else None, preset=args.preset or "")
        single_prompt = len(prompts) == 1
        if args.preset == "scarlett":
            print("preset scarlett prompts loaded")
            print("hard negatives to collect separately: " + ", ".join(HARD_NEGATIVE_PROMPTS))
            print("environmental negatives to collect separately: " + ", ".join(ENVIRONMENTAL_NEGATIVE_PROMPTS))
        controls = "space=save+next enter=save+done q=discard+quit ctrl-c=abort"
        print(controls)
        while True:
            take_number = next_take_index(dataset_dir, args.label)
            prompt = prompts[0] if single_prompt else prompts[(take_number - 1) % len(prompts)]

            def status(payload: dict[str, object]) -> None:
                print(
                    "\r"
                    f"label={args.label} take={take_number:04d} "
                    f"prompt={prompt or '-'} tag={args.tag or '-'} "
                    f"elapsed={float(payload['seconds']):.2f}s "
                    f"rms={float(payload['rms']):.1f} peak={int(payload['peak'])} "
                    f"{controls}",
                    end="",
                    flush=True,
                )

            try:
                result = capture_take(
                    cfg,
                    dataset_dir,
                    label=args.label,
                    prompt=prompt,
                    tag=args.tag,
                    mode="session",
                    command="wake samples session",
                    min_rms=float(args.min_rms),
                    min_peak=int(args.min_peak),
                    keep_weak=bool(args.keep_weak),
                    stop_keys=("space", "enter"),
                    discard_keys=("q",),
                    status=status,
                )
            except KeyboardInterrupt:
                print("\naborted; current take discarded")
                return 130
            print()
            _print_capture_result(dataset_dir, result)
            if result.action in {"saved_done", "discard_quit"}:
                return 0
            if result.action == "weak_discarded":
                continue
    if args.samples_command == "list":
        summary = summarize_dataset(dataset_dir)
        if args.json:
            print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
        else:
            print(render_dataset_summary(summary))
        return 0
    if args.samples_command == "score":
        summary = score_dataset(
            config,
            dataset_dir,
            model=args.model,
            threshold=args.threshold,
            label=args.label,
            write_receipts=not bool(args.no_receipt),
            scores_path=Path(args.scores_path) if args.scores_path else None,
        )
        if args.json:
            print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
        else:
            print(render_score_summary(summary))
        return 0
    if args.samples_command == "synthetic-msd":
        prompts: list[str] = []
        for prompts_file in args.prompts:
            prompts.extend(read_prompts_file(Path(prompts_file)))
        progress_every = max(0, int(args.progress_every))

        def status(done: int, total: int, path: Path) -> None:
            if progress_every and (done == 1 or done == total or done % progress_every == 0):
                print(f"generated {done}/{total}: {path}", flush=True)

        summary = generate_synthetic_msd_samples(
            dataset_dir,
            prompts=prompts,
            count=int(args.count),
            tag=str(args.tag),
            voices=split_csv_values(str(args.voices)),
            languages=split_csv_values(str(args.languages)),
            instructs=[str(item).strip() for item in args.instructs if str(item).strip()],
            model=str(args.model),
            speed=args.speed,
            ttl=args.ttl,
            msd_bin=str(args.msd_bin),
            converter=args.converter,
            dry_run=bool(args.dry_run),
            status=None if args.json else status,
        )
        if args.json:
            print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
        else:
            print(render_synthetic_msd_summary(summary))
        return 0
    raise ValueError(f"unknown wake samples command: {args.samples_command}")


def _print_capture_result(dataset_dir: Path, result) -> int:
    if result.take is None:
        print(result.reason or result.action)
        return 1 if result.action == "weak_discarded" else 0
    rel = result.take.path.relative_to(dataset_dir)
    weak = " weak" if result.take.weak else ""
    print(
        f"saved{weak}: {rel} "
        f"({result.take.seconds:.2f}s rms={result.take.rms:.1f} peak={result.take.peak} clipped={result.take.clipped_ratio:.3f})"
    )
    return 0


def _audio_override_config(config: Config, device: str | None = None, gain_db: float | None = None) -> Config:
    audio: dict[str, Any] = {}
    if device:
        audio["device"] = device
    if gain_db is not None:
        audio["input_gain_db"] = gain_db
    return config.with_overrides({"audio": audio}) if audio else config


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
    if getattr(args, "fast", False):
        codex["fast"] = True
    if getattr(args, "effort", None):
        codex["effort"] = args.effort
    if getattr(args, "jsonl", False):
        ui["mode"] = "jsonl"
    if getattr(args, "quiet", False):
        ui["mode"] = "quiet"
    if getattr(args, "show_partials", False):
        ui["show_partial_transcripts"] = True
    if getattr(args, "timestamp_opacity", None) is not None:
        ui["timestamp_opacity"] = float(args.timestamp_opacity)
    if getattr(args, "plain_labels", False):
        ui["bold_labels"] = False
    if getattr(args, "show_events", None):
        ui["visible_events"] = split_csv_values(str(args.show_events))
    if getattr(args, "hide_events", None):
        ui["hidden_events"] = split_csv_values(str(args.hide_events))
    if codex:
        overrides["codex"] = codex
    if ui:
        overrides["ui"] = ui
    return overrides


if __name__ == "__main__":
    raise SystemExit(main())
