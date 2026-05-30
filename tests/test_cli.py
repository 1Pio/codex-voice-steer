from __future__ import annotations

from codex_voice_steer.audio import AudioDevice
from codex_voice_steer.cli import _payload, _render_audio_devices, _render_compact_status, build_parser


def test_core_commands_parse() -> None:
    parser = build_parser()
    for command in ["up", "down", "status", "listen", "pause", "models", "doctor", "serve"]:
        args = parser.parse_args([command])
        assert args.command == command


def test_ttt_alias_parses_text() -> None:
    args = build_parser().parse_args(["ttt", "hello", "codex"])
    assert args.command == "ttt"
    assert args.text == ["hello", "codex"]


def test_bind_parses_thread_and_cwd() -> None:
    args = build_parser().parse_args(["bind", "--thread", "thr_123", "--cwd", "/tmp"])
    assert args.thread == "thr_123"
    assert args.cwd == "/tmp"


def test_wake_training_status_parses() -> None:
    args = build_parser().parse_args(["wake", "training-status"])
    assert args.command == "wake"
    assert args.wake_command == "training-status"


def test_wake_test_audio_parses() -> None:
    args = build_parser().parse_args(["wake", "test-audio", "/tmp/scarlett.wav", "--threshold", "0.4"])
    assert args.command == "wake"
    assert args.wake_command == "test-audio"
    assert args.wav == "/tmp/scarlett.wav"
    assert args.threshold == 0.4


def test_wake_calibrate_parses() -> None:
    args = build_parser().parse_args(
        [
            "wake",
            "calibrate",
            "/tmp/scarlett-live.wav",
            "--seconds",
            "3",
            "--device",
            "0",
            "--threshold",
            "0.45",
            "--min-rms",
            "500",
            "--min-peak",
            "2000",
        ]
    )
    assert args.command == "wake"
    assert args.wake_command == "calibrate"
    assert args.wav == "/tmp/scarlett-live.wav"
    assert args.seconds == 3
    assert args.device == "0"
    assert args.threshold == 0.45
    assert args.min_rms == 500
    assert args.min_peak == 2000


def test_voice_test_audio_parses() -> None:
    args = build_parser().parse_args(["voice", "test-audio", "/tmp/turn.wav", "--send"])
    assert args.command == "voice"
    assert args.voice_command == "test-audio"
    assert args.wav == "/tmp/turn.wav"
    assert args.send is True


def test_audio_devices_parses() -> None:
    args = build_parser().parse_args(["audio", "devices", "--json"])
    assert args.command == "audio"
    assert args.audio_command == "devices"
    assert args.json is True


def test_audio_record_parses() -> None:
    args = build_parser().parse_args(["audio", "record", "/tmp/in.wav", "--seconds", "1.5", "--device", "2", "--json"])
    assert args.command == "audio"
    assert args.audio_command == "record"
    assert args.wav == "/tmp/in.wav"
    assert args.seconds == 1.5
    assert args.device == "2"
    assert args.json is True


def test_audio_meter_parses() -> None:
    args = build_parser().parse_args(["audio", "meter", "--seconds", "2", "--interval-ms", "250", "--device", "0", "--jsonl"])
    assert args.command == "audio"
    assert args.audio_command == "meter"
    assert args.seconds == 2
    assert args.interval_ms == 250
    assert args.device == "0"
    assert args.jsonl is True


def test_daemon_payload_includes_cli_overrides() -> None:
    args = build_parser().parse_args(["--cwd", "/tmp/cxv-cwd", "--model", "gpt-test", "text", "hello"])
    payload = _payload(args, "text", text="hello")
    assert payload["overrides"] == {"codex": {"cwd": "/tmp/cxv-cwd", "model": "gpt-test"}}


def test_ui_mode_flags_parse_as_overrides() -> None:
    jsonl = _payload(build_parser().parse_args(["--jsonl", "listen"]), "listen")
    quiet = _payload(build_parser().parse_args(["--quiet", "--show-partials", "listen"]), "listen")
    assert jsonl["overrides"] == {"ui": {"mode": "jsonl"}}
    assert quiet["overrides"] == {"ui": {"mode": "quiet", "show_partial_transcripts": True}}


def test_status_flags_parse() -> None:
    args = build_parser().parse_args(["status", "--json", "--events", "2"])
    assert args.command == "status"
    assert args.json is True
    assert args.events == 2


def test_compact_status_does_not_dump_full_event_history() -> None:
    output = _render_compact_status(
        {
            "ok": True,
            "state": {
                "listening": True,
                "thread_id": "thread_1",
                "session_id": "session_1",
                "active_turn_id": "turn_1",
                "queued_inputs": ["next"],
                "cwd": "/tmp/cxv",
                "events": [
                    {"event": "old", "transcript": "ignore me"},
                    {"event": "stt_final", "transcript": "hello " * 40},
                ],
            },
        },
        event_limit=1,
    )
    assert "cxv daemon: running" in output
    assert "listening: yes" in output
    assert "queued inputs: 1" in output
    assert "old" not in output
    assert "hello " * 20 not in output


def test_render_audio_devices_marks_default_and_config_hint() -> None:
    output = _render_audio_devices(
        [
            AudioDevice(index=1, name="Loopback Input", max_input_channels=2),
            AudioDevice(index=2, name="MacBook Pro Microphone", max_input_channels=1, is_default=True),
        ]
    )
    assert "1: Loopback Input (2 input channel(s))" in output
    assert "2: MacBook Pro Microphone (1 input channel(s)) *" in output
    assert "cxv config set audio.device" in output
