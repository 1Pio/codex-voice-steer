from __future__ import annotations

from types import SimpleNamespace

from codex_voice_steer.audio import AudioDevice, AudioLevel
from codex_voice_steer import cli
from codex_voice_steer.cli import _payload, _render_audio_devices, _render_compact_status, build_parser
from codex_voice_steer.config import load_config


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
    args = build_parser().parse_args(["wake", "training-status", "--python", "/tmp/train/bin/python"])
    assert args.command == "wake"
    assert args.wake_command == "training-status"
    assert args.training_python == "/tmp/train/bin/python"


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
            "--gain-db",
            "12",
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
    assert args.gain_db == 12
    assert args.threshold == 0.45
    assert args.min_rms == 500
    assert args.min_peak == 2000


def test_wake_samples_commands_parse() -> None:
    parser = build_parser()
    init = parser.parse_args(["wake", "samples", "init", "/tmp/samples"])
    record = parser.parse_args(
        [
            "wake",
            "samples",
            "record",
            "/tmp/samples",
            "--label",
            "positive",
            "--prompt",
            "scarlett",
            "--tag",
            "near",
            "--device",
            "0",
            "--gain-db",
            "6",
            "--min-rms",
            "50",
            "--min-peak",
            "200",
            "--keep-weak",
        ]
    )
    session = parser.parse_args(["wake", "samples", "session", "/tmp/samples", "--label", "negative", "--preset", "scarlett"])
    listing = parser.parse_args(["wake", "samples", "list", "/tmp/samples", "--json"])
    score = parser.parse_args(
        [
            "wake",
            "samples",
            "score",
            "/tmp/samples",
            "--model",
            "models/wake/scarlett.onnx",
            "--threshold",
            "0.4",
            "--label",
            "positive",
            "--no-receipt",
            "--scores-path",
            "/tmp/scores.jsonl",
            "--json",
        ]
    )
    synthetic = parser.parse_args(
        [
            "wake",
            "samples",
            "synthetic-msd",
            "/tmp/synthetic",
            "--prompts",
            "tools/prompts/wake-negative-hard.txt",
            "--prompts",
            "tools/prompts/wake-negative-normal.txt",
            "--tag",
            "synthetic-hard-negative",
            "--count",
            "300",
            "--voices",
            "Aiden,Ryan",
            "--languages",
            "English,German",
            "--instructs",
            "neutral",
            "fast",
            "--model",
            "cv-test",
            "--speed",
            "1.05",
            "--ttl",
            "3",
            "--progress-every",
            "10",
            "--dry-run",
            "--json",
        ]
    )

    assert init.wake_command == "samples"
    assert init.samples_command == "init"
    assert record.samples_command == "record"
    assert record.label == "positive"
    assert record.prompt == "scarlett"
    assert record.tag == "near"
    assert record.device == "0"
    assert record.gain_db == 6
    assert record.min_rms == 50
    assert record.min_peak == 200
    assert record.keep_weak is True
    assert session.samples_command == "session"
    assert session.preset == "scarlett"
    assert listing.samples_command == "list"
    assert listing.json is True
    assert score.samples_command == "score"
    assert score.model == "models/wake/scarlett.onnx"
    assert score.threshold == 0.4
    assert score.label == "positive"
    assert score.no_receipt is True
    assert score.scores_path == "/tmp/scores.jsonl"
    assert score.json is True
    assert synthetic.samples_command == "synthetic-msd"
    assert synthetic.dir == "/tmp/synthetic"
    assert synthetic.prompts == ["tools/prompts/wake-negative-hard.txt", "tools/prompts/wake-negative-normal.txt"]
    assert synthetic.tag == "synthetic-hard-negative"
    assert synthetic.count == 300
    assert synthetic.voices == "Aiden,Ryan"
    assert synthetic.languages == "English,German"
    assert synthetic.instructs == ["neutral", "fast"]
    assert synthetic.model == "cv-test"
    assert synthetic.speed == 1.05
    assert synthetic.ttl == 3
    assert synthetic.progress_every == 10
    assert synthetic.dry_run is True
    assert synthetic.json is True


def test_voice_test_audio_parses() -> None:
    args = build_parser().parse_args(["voice", "test-audio", "/tmp/turn.wav", "--send"])
    assert args.command == "voice"
    assert args.voice_command == "test-audio"
    assert args.wav == "/tmp/turn.wav"
    assert args.send is True


def test_audio_devices_parses() -> None:
    args = build_parser().parse_args(["audio", "devices", "--json", "--kind", "output"])
    assert args.command == "audio"
    assert args.audio_command == "devices"
    assert args.json is True
    assert args.kind == "output"


def test_audio_record_parses() -> None:
    args = build_parser().parse_args(["audio", "record", "/tmp/in.wav", "--seconds", "1.5", "--device", "2", "--gain-db", "9", "--json"])
    assert args.command == "audio"
    assert args.audio_command == "record"
    assert args.wav == "/tmp/in.wav"
    assert args.seconds == 1.5
    assert args.device == "2"
    assert args.gain_db == 9
    assert args.json is True


def test_audio_loopback_test_parses() -> None:
    args = build_parser().parse_args(
        [
            "audio",
            "loopback-test",
            "/tmp/source.wav",
            "/tmp/captured.wav",
            "--seconds",
            "1.5",
            "--device",
            "2",
            "--output-device",
            "3",
            "--gain-db",
            "6",
            "--json",
        ]
    )
    assert args.command == "audio"
    assert args.audio_command == "loopback-test"
    assert args.source_wav == "/tmp/source.wav"
    assert args.captured_wav == "/tmp/captured.wav"
    assert args.seconds == 1.5
    assert args.device == "2"
    assert args.output_device == "3"
    assert args.gain_db == 6
    assert args.json is True


def test_audio_play_parses() -> None:
    args = build_parser().parse_args(["audio", "play", "/tmp/source.wav", "--output-device", "3", "--json"])
    assert args.command == "audio"
    assert args.audio_command == "play"
    assert args.source_wav == "/tmp/source.wav"
    assert args.output_device == "3"
    assert args.json is True


def test_audio_meter_parses() -> None:
    args = build_parser().parse_args(["audio", "meter", "--seconds", "2", "--interval-ms", "250", "--device", "0", "--gain-db", "6", "--jsonl"])
    assert args.command == "audio"
    assert args.audio_command == "meter"
    assert args.seconds == 2
    assert args.interval_ms == 250
    assert args.device == "0"
    assert args.gain_db == 6
    assert args.jsonl is True


def test_config_unset_parses() -> None:
    args = build_parser().parse_args(["config", "unset", "audio.devices"])
    assert args.command == "config"
    assert args.config_command == "unset"
    assert args.key == "audio.devices"


def test_daemon_payload_includes_cli_overrides() -> None:
    args = build_parser().parse_args(["--cwd", "/tmp/cxv-cwd", "--model", "gpt-test", "text", "hello"])
    payload = _payload(args, "text", text="hello")
    assert payload["overrides"] == {"codex": {"cwd": "/tmp/cxv-cwd", "model": "gpt-test"}}


def test_daemon_payload_includes_latency_overrides() -> None:
    args = build_parser().parse_args(["--fast", "--effort", "minimal", "listen"])
    payload = _payload(args, "listen")
    assert payload["overrides"] == {"codex": {"fast": True, "effort": "minimal"}}


def test_ui_mode_flags_parse_as_overrides() -> None:
    jsonl = _payload(build_parser().parse_args(["--jsonl", "listen"]), "listen")
    quiet = _payload(build_parser().parse_args(["--quiet", "--show-partials", "listen"]), "listen")
    assert jsonl["overrides"] == {"ui": {"mode": "jsonl"}}
    assert quiet["overrides"] == {"ui": {"mode": "quiet", "show_partial_transcripts": True}}


def test_interactive_status_display_flags_parse_as_overrides() -> None:
    payload = _payload(
        build_parser().parse_args(
            [
                "--timestamp-opacity",
                "0.45",
                "--plain-labels",
                "--show-events",
                "wake_detected,sent,codex_msd_started",
                "--hide-events",
                "turn_started",
                "listen",
            ]
        ),
        "listen",
    )
    assert payload["overrides"] == {
        "ui": {
            "timestamp_opacity": 0.45,
            "bold_labels": False,
            "visible_events": ["wake_detected", "sent", "codex_msd_started"],
            "hidden_events": ["turn_started"],
        }
    }


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
                    {"event": "old", "transcript": "ignore me", "ts": 100.0},
                    {"event": "stt_final", "transcript": "hello " * 40, "ts": 200.0},
                ],
            },
        },
        event_limit=1,
    )
    assert "cxv daemon: running" in output
    assert "listening: yes" in output
    assert "queued inputs: 1" in output
    assert "at=" in output
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
    assert '"MacBook Pro Microphone"' in output


def test_render_output_devices_uses_loopback_hint() -> None:
    output = _render_audio_devices(
        [
            AudioDevice(index=3, name="MacBook Pro Speakers", max_output_channels=2, is_default=True),
        ],
        kind="output",
    )
    assert "3: MacBook Pro Speakers (2 output channel(s)) *" in output
    assert "loopback-test --output-device" in output


def test_audio_meter_warns_when_stream_is_silent(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(
        cli,
        "input_levels",
        lambda *_args, **_kwargs: [
            AudioLevel(
                elapsed_sec=0.5,
                rms=0.0,
                peak=0,
                samples=8000,
                device="0",
                device_name="MacBook Pro Microphone",
                gain_db=0.0,
                clipped_samples=0,
                clipped_ratio=0.0,
            )
        ],
    )
    args = SimpleNamespace(audio_command="meter", device=None, gain_db=None, seconds=0.5, interval_ms=500, jsonl=False)

    result = cli._audio_command(args, load_config(path=tmp_path / "missing.toml"))

    output = capsys.readouterr().out
    assert result == 0
    assert "device=0 (MacBook Pro Microphone)" in output
    assert "captured digital silence" in output


def test_doctor_returns_nonzero_when_any_check_blocks(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(cli, "run_doctor", lambda _config: [SimpleNamespace(ok=True), SimpleNamespace(ok=False)])
    monkeypatch.setattr(cli, "render_doctor", lambda _checks: "cxv doctor\nblocked test: no")

    result = cli.dispatch(build_parser().parse_args(["doctor"]), load_config(path=tmp_path / "missing.toml"))

    assert result == 1
    assert "blocked test" in capsys.readouterr().out
