from __future__ import annotations

from codex_voice_steer.cli import build_parser


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
