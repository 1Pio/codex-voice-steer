from __future__ import annotations

import subprocess
from pathlib import Path

from codex_voice_steer import wake_training


def test_livekit_probe_handles_timeout(tmp_path, monkeypatch) -> None:
    cli = tmp_path / wake_training.TRAINER_COMMAND
    cli.write_text("ok")

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=["python"], timeout=30)

    monkeypatch.setattr(wake_training.subprocess, "run", fake_run)
    check = wake_training._probe_trainer_cli(cli)
    assert check.ok is False
    assert "timed out" in check.detail


def test_livekit_probe_reports_missing_cli() -> None:
    check = wake_training._probe_trainer_cli(Path("/tmp/missing-wakeword-trainer"))
    assert check.ok is False
    assert "missing" in check.detail


def test_livekit_probe_reports_failed_help(tmp_path, monkeypatch) -> None:
    cli = tmp_path / wake_training.TRAINER_COMMAND
    cli.write_text("ok")

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=2, stdout="", stderr="bad help\nlast line")

    monkeypatch.setattr(wake_training.subprocess, "run", fake_run)
    check = wake_training._probe_trainer_cli(cli)
    assert check.ok is False
    assert check.detail == "last line"
