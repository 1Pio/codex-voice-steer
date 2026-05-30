from __future__ import annotations

import subprocess

from codex_voice_steer import wake_training


def test_openwakeword_train_probe_handles_timeout(monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=["python"], timeout=30)

    monkeypatch.setattr(wake_training.subprocess, "run", fake_run)
    check = wake_training._probe_openwakeword_train()
    assert check.ok is False
    assert "timed out" in check.detail


def test_piper_probe_handles_timeout(monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=["python"], timeout=30)

    monkeypatch.setattr(wake_training.subprocess, "run", fake_run)
    check = wake_training._probe_piper_sample_generator()
    assert check.ok is False
    assert "timed out" in check.detail
