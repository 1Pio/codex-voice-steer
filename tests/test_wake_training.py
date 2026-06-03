from __future__ import annotations

import subprocess

from codex_voice_steer import wake_training
from codex_voice_steer.wake_training import TrainingCheck, render_wake_training_checks, wake_training_checks


def test_render_wake_training_checks() -> None:
    text = render_wake_training_checks([TrainingCheck("thing", False, "missing")])
    assert "cxv wake training readiness" in text
    assert "blocked thing: missing" in text


def test_wake_training_checks_use_durable_livekit_cache(tmp_path, monkeypatch) -> None:
    cache = tmp_path / "cache"
    repo = tmp_path / "repo"
    training_python = cache / "venv" / "bin" / "python"
    trainer_cli = cache / "venv" / "bin" / wake_training.TRAINER_COMMAND
    acav = cache / "data" / "features" / wake_training.ACAV_FEATURES
    active_model = cache / "output" / "scarlett" / "scarlett.onnx"
    bundled_model = repo / "models" / "wake" / "scarlett.onnx"
    config = repo / "tools" / "livekit-wakeword" / "scarlett.yaml"

    for path in [training_python, trainer_cli, acav, active_model, bundled_model, config]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ok")
    (cache / "data" / "backgrounds").mkdir(parents=True)
    (cache / "data" / "rirs").mkdir(parents=True)

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout="Usage: wakeword trainer", stderr="")

    monkeypatch.setattr(wake_training.subprocess, "run", fake_run)
    monkeypatch.setattr(wake_training.shutil, "which", lambda name: "/tmp/msd" if name == "msd" else None)

    checks = wake_training_checks(cache_root=cache, repo_root=repo)

    assert all(check.ok for check in checks)
    assert next(check for check in checks if check.name == "training python").detail.endswith("python exists")
    assert next(check for check in checks if check.name == "wakeword trainer help").detail == "help command works"


def test_wake_training_checks_accept_python_override(tmp_path, monkeypatch) -> None:
    override = tmp_path / "custom" / "bin" / "python"
    override.parent.mkdir(parents=True)
    override.write_text("ok")
    monkeypatch.setenv("CXV_WAKE_TRAINING_PYTHON", "/env/python")

    checks = wake_training_checks(python=override, cache_root=tmp_path / "cache", repo_root=tmp_path / "repo")

    assert next(check for check in checks if check.name == "training python").detail == f"{override} exists"
