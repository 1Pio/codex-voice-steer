from __future__ import annotations

from types import SimpleNamespace

from codex_voice_steer import doctor
from codex_voice_steer.audio import AudioReadiness
from codex_voice_steer.config import load_config


def test_doctor_blocks_when_msd_is_required_but_missing(monkeypatch, tmp_path) -> None:
    cfg = load_config(
        overrides={"instructions": {"msd": {"enabled": True, "require_msd_on_path": True}}},
        path=tmp_path / "missing.toml",
    )
    _stub_runtime_checks(monkeypatch)
    monkeypatch.setattr(doctor.shutil, "which", lambda command: None if command == "msd" else f"/bin/{command}")

    checks = doctor.run_doctor(cfg)

    msd = next(check for check in checks if check.name == "msd required")
    assert msd.ok is False
    assert "required by config" in msd.detail


def test_doctor_keeps_msd_optional_by_default(monkeypatch, tmp_path) -> None:
    cfg = load_config(path=tmp_path / "missing.toml")
    _stub_runtime_checks(monkeypatch)
    monkeypatch.setattr(doctor.shutil, "which", lambda command: None if command == "msd" else f"/bin/{command}")

    checks = doctor.run_doctor(cfg)

    msd = next(check for check in checks if check.name == "msd optional")
    assert msd.ok is True
    assert "optional only" in msd.detail


def _stub_runtime_checks(monkeypatch) -> None:
    monkeypatch.setattr(doctor, "audio_readiness", lambda _config, probe_stream=False: AudioReadiness(True, "audio ok"))
    monkeypatch.setattr(doctor, "vad_readiness", lambda: SimpleNamespace(ok=True, reason="vad ok"))
    monkeypatch.setattr(doctor, "wake_readiness", lambda _config, repo_root=None: SimpleNamespace(ok=True, reason="wake ok"))
    monkeypatch.setattr(doctor.subprocess, "run", lambda *args, **kwargs: type("Proc", (), {"returncode": 0, "stderr": ""})())
