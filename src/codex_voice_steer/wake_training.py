from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingCheck:
    name: str
    ok: bool
    detail: str


TRAINER_NAME = "live" + "kit"
TRAINER_COMMAND = TRAINER_NAME + "-wakeword"
DEFAULT_CACHE_ROOT = Path.home() / "Documents" / "cxv-wake-samples" / (TRAINER_COMMAND + "-cache-v1")
ACAV_FEATURES = "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"


def wake_training_checks(
    *,
    python: str | Path | None = None,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    repo_root: Path | None = None,
) -> list[TrainingCheck]:
    root = repo_root or Path.cwd()
    training_python = _training_python(python, cache_root=cache_root)
    trainer_cli = training_python.parent / TRAINER_COMMAND

    checks = [
        _path_check("training python", training_python),
        _path_check("wakeword trainer CLI", trainer_cli),
        _probe_trainer_cli(trainer_cli),
        _path_check("training config", root / "tools" / TRAINER_COMMAND / "scarlett.yaml"),
        _dir_check("background cache", cache_root / "data" / "backgrounds"),
        _dir_check("RIR cache", cache_root / "data" / "rirs"),
        _path_check("ACAV features", cache_root / "data" / "features" / ACAV_FEATURES),
        _path_check("active output model", cache_root / "output" / "scarlett" / "scarlett.onnx"),
        _path_check("bundled wake model", root / "models" / "wake" / "scarlett.onnx"),
        _msd_check(),
    ]
    return checks


def render_wake_training_checks(checks: list[TrainingCheck]) -> str:
    lines = ["cxv wake training readiness"]
    for check in checks:
        mark = "ok" if check.ok else "blocked"
        lines.append(f"{mark:7} {check.name}: {check.detail}")
    return "\n".join(lines)


def _training_python(python: str | Path | None, *, cache_root: Path) -> Path:
    if python is not None:
        return Path(python).expanduser()
    env_python = os.environ.get("CXV_WAKE_TRAINING_PYTHON")
    if env_python:
        return Path(env_python).expanduser()
    return cache_root / "venv" / "bin" / "python"


def _probe_trainer_cli(trainer_cli: Path) -> TrainingCheck:
    if not trainer_cli.exists():
        return TrainingCheck("wakeword trainer help", False, f"{trainer_cli} missing")
    try:
        proc = subprocess.run(
            [str(trainer_cli), "--help"],
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return TrainingCheck("wakeword trainer help", False, "timed out after 30 seconds")
    if proc.returncode == 0:
        return TrainingCheck("wakeword trainer help", True, "help command works")
    return TrainingCheck("wakeword trainer help", False, _last_line(proc.stderr or proc.stdout))


def _path_check(name: str, path: Path) -> TrainingCheck:
    return TrainingCheck(name, path.exists(), str(path if path.exists() else path) + (" exists" if path.exists() else " missing"))


def _dir_check(name: str, path: Path) -> TrainingCheck:
    return TrainingCheck(name, path.is_dir(), str(path) + (" exists" if path.is_dir() else " missing"))


def _msd_check() -> TrainingCheck:
    msd_path = shutil.which("msd")
    return TrainingCheck("msd optional", True, msd_path or "msd not on PATH; only needed for synthetic generation")


def _last_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else "no output"
