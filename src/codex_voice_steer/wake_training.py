from __future__ import annotations

import importlib.util
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingCheck:
    name: str
    ok: bool
    detail: str


TRAINING_PACKAGES = [
    "torchinfo",
    "torchmetrics",
    "yaml",
    "pronouncing",
    "torch_audiomentations",
    "speechbrain",
    "piper_sample_generator",
    "piper_train",
]


def wake_training_checks() -> list[TrainingCheck]:
    checks = [TrainingCheck("python", True, sys.executable)]
    for package in TRAINING_PACKAGES:
        spec = importlib.util.find_spec(package)
        checks.append(TrainingCheck(package, spec is not None, "importable" if spec is not None else "not importable"))
    checks.append(_probe_openwakeword_train())
    checks.append(_probe_piper_sample_generator())
    checks.append(_path_check("wake model output", Path("models/wake/scarlett.onnx")))
    return checks


def render_wake_training_checks(checks: list[TrainingCheck]) -> str:
    lines = ["cxv wake training readiness"]
    for check in checks:
        mark = "ok" if check.ok else "blocked"
        lines.append(f"{mark:7} {check.name}: {check.detail}")
    return "\n".join(lines)


def _probe_openwakeword_train() -> TrainingCheck:
    try:
        proc = subprocess.run(
            [sys.executable, "-c", "import openwakeword.train; print(openwakeword.train.__file__)"],
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return TrainingCheck("openwakeword.train import", False, "timed out after 30 seconds")
    if proc.returncode == 0:
        return TrainingCheck("openwakeword.train import", True, proc.stdout.strip())
    return TrainingCheck("openwakeword.train import", False, _last_line(proc.stderr or proc.stdout))


def _probe_piper_sample_generator() -> TrainingCheck:
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "piper_sample_generator", "--help"],
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return TrainingCheck("piper sample generator CLI", False, "timed out after 30 seconds")
    if proc.returncode == 0:
        return TrainingCheck("piper sample generator CLI", True, "help command works")
    return TrainingCheck("piper sample generator CLI", False, _last_line(proc.stderr or proc.stdout))


def _path_check(name: str, path: Path) -> TrainingCheck:
    return TrainingCheck(name, path.exists(), str(path if path.exists() else path) + (" exists" if path.exists() else " missing"))


def _last_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else "no output"
