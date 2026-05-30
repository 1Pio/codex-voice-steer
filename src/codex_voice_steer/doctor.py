from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .audio import audio_readiness
from .config import Config
from .vad import vad_readiness
from .wake import wake_readiness


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def run_doctor(config: Config, repo_root: Path | None = None) -> list[Check]:
    checks: list[Check] = []
    checks.append(Check("codex", shutil.which("codex") is not None, shutil.which("codex") or "codex not on PATH"))
    checks.append(Check("macparakeet", shutil.which(str(config.get("stt.macparakeet.command", "macparakeet-cli"))) is not None, shutil.which(str(config.get("stt.macparakeet.command", "macparakeet-cli"))) or "macparakeet-cli not on PATH"))
    checks.append(Check("msd optional", True, shutil.which("msd") or "msd not on PATH; optional only"))
    audio = audio_readiness()
    checks.append(Check("microphone adapter", audio.ok, audio.reason))
    vad = vad_readiness()
    checks.append(Check("silero vad", vad.ok, vad.reason))
    wake = wake_readiness(config, repo_root=repo_root)
    checks.append(Check("scarlett wake model", wake.ok, wake.reason))
    if shutil.which("codex"):
        proc = subprocess.run(["codex", "app-server", "--help"], text=True, capture_output=True, timeout=10, check=False)
        checks.append(Check("codex app-server", proc.returncode == 0 and "turn/start" not in proc.stderr, "help command returned exit " + str(proc.returncode)))
    return checks


def render_doctor(checks: list[Check]) -> str:
    lines = ["cxv doctor"]
    for check in checks:
        mark = "ok" if check.ok else "blocked"
        lines.append(f"{mark:7} {check.name}: {check.detail}")
    return "\n".join(lines)
