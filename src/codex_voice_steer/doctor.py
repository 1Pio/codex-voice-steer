from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .audio import audio_readiness
from .config import Config, config_key_suggestion, unknown_config_keys
from .vad import vad_readiness
from .wake import wake_readiness


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def run_doctor(config: Config, repo_root: Path | None = None) -> list[Check]:
    checks: list[Check] = []
    unknown_keys = unknown_config_keys(config.data)
    if unknown_keys:
        details = []
        for key in unknown_keys:
            suggestion = config_key_suggestion(key)
            details.append(f"{key} (did you mean {suggestion}?)" if suggestion else key)
        checks.append(Check("config", False, "unknown key(s): " + ", ".join(details)))
    else:
        checks.append(Check("config", True, f"loaded {config.path}"))
    codex_path = shutil.which("codex")
    macparakeet_command = str(config.get("stt.macparakeet.command", "macparakeet-cli"))
    macparakeet_path = shutil.which(macparakeet_command)
    checks.append(Check("codex", codex_path is not None, codex_path or "codex not on PATH"))
    checks.append(Check("macparakeet", macparakeet_path is not None, macparakeet_path or "macparakeet-cli not on PATH"))
    msd_path = shutil.which("msd")
    msd_required = bool(config.get("instructions.msd.enabled", False)) and bool(config.get("instructions.msd.require_msd_on_path", False))
    msd_detail = msd_path or ("msd not on PATH; required by config" if msd_required else "msd not on PATH; optional only")
    checks.append(Check("msd required" if msd_required else "msd optional", msd_path is not None if msd_required else True, msd_detail))
    audio = audio_readiness(config, probe_stream=True)
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
