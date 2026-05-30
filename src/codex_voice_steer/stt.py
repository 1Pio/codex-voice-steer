from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import Config


@dataclass(frozen=True)
class SttResult:
    text: str
    command: list[str]


class MacParakeetStt:
    def __init__(self, config: Config) -> None:
        self.config = config

    def available(self) -> bool:
        return shutil.which(self.command) is not None

    @property
    def command(self) -> str:
        return str(self.config.get("stt.macparakeet.command", "macparakeet-cli"))

    def build_command(self, wav_path: Path) -> list[str]:
        cmd = [
            self.command,
            "transcribe",
            str(wav_path),
            "--format",
            str(self.config.get("stt.format", "text")),
            "--mode",
            str(self.config.get("stt.mode", "clean")),
            "--engine",
            str(self.config.get("stt.macparakeet.engine", "parakeet")),
            "--speaker-detection",
            str(self.config.get("stt.macparakeet.speaker_detection", "off")),
        ]
        database = str(self.config.get("stt.macparakeet.database", ""))
        if database:
            cmd.extend(["--database", database])
        if self.config.get("stt.no_history", True):
            cmd.append("--no-history")
        return cmd

    def transcribe(self, wav_path: Path, timeout_sec: int = 120) -> SttResult:
        env = os.environ.copy()
        if self.config.get("stt.telemetry", False) is False:
            env["MACPARAKEET_TELEMETRY"] = "0"
        cmd = self.build_command(wav_path)
        proc = subprocess.run(cmd, text=True, capture_output=True, env=env, timeout=timeout_sec, check=False)
        if proc.returncode != 0:
            stderr = proc.stderr.strip() or proc.stdout.strip()
            raise RuntimeError(f"macparakeet-cli failed with exit {proc.returncode}: {stderr}")
        return SttResult(text=clean_macparakeet_text(proc.stdout), command=cmd)


def clean_macparakeet_text(output: str) -> str:
    body = output.split("--- Word Timestamps ---", 1)[0]
    lines = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        if stripped.startswith("Transcribing ") or stripped.startswith("File:") or stripped.startswith("Duration:"):
            continue
        lines.append(line.rstrip())
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines).strip()
