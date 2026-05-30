from __future__ import annotations

from pathlib import Path

from codex_voice_steer.config import load_config
from codex_voice_steer.stt import MacParakeetStt


def test_macparakeet_command_matches_v1_default(tmp_path) -> None:
    cfg = load_config(path=tmp_path / "missing.toml")
    cmd = MacParakeetStt(cfg).build_command(Path("/tmp/segment.wav"))
    assert cmd == [
        "macparakeet-cli",
        "transcribe",
        "/tmp/segment.wav",
        "--format",
        "text",
        "--mode",
        "clean",
        "--engine",
        "parakeet",
        "--speaker-detection",
        "off",
        "--no-history",
    ]
