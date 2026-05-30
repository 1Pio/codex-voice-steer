from __future__ import annotations

from pathlib import Path

from codex_voice_steer.config import load_config
from codex_voice_steer.stt import MacParakeetStt, clean_macparakeet_text


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


def test_macparakeet_text_cleanup_removes_cli_metadata() -> None:
    output = """Transcribing segment.wav with parakeet...

File: segment.wav
Duration: 0m 3s

115 smoke okay.

--- Word Timestamps ---
[0.96-1.60] 115 (99%)
"""
    assert clean_macparakeet_text(output) == "115 smoke okay."
