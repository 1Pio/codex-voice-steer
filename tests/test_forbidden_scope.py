from __future__ import annotations

from pathlib import Path


def test_no_forbidden_runtime_paths() -> None:
    forbidden = ["hermes", "launchd", "cron", "docker", "container", "livekit", "codex exec"]
    root = Path(__file__).resolve().parents[1] / "src"
    haystack = "\n".join(path.read_text().lower() for path in root.rglob("*.py"))
    for term in forbidden:
        assert term not in haystack
