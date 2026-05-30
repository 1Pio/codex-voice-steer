from __future__ import annotations

from codex_voice_steer.vad import vad_readiness


def test_vad_readiness_returns_structured_result() -> None:
    result = vad_readiness()
    assert isinstance(result.ok, bool)
    assert result.reason
