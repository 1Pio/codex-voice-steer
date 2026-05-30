from __future__ import annotations

from codex_voice_steer.wake_training import TrainingCheck, render_wake_training_checks


def test_render_wake_training_checks() -> None:
    text = render_wake_training_checks([TrainingCheck("thing", False, "missing")])
    assert "cxv wake training readiness" in text
    assert "blocked thing: missing" in text
