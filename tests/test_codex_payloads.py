from __future__ import annotations

from codex_voice_steer.codex_app_server import CodexAppServer
from codex_voice_steer.config import load_config


def test_thread_start_injects_developer_instructions(tmp_path) -> None:
    cfg = load_config(path=tmp_path / "missing.toml")
    bridge = CodexAppServer(cfg)
    params = bridge._thread_start_params()
    assert params["developerInstructions"]
    assert params["approvalPolicy"] == "on-request"
    assert params["approvalsReviewer"] == "auto_review"
    assert params["config"]["default_permissions"] == ":workspace"


def test_text_input_contains_voice_metadata(tmp_path) -> None:
    cfg = load_config(path=tmp_path / "missing.toml")
    bridge = CodexAppServer(cfg)
    payload = bridge._text_input("check status")
    assert payload["type"] == "text"
    assert "wake=scarlett" in payload["text"]
    assert "check status" in payload["text"]
