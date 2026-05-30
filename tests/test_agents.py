from __future__ import annotations

from codex_voice_steer.agents import install_agent, list_agents, print_agent


def test_agents_are_available() -> None:
    listing = list_agents()
    assert "slim" in listing
    assert "msd" in listing
    assert "cxv-voice-slim" in print_agent("slim")


def test_install_agent(tmp_path) -> None:
    target = install_agent("slim", codex_home=tmp_path)
    assert target.exists()
    assert target.name == "cxv-voice-slim.toml"
