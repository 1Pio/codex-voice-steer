from __future__ import annotations

from codex_voice_steer.agents import agent_developer_instructions, install_agent, list_agents, print_agent


def test_agents_are_available() -> None:
    listing = list_agents()
    assert "slim" in listing
    assert "msd" in listing
    assert "cxv-voice-slim" in print_agent("slim")


def test_install_agent(tmp_path) -> None:
    target = install_agent("slim", codex_home=tmp_path)
    assert target.exists()
    assert target.name == "cxv-voice-slim.toml"


def test_bundled_agent_developer_instructions_can_be_resolved() -> None:
    instructions = agent_developer_instructions("cxv-voice-msd")
    assert "msd say" in instructions


def test_installed_custom_agent_developer_instructions_can_be_resolved(tmp_path) -> None:
    agent = tmp_path / "agents" / "custom.toml"
    agent.parent.mkdir()
    agent.write_text('name = "custom"\ndescription = "Custom"\ndeveloper_instructions = "Custom voice rules."\n')
    assert agent_developer_instructions("custom", codex_home=tmp_path) == "Custom voice rules."
