from __future__ import annotations

import tomllib

import pytest

from codex_voice_steer.config import DEFAULT_CONFIG, default_config_toml, load_config, set_config_value, unknown_config_keys, unset_config_value


def test_default_config_has_no_version_key() -> None:
    parsed = tomllib.loads(default_config_toml())
    assert "version" not in parsed
    assert parsed["wake"]["word"] == "scarlett"
    assert parsed["audio"]["input_gain_db"] == 0.0
    assert parsed["stt"]["engine"] == "macparakeet"
    assert parsed["codex"]["permission_profile"] == ":workspace"
    assert "permissions" not in parsed["codex"]


def test_user_config_overrides_defaults(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[codex]\nmodel = "gpt-5"\n')
    cfg = load_config(path=path)
    assert cfg.get("codex.model") == "gpt-5"
    assert cfg.get("wake.word") == DEFAULT_CONFIG["wake"]["word"]


def test_version_key_is_rejected(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('version = "1"\n')
    with pytest.raises(ValueError):
        load_config(path=path)


def test_config_set_writes_dotted_value(tmp_path) -> None:
    path = tmp_path / "config.toml"
    set_config_value("instructions.msd.enabled", "true", path=path)
    cfg = load_config(path=path)
    assert cfg.get("instructions.msd.enabled") is True


def test_config_set_rejects_unknown_keys(tmp_path) -> None:
    path = tmp_path / "config.toml"
    with pytest.raises(ValueError, match="audio\\.device"):
        set_config_value("audio.devices", "0", path=path)


def test_config_unset_removes_accidental_key(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("[audio]\ndevices = 0\ndevice = \"default\"\n")
    unset_config_value("audio.devices", path=path)
    cfg = load_config(path=path)
    assert cfg.get("audio.devices") is None
    assert unknown_config_keys(cfg.data) == []


def test_config_set_maps_legacy_permissions_key(tmp_path) -> None:
    path = tmp_path / "config.toml"
    set_config_value("codex.permissions", '":read-only"', path=path)
    parsed = tomllib.loads(path.read_text())
    assert parsed["codex"]["permission_profile"] == ":read-only"
    assert "permissions" not in parsed["codex"]


def test_legacy_permissions_key_is_removed_from_resolved_config(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[codex]\npermissions = ":read-only"\n')
    cfg = load_config(path=path)
    assert cfg.get("codex.permission_profile") == ":read-only"
    assert "permissions" not in cfg.data["codex"]
