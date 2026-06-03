from __future__ import annotations

import copy
import difflib
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import config_path, default_pid_path, default_socket_path, log_dir, state_db_path


DEFAULT_DEVELOPER_INSTRUCTIONS = """You are being controlled through codex-voice-steer, a local voice-to-Codex bridge.
Inputs may come from speech-to-text and may contain fragments, homophones, missing punctuation, or wake-word references.
Ask concise clarification when an utterance seems cut off or dangerously ambiguous.
"""

DEFAULT_MSD_INSTRUCTIONS = """When msd speech output is enabled, important user-visible communication should be spoken with `msd say`.
Keep spoken updates short and meaningful. Do not speak every tool call.
"""


DEFAULT_CONFIG: dict[str, Any] = {
    "ui": {
        "mode": "interactive",
        "show_timestamps": True,
        "show_wake_events": True,
        "show_vad_events": True,
        "show_partial_transcripts": False,
        "show_final_transcripts": True,
        "show_codex_visible_messages": True,
        "show_codex_tool_traces": True,
        "show_codex_reasoning": False,
        "show_status_line": True,
        "max_transcript_lines": 200,
    },
    "server": {
        "socket_path": str(default_socket_path()),
        "pid_path": str(default_pid_path()),
        "state_db": str(state_db_path()),
        "log_file": str(log_dir() / "cxv.log"),
        "idle_timeout_minutes": 0,
    },
    "audio": {
        "device": "default",
        "sample_rate": 16000,
        "channels": 1,
        "pre_roll_ms": 750,
        "post_wake_grace_ms": 250,
        "input_gain_db": 0.0,
    },
    "wake": {
        "enabled": True,
        "engine": "openwakeword",
        "word": "scarlett",
        "sensitivity": 0.5,
        "refractory_ms": 1200,
        "allow_barge_in": True,
        "model_path": "models/wake/scarlett.onnx",
    },
    "vad": {
        "engine": "silero",
        "speech_threshold": 0.5,
        "min_speech_ms": 180,
        "min_silence_ms": 450,
        "final_silence_ms": 900,
        "force_final_silence_ms": 3000,
        "max_utterance_sec": 45,
    },
    "endpointing": {
        "mode": "vad_plus_heuristics",
        "min_chars_to_send": 8,
        "ask_if_fragment": True,
        "fragment_prompt_policy": "send_to_agent",
        "trailing_fragment_words": ["and", "but", "or", "because", "so", "also", "then"],
    },
    "stt": {
        "engine": "macparakeet",
        "mode": "clean",
        "format": "text",
        "no_history": True,
        "telemetry": False,
        "macparakeet": {
            "command": "macparakeet-cli",
            "engine": "parakeet",
            "speaker_detection": "off",
            "database": "",
        },
        "mlx_whisper": {
            "model": "mlx-community/whisper-large-v3-turbo",
            "quality_model": "mlx-community/whisper-large-v3-mlx",
            "language": "auto",
            "local_agreement": 2,
            "partial_window_sec": 6,
            "max_window_sec": 14,
        },
    },
    "codex": {
        "app_server": "managed",
        "app_server_listen": "stdio://",
        "cwd": ".",
        "thread_id": "",
        "resume_thread_id": "",
        "create_thread_if_missing": True,
        "model": "gpt-5.5",
        "effort": "medium",
        "summary": "concise",
        "personality": "pragmatic",
        "fast": False,
        "agent": "",
        "permission_profile": ":workspace",
        "approval_policy": "on-request",
        "approvals_reviewer": "auto_review",
    },
    "instructions": {
        "mode": "inject",
        "developer_instructions": DEFAULT_DEVELOPER_INSTRUCTIONS,
        "msd": {
            "enabled": False,
            "require_msd_on_path": False,
            "spoken_acknowledgements": "brief",
            "spoken_status_updates": "important",
            "spoken_final_results": True,
            "developer_instructions": DEFAULT_MSD_INSTRUCTIONS,
        },
    },
    "delivery": {
        "when_idle": "start",
        "when_active": "steer",
        "when_not_steerable": "queue",
        "include_voice_metadata": True,
        "include_wake_word": True,
        "include_stt_diagnostics": False,
    },
    "commands": {
        "typed_input_command": "text",
        "typed_input_aliases": ["ttt"],
    },
    "agents": {
        "ship_agent_templates": True,
        "default_slim": "cxv-voice-slim",
        "default_msd": "cxv-voice-msd",
    },
}


@dataclass(frozen=True)
class Config:
    data: dict[str, Any]
    path: Path
    loaded: bool

    def get(self, dotted: str, default: Any = None) -> Any:
        current: Any = self.data
        for part in dotted.split("."):
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current

    def with_overrides(self, overrides: dict[str, Any] | None) -> "Config":
        if not overrides:
            return self
        return Config(data=deep_merge(self.data, overrides), path=self.path, loaded=self.loaded)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(overrides: dict[str, Any] | None = None, path: Path | None = None) -> Config:
    cfg_path = path or config_path()
    loaded = False
    user_cfg: dict[str, Any] = {}
    if cfg_path.exists():
        user_cfg = tomllib.loads(cfg_path.read_text())
        loaded = True
        if "version" in user_cfg:
            raise ValueError("cxv config must not contain a top-level version key")
        user_cfg = _normalize_user_config(user_cfg)
    data = deep_merge(DEFAULT_CONFIG, user_cfg)
    if overrides:
        data = deep_merge(data, overrides)
    return Config(data=data, path=cfg_path, loaded=loaded)


def default_config_toml() -> str:
    return _toml(DEFAULT_CONFIG)


def write_default_config(path: Path | None = None, force: bool = False) -> Path:
    cfg_path = path or config_path()
    if cfg_path.exists() and not force:
        return cfg_path
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(default_config_toml())
    return cfg_path


def set_config_value(dotted: str, value: str, path: Path | None = None) -> Path:
    cfg_path = path or config_path()
    cfg = load_config(path=cfg_path).data if cfg_path.exists() else copy.deepcopy(DEFAULT_CONFIG)
    dotted = _canonical_key(dotted)
    _require_known_config_key(dotted)
    parsed = parse_value(value)
    current = cfg
    parts = dotted.split(".")
    for part in parts[:-1]:
        next_value = current.setdefault(part, {})
        if not isinstance(next_value, dict):
            raise ValueError(f"{'.'.join(parts[:-1])} is not a table")
        current = next_value
    current[parts[-1]] = parsed
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(_toml(cfg))
    return cfg_path


def unset_config_value(dotted: str, path: Path | None = None) -> Path:
    cfg_path = path or config_path()
    if not cfg_path.exists():
        raise FileNotFoundError(f"config file does not exist: {cfg_path}")
    cfg = tomllib.loads(cfg_path.read_text())
    dotted = _canonical_key(dotted)
    parts = dotted.split(".")
    current: Any = cfg
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            raise ValueError(f"config key is not set: {dotted}")
        current = current[part]
    if not isinstance(current, dict) or parts[-1] not in current:
        raise ValueError(f"config key is not set: {dotted}")
    del current[parts[-1]]
    _prune_empty_tables(cfg, parts[:-1])
    cfg_path.write_text(_toml(cfg))
    return cfg_path


def unknown_config_keys(data: dict[str, Any]) -> list[str]:
    unknown: list[str] = []
    _collect_unknown_keys(data, DEFAULT_CONFIG, [], unknown)
    return unknown


def config_key_suggestion(dotted: str) -> str:
    matches = difflib.get_close_matches(_canonical_key(dotted), known_config_keys(), n=1, cutoff=0.75)
    return matches[0] if matches else ""


def known_config_keys() -> list[str]:
    keys: list[str] = []
    _collect_known_keys(DEFAULT_CONFIG, [], keys)
    return keys


def _normalize_user_config(config: dict[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(config)
    codex = config.get("codex")
    if isinstance(codex, dict) and "permissions" in codex and "permission_profile" not in codex:
        codex["permission_profile"] = codex["permissions"]
    if isinstance(codex, dict):
        codex.pop("permissions", None)
    return config


def _canonical_key(dotted: str) -> str:
    if dotted == "codex.permissions":
        return "codex.permission_profile"
    return dotted


def _require_known_config_key(dotted: str) -> None:
    if dotted in known_config_keys():
        return
    suggestion = config_key_suggestion(dotted)
    suffix = f"; did you mean {suggestion!r}?" if suggestion else ""
    raise ValueError(f"unknown config key: {dotted!r}{suffix}")


def _collect_known_keys(data: dict[str, Any], prefix: list[str], keys: list[str]) -> None:
    for key, value in data.items():
        path = [*prefix, key]
        if isinstance(value, dict):
            _collect_known_keys(value, path, keys)
        else:
            keys.append(".".join(path))


def _collect_unknown_keys(data: dict[str, Any], schema: dict[str, Any], prefix: list[str], unknown: list[str]) -> None:
    for key, value in data.items():
        path = [*prefix, key]
        expected = schema.get(key)
        if key not in schema:
            unknown.append(".".join(path))
        elif isinstance(value, dict) and isinstance(expected, dict):
            _collect_unknown_keys(value, expected, path, unknown)


def _prune_empty_tables(data: dict[str, Any], parts: list[str]) -> None:
    if not parts:
        return
    parents: list[tuple[dict[str, Any], str]] = []
    current: Any = data
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return
        parents.append((current, part))
        current = current[part]
    for parent, key in reversed(parents):
        value = parent.get(key)
        if isinstance(value, dict) and not value:
            del parent[key]
        else:
            break


def parse_value(value: str) -> Any:
    try:
        return tomllib.loads(f"value = {value}\n")["value"]
    except tomllib.TOMLDecodeError:
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        return value


def _toml(data: dict[str, Any]) -> str:
    lines: list[str] = []
    for section, value in data.items():
        if isinstance(value, dict):
            _write_section(lines, [section], value)
        else:
            lines.append(f"{section} = {_format_value(value)}")
    return "\n".join(lines).rstrip() + "\n"


def _write_section(lines: list[str], path: list[str], table: dict[str, Any]) -> None:
    scalar_items = [(k, v) for k, v in table.items() if not isinstance(v, dict)]
    nested_items = [(k, v) for k, v in table.items() if isinstance(v, dict)]
    lines.append("")
    lines.append(f"[{'.'.join(path)}]")
    for key, value in scalar_items:
        lines.append(f"{key} = {_format_value(value)}")
    for key, value in nested_items:
        _write_section(lines, [*path, key], value)


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_format_value(item) for item in value) + "]"
    if isinstance(value, str) and "\n" in value:
        escaped = value.replace('"""', '\\"\\"\\"')
        return f'"""{escaped}"""'
    return json.dumps(value)
