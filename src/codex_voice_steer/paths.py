from __future__ import annotations

import os
from pathlib import Path


APP_NAME = "codex-voice-steer"


def expand_path(value: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(value))).resolve()


def config_dir() -> Path:
    return Path.home() / ".config" / APP_NAME


def config_path() -> Path:
    return config_dir() / "config.toml"


def app_support_dir() -> Path:
    return Path.home() / "Library" / "Application Support" / APP_NAME


def log_dir() -> Path:
    return Path.home() / "Library" / "Logs" / APP_NAME


def run_dir() -> Path:
    return app_support_dir() / "run"


def state_db_path() -> Path:
    return app_support_dir() / "state.json"


def default_socket_path() -> Path:
    return run_dir() / "cxv.sock"


def default_pid_path() -> Path:
    return run_dir() / "cxv.pid"
