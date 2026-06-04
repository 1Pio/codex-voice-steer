from __future__ import annotations

from typing import Any

from .config import Config
from .paths import expand_path
from .state import CxvState, StateStore


def configured_resume_thread_id(config: Config) -> tuple[str, str]:
    thread_id = str(config.get("codex.thread_id", "") or "")
    if thread_id:
        return thread_id, "codex.thread_id"
    resume_thread_id = str(config.get("codex.resume_thread_id", "") or "")
    if resume_thread_id:
        return resume_thread_id, "codex.resume_thread_id"
    return "", ""


def load_session_state(config: Config) -> CxvState:
    return StateStore(expand_path(str(config.get("server.state_db")))).load()


def session_status_info(config: Config, state: CxvState | None = None) -> dict[str, Any]:
    state = state or load_session_state(config)
    configured_thread_id, configured_source = configured_resume_thread_id(config)
    effective_thread_id = configured_thread_id or state.thread_id
    if configured_thread_id:
        effective_source = configured_source
    elif state.thread_id:
        effective_source = "saved state"
    else:
        effective_source = "new thread"
    return {
        "effective_resume_thread_id": effective_thread_id,
        "effective_resume_source": effective_source,
        "saved_thread_id": state.thread_id,
        "saved_session_id": state.session_id,
        "saved_cwd": state.cwd,
        "active_turn_id": state.active_turn_id,
        "queued_inputs": len(state.queued_inputs or []),
        "config_thread_id": str(config.get("codex.thread_id", "") or ""),
        "config_resume_thread_id": str(config.get("codex.resume_thread_id", "") or ""),
        "config_cwd": str(config.get("codex.cwd", ".")),
        "create_thread_if_missing": bool(config.get("codex.create_thread_if_missing", True)),
        "model": str(config.get("codex.model", "") or ""),
        "effort": str(config.get("codex.effort", "") or ""),
        "fast": bool(config.get("codex.fast", False)),
        "agent": str(config.get("codex.agent", "") or ""),
        "permission_profile": str(config.get("codex.permission_profile", config.get("codex.permissions", "")) or ""),
        "approval_policy": str(config.get("codex.approval_policy", "") or ""),
        "delivery_when_idle": str(config.get("delivery.when_idle", "") or ""),
        "delivery_when_active": str(config.get("delivery.when_active", "") or ""),
        "delivery_when_not_steerable": str(config.get("delivery.when_not_steerable", "") or ""),
        "behavior": session_behavior(config, state),
    }


def session_behavior(config: Config, state: CxvState | None = None) -> str:
    state = state or load_session_state(config)
    configured_thread_id, configured_source = configured_resume_thread_id(config)
    when_idle = str(config.get("delivery.when_idle", "start"))
    when_active = str(config.get("delivery.when_active", "steer"))
    if configured_thread_id:
        resume = f"resume {configured_source}"
    elif state.thread_id:
        resume = "resume saved state thread"
    else:
        resume = "start a new thread on next send"
    active = "steer active turn" if when_active == "steer" else f"{when_active} active-turn input"
    return f"{resume}; when idle: {when_idle}; when active: {active}"


def render_session_status(info: dict[str, Any]) -> str:
    lines = [
        "cxv session:",
        f"  effective resume thread: {_value(info['effective_resume_thread_id'])} ({info['effective_resume_source']})",
        f"  saved thread: {_value(info['saved_thread_id'])}",
        f"  saved session: {_value(info['saved_session_id'])}",
        f"  saved cwd: {_value(info['saved_cwd'])}",
        f"  active turn: {_value(info['active_turn_id'])}",
        f"  queued inputs: {info['queued_inputs']}",
        f"  config codex.thread_id: {_value(info['config_thread_id'])}",
        f"  config codex.resume_thread_id: {_value(info['config_resume_thread_id'])}",
        f"  config codex.cwd: {_value(info['config_cwd'])}",
        f"  create_thread_if_missing: {_yes_no(info['create_thread_if_missing'])}",
        f"  model: {_value(info['model'])}",
        f"  effort: {_value(info['effort'])}",
        f"  fast: {_yes_no(info['fast'])}",
        f"  agent: {_value(info['agent'])}",
        f"  permission profile: {_value(info['permission_profile'])}",
        f"  approval policy: {_value(info['approval_policy'])}",
        f"  delivery: idle={_value(info['delivery_when_idle'])} active={_value(info['delivery_when_active'])} not_steerable={_value(info['delivery_when_not_steerable'])}",
        f"  behavior: {info['behavior']}",
    ]
    return "\n".join(lines)


def render_session_header(config: Config) -> str:
    info = session_status_info(config)
    session_id = _value(info["saved_session_id"])
    resume_thread_id = info["effective_resume_thread_id"] or "new"
    return f"session: {session_id}     resume: {resume_thread_id}"


def _value(value: Any) -> str:
    value = str(value or "").strip()
    return value if value else "-"


def _yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"
