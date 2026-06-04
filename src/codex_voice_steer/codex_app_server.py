from __future__ import annotations

import json
import re
import shlex
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .agents import agent_developer_instructions
from .config import Config
from .session import configured_resume_thread_id
from .state import CxvState, StateStore


EventHandler = Callable[[str, dict[str, Any]], None]


@dataclass
class DeliveryResult:
    action: str
    thread_id: str
    turn_id: str = ""
    queued: bool = False


class JsonRpcError(RuntimeError):
    def __init__(self, message: str, error: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.error = error or {}


class CodexAppServer:
    def __init__(self, config: Config, state_store: StateStore | None = None, on_event: EventHandler | None = None) -> None:
        self.config = config
        self.state_store = state_store or StateStore()
        self.on_event = on_event or (lambda _method, _params: None)
        self.proc: subprocess.Popen[str] | None = None
        self._next_id = 1
        self._pending: dict[int, dict[str, Any]] = {}
        self._condition = threading.Condition()
        self._reader: threading.Thread | None = None
        self.codex_home = ""

    def __enter__(self) -> "CodexAppServer":
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def start(self) -> None:
        if self.proc is not None:
            return
        listen = self._app_server_listen()
        self.proc = subprocess.Popen(
            ["codex", "app-server", "--listen", listen],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        init = self.request(
            "initialize",
            {
                "clientInfo": {"name": "codex-voice-steer", "title": "cxv", "version": "0.1.0"},
                "capabilities": {"experimentalApi": True},
            },
        )
        self.codex_home = str(init.get("codexHome", ""))

    def _app_server_listen(self) -> str:
        mode = str(self.config.get("codex.app_server", "managed"))
        if mode != "managed":
            raise ValueError(f"unsupported codex.app_server: {mode}")
        return str(self.config.get("codex.app_server_listen", "stdio://"))

    def close(self) -> None:
        if self.proc is None:
            return
        if self.proc.stdin:
            self.proc.stdin.close()
        self.proc.terminate()
        try:
            self.proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.proc = None

    def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if self.proc is None or self.proc.stdin is None:
            raise RuntimeError("app-server is not running")
        with self._condition:
            request_id = self._next_id
            self._next_id += 1
            payload = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
            self.proc.stdin.write(json.dumps(payload) + "\n")
            self.proc.stdin.flush()
            while request_id not in self._pending:
                self._condition.wait()
            response = self._pending.pop(request_id)
        if "error" in response:
            raise JsonRpcError(f"{method} failed: {response['error']}", response["error"])
        return dict(response.get("result") or {})

    def _read_loop(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        for line in self.proc.stdout:
            if not line.strip():
                continue
            message = json.loads(line)
            if "id" in message:
                with self._condition:
                    self._pending[int(message["id"])] = message
                    self._condition.notify_all()
                continue
            method = str(message.get("method", ""))
            params = dict(message.get("params") or {})
            self._handle_notification(method, params)
            self.on_event(method, params)

    def _handle_notification(self, method: str, params: dict[str, Any]) -> None:
        if method == "turn/started":
            turn_id = self._id_param(params, "turnId", "turn")
            if turn_id:
                self.state_store.update(active_turn_id=turn_id)
            self.state_store.append_event("turn_started", turn_id=turn_id)
        elif method in {"agentMessage/delta", "item/agentMessage/delta"}:
            self.state_store.append_event(
                "codex_visible_delta",
                thread_id=self._id_param(params, "threadId", "thread"),
                turn_id=self._id_param(params, "turnId", "turn"),
                delta=str(params.get("delta", "")),
            )
        elif method == "item/started":
            item = dict(params.get("item") or {})
            msd_command = self._tool_item_is_msd_command(item)
            summary = self._tool_item_summary(item, clip=not msd_command)
            if summary:
                self.state_store.append_event(
                    "codex_msd_started" if msd_command else "codex_tool_started",
                    thread_id=self._id_param(params, "threadId", "thread"),
                    turn_id=self._id_param(params, "turnId", "turn"),
                    item_id=str(item.get("id", "")),
                    item_type=str(item.get("type", "")),
                    summary=summary,
                    msd_args=_msd_say_args(_command_text(item.get("command", ""))) if msd_command else "",
                )
        elif method == "item/completed":
            item = dict(params.get("item") or {})
            if str(item.get("type", "")) == "agentMessage" and str(item.get("phase", "")) == "final_answer":
                text = str(item.get("text", "")).strip()
                if text:
                    self.state_store.append_event(
                        "codex_final_answer",
                        thread_id=self._id_param(params, "threadId", "thread"),
                        turn_id=self._id_param(params, "turnId", "turn"),
                        item_id=str(item.get("id", "")),
                        text=text,
                    )
        elif method == "item/mcpToolCall/progress":
            self.state_store.append_event(
                "codex_tool_progress",
                thread_id=self._id_param(params, "threadId", "thread"),
                turn_id=self._id_param(params, "turnId", "turn"),
                item_id=str(params.get("itemId", "")),
                message=str(params.get("message", "")),
            )
        elif method == "turn/completed":
            state = self.state_store.load()
            turn_id = self._id_param(params, "turnId", "turn") or state.active_turn_id
            self.state_store.append_event(
                "turn_completed",
                thread_id=self._id_param(params, "threadId", "thread"),
                turn_id=turn_id,
            )
            if not turn_id or not state.active_turn_id or turn_id == state.active_turn_id:
                state = self.state_store.update(active_turn_id="")
            if state.queued_inputs:
                queued = state.queued_inputs.pop(0)
                self.state_store.save(state)
                self.deliver_text(queued)

    @staticmethod
    def _id_param(params: dict[str, Any], key: str, nested_key: str) -> str:
        value = params.get(key)
        if value:
            return str(value)
        nested = params.get(nested_key)
        if isinstance(nested, dict) and nested.get("id"):
            return str(nested["id"])
        return ""

    @staticmethod
    def _tool_item_summary(item: dict[str, Any], clip: bool = True) -> str:
        item_type = str(item.get("type", ""))
        if item_type == "commandExecution":
            command = _command_text(item.get("command", ""))
            if not command:
                return "command"
            summary = f"command: {command}"
            return _clip(summary) if clip else summary
        if item_type == "mcpToolCall":
            server = str(item.get("server", "")).strip()
            tool = str(item.get("tool", "")).strip()
            name = ".".join(part for part in (server, tool) if part)
            return _clip(f"tool: {name}") if name else "tool"
        if item_type == "dynamicToolCall":
            namespace = str(item.get("namespace", "") or "").strip()
            tool = str(item.get("tool", "")).strip()
            name = ".".join(part for part in (namespace, tool) if part)
            return _clip(f"tool: {name}") if name else "tool"
        if item_type == "fileChange":
            changes = item.get("changes")
            count = len(changes) if isinstance(changes, list) else 0
            return f"file change: {count} update(s)" if count else "file change"
        return ""

    @staticmethod
    def _tool_item_is_msd_command(item: dict[str, Any]) -> bool:
        if str(item.get("type", "")) != "commandExecution":
            return False
        return _command_invokes_msd(_command_text(item.get("command", "")))

    def ensure_thread(self, config: Config | None = None) -> str:
        config = config or self.config
        state = self.state_store.load()
        configured, _source = configured_resume_thread_id(config)
        thread_id = configured or state.thread_id
        if thread_id:
            try:
                result = self.request("thread/resume", self._thread_resume_params(thread_id, config))
                thread = result.get("thread", {})
                thread_id = str(thread.get("id", thread_id))
                self.state_store.update(thread_id=thread_id, session_id=str(thread.get("sessionId", "")))
                return thread_id
            except JsonRpcError:
                if not config.get("codex.create_thread_if_missing", True):
                    raise
        result = self.request("thread/start", self._thread_start_params(config))
        thread = result.get("thread", {})
        thread_id = str(thread.get("id", ""))
        self.state_store.update(thread_id=thread_id, session_id=str(thread.get("sessionId", "")), cwd=str(thread.get("cwd", config.get("codex.cwd", "."))))
        return thread_id

    def start_new_thread(self, config: Config | None = None) -> CxvState:
        config = config or self.config
        result = self.request("thread/start", self._thread_start_params(config))
        thread = result.get("thread", {})
        thread_id = str(thread.get("id", ""))
        state = self.state_store.update(
            thread_id=thread_id,
            session_id=str(thread.get("sessionId", "")),
            cwd=str(thread.get("cwd", config.get("codex.cwd", "."))),
            active_turn_id="",
            queued_inputs=[],
        )
        self.state_store.append_event("session_started", thread_id=thread_id, session_id=state.session_id, cwd=state.cwd)
        return state

    def deliver_text(self, text: str, force_steer: bool = False, config: Config | None = None) -> DeliveryResult:
        config = config or self.config
        state = self.state_store.load()
        active_turn_id = state.active_turn_id
        when_active = str(config.get("delivery.when_active", "steer"))
        when_idle = str(config.get("delivery.when_idle", "start"))
        if when_active not in {"queue", "steer"}:
            raise ValueError(f"unsupported delivery.when_active: {when_active}")
        if when_idle != "start":
            raise ValueError(f"unsupported delivery.when_idle: {when_idle}")
        thread_id = self.ensure_thread(config)
        self.state_store.append_event("user_final", text=text, source="text")
        active_turn_id = self.state_store.load().active_turn_id
        if active_turn_id and not force_steer and when_active == "queue":
            queued = state.queued_inputs or []
            queued.append(text)
            self.state_store.update(queued_inputs=queued)
            return DeliveryResult(action="queue", thread_id=thread_id, queued=True)
        if active_turn_id and (force_steer or when_active == "steer"):
            try:
                result = self.request(
                    "turn/steer",
                    {"threadId": thread_id, "expectedTurnId": active_turn_id, "input": [self._text_input(text, config)]},
                )
                turn = result.get("turn", {})
                self.state_store.append_event("sent", action="turn/steer", thread_id=thread_id, turn_id=str(turn.get("id", active_turn_id)))
                return DeliveryResult(action="turn/steer", thread_id=thread_id, turn_id=str(turn.get("id", active_turn_id)))
            except JsonRpcError:
                if config.get("delivery.when_not_steerable", "queue") == "queue":
                    queued = state.queued_inputs or []
                    queued.append(text)
                    self.state_store.update(queued_inputs=queued)
                    return DeliveryResult(action="queue", thread_id=thread_id, queued=True)
                raise
        result = self.request("turn/start", self._turn_start_params(thread_id, text, config))
        turn = result.get("turn", {})
        turn_id = str(turn.get("id", ""))
        self.state_store.update(active_turn_id=turn_id)
        self.state_store.append_event("sent", action="turn/start", thread_id=thread_id, turn_id=turn_id)
        return DeliveryResult(action="turn/start", thread_id=thread_id, turn_id=turn_id)

    def interrupt(self) -> DeliveryResult:
        state = self.state_store.load()
        if not state.thread_id or not state.active_turn_id:
            self.state_store.append_event("sent", action="turn/interrupt", thread_id=state.thread_id, turn_id=state.active_turn_id, noop=True)
            return DeliveryResult(action="noop", thread_id=state.thread_id)
        self.request("turn/interrupt", {"threadId": state.thread_id, "turnId": state.active_turn_id})
        self.state_store.append_event("sent", action="turn/interrupt", thread_id=state.thread_id, turn_id=state.active_turn_id)
        self.state_store.update(active_turn_id="")
        return DeliveryResult(action="turn/interrupt", thread_id=state.thread_id, turn_id=state.active_turn_id)

    def _thread_start_params(self, config: Config | None = None) -> dict[str, Any]:
        config = config or self.config
        params: dict[str, Any] = {
            "cwd": str(Path(str(config.get("codex.cwd", "."))).resolve()),
            "developerInstructions": self._developer_instructions(config),
            "model": config.get("codex.model", None),
            "personality": config.get("codex.personality", "pragmatic"),
            "approvalPolicy": config.get("codex.approval_policy", "on-request"),
            "approvalsReviewer": config.get("codex.approvals_reviewer", "auto_review"),
            "config": {"default_permissions": self._permission_profile(config)},
            "serviceName": "codex-voice-steer",
            "sessionStartSource": "startup",
            "threadSource": "user",
        }
        if config.get("codex.fast", False):
            params["serviceTier"] = "fast"
        return params

    def _thread_resume_params(self, thread_id: str, config: Config | None = None) -> dict[str, Any]:
        config = config or self.config
        return {
            "threadId": thread_id,
            "cwd": str(Path(str(config.get("codex.cwd", "."))).resolve()),
            "developerInstructions": self._developer_instructions(config),
            "model": config.get("codex.model", None),
            "approvalPolicy": config.get("codex.approval_policy", "on-request"),
            "approvalsReviewer": config.get("codex.approvals_reviewer", "auto_review"),
            "config": {"default_permissions": self._permission_profile(config)},
        }

    def _turn_start_params(self, thread_id: str, text: str, config: Config | None = None) -> dict[str, Any]:
        config = config or self.config
        params: dict[str, Any] = {
            "threadId": thread_id,
            "input": [self._text_input(text, config)],
            "cwd": str(Path(str(config.get("codex.cwd", "."))).resolve()),
            "model": config.get("codex.model", None),
            "effort": config.get("codex.effort", "medium"),
            "summary": config.get("codex.summary", "concise"),
            "personality": config.get("codex.personality", "pragmatic"),
            "approvalPolicy": config.get("codex.approval_policy", "on-request"),
            "approvalsReviewer": config.get("codex.approvals_reviewer", "auto_review"),
        }
        if config.get("codex.fast", False):
            params["serviceTier"] = "fast"
        return params

    def _text_input(self, text: str, config: Config | None = None) -> dict[str, str]:
        config = config or self.config
        if config.get("delivery.include_voice_metadata", True):
            fields = ["cxv voice/text input"]
            if config.get("delivery.include_wake_word", True):
                fields.append(f"wake={config.get('wake.word', 'scarlett')}")
            if config.get("delivery.include_stt_diagnostics", False):
                fields.append(f"stt={config.get('stt.engine', 'macparakeet')} mode={config.get('stt.mode', 'clean')}")
            text = f"[{'; '.join(fields)}]\n{text}"
        return {"type": "text", "text": text}

    def _developer_instructions(self, config: Config | None = None) -> str:
        config = config or self.config
        mode = str(config.get("instructions.mode", "inject"))
        if mode not in {"inject", "native_agent"}:
            return ""
        parts = [agent_developer_instructions(str(config.get("codex.agent", "")))]
        parts.append(str(config.get("instructions.developer_instructions", "")))
        if config.get("instructions.msd.enabled", False):
            parts.append(str(config.get("instructions.msd.developer_instructions", "")))
        return "\n\n".join(part for part in parts if part.strip())

    @staticmethod
    def _permission_profile(config: Config) -> str:
        return str(config.get("codex.permission_profile", config.get("codex.permissions", ":workspace")))


def _clip(value: str, limit: int = 120) -> str:
    clean = " ".join(value.split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1] + "..."


def _command_text(value: object) -> str:
    if isinstance(value, list):
        return " ".join(str(part) for part in value)
    return " ".join(str(value).split())


def _command_invokes_msd(command: str) -> bool:
    command = command.strip()
    if not command:
        return False
    try:
        parts = shlex.split(command)
    except ValueError:
        parts = []
    if parts:
        executable = Path(parts[0]).name
        if executable == "msd":
            return True
        if executable in {"sh", "bash", "zsh"}:
            shell_command = _shell_command_arg(parts)
            if shell_command:
                return _command_invokes_msd(shell_command)
    return re.search(r"(^|[\s;&|('\"])(?:[\w./-]+/)?msd(?:\s|$)", command) is not None


def _msd_say_args(command: str) -> str:
    command = command.strip()
    if not command:
        return ""
    try:
        parts = shlex.split(command)
    except ValueError:
        return _msd_say_args_fallback(command)
    if not parts:
        return ""
    executable = Path(parts[0]).name
    if executable == "msd" and len(parts) >= 2 and parts[1] == "say":
        return shlex.join(parts[2:])
    if executable in {"sh", "bash", "zsh"}:
        shell_command = _shell_command_arg(parts)
        if shell_command:
            return _msd_say_args(shell_command)
    return _msd_say_args_fallback(command)


def _msd_say_args_fallback(command: str) -> str:
    match = re.search(r"(^|[\s;&|('\"])(?:[\w./-]+/)?msd\s+say(?P<args>\s+.*|$)", command)
    if not match:
        return ""
    return match.group("args").strip().strip("'\"")


def _shell_command_arg(parts: list[str]) -> str:
    for index, part in enumerate(parts[1:], start=1):
        if part == "-c" or (part.startswith("-") and "c" in part[1:]):
            if index + 1 < len(parts):
                return parts[index + 1]
            return ""
    return ""
