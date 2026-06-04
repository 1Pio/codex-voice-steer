from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .codex_app_server import CodexAppServer
from .config import Config, load_config
from .audio import MicCapture, audio_readiness, wav_frames
from .paths import expand_path
from .session import configured_resume_thread_id
from .state import StateStore
from .stt import MacParakeetStt
from .vad import SileroVad, vad_readiness
from .voice_pipeline import VoicePipeline
from .wake import OpenWakeWordDetector, wake_readiness


@dataclass(frozen=True)
class TokenUsageSnapshot:
    thread_id: str
    turn_id: str
    total_tokens: int
    model_context_window: int
    usage_ratio: float


class CxvDaemon:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.socket_path = expand_path(str(config.get("server.socket_path")))
        self.pid_path = expand_path(str(config.get("server.pid_path")))
        self.state_store = StateStore(expand_path(str(config.get("server.state_db"))))
        self.codex: CodexAppServer | None = None
        self.listen_task: asyncio.Task[None] | None = None
        self.listen_stop_event: threading.Event | None = None
        self.listen_overrides: dict[str, Any] = {}
        self.live_wake: OpenWakeWordDetector | None = None
        self.live_wake_key: tuple[object, ...] | None = None
        self.last_activity_monotonic = time.monotonic()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.latest_token_usage: dict[str, TokenUsageSnapshot] = {}
        self.last_completed_turn: dict[str, str] = {}
        self.last_auto_compact_monotonic: dict[str, float] = {}
        self.auto_compact_task: asyncio.Task[None] | None = None
        self.auto_compact_activity_seq = 0
        self.auto_compact_overrides: dict[str, Any] = {}

    async def serve(self) -> None:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        self.pid_path.parent.mkdir(parents=True, exist_ok=True)
        if self.socket_path.exists():
            self.socket_path.unlink()
        self.pid_path.write_text(str(os.getpid()))
        server = await asyncio.start_unix_server(self._handle_client, path=str(self.socket_path))
        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        self.loop = loop
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop_event.set)
            except NotImplementedError:
                pass
        idle_task: asyncio.Task[None] | None = None
        if self._idle_timeout_sec() > 0:
            idle_task = asyncio.create_task(self._idle_monitor(stop_event))
        try:
            async with server:
                await stop_event.wait()
                server.close()
                await server.wait_closed()
        finally:
            await self._stop_listening(record_event=False)
            if idle_task is not None:
                idle_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await idle_task
            if self.socket_path.exists():
                self.socket_path.unlink()
            if self.pid_path.exists() and self.pid_path.read_text().strip() == str(os.getpid()):
                self.pid_path.unlink()
            if self.codex is not None:
                self.codex.close()
            self.loop = None

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        line = await reader.readline()
        try:
            request = json.loads(line.decode())
            response = await self._dispatch(request)
        except Exception as exc:
            response = {"ok": False, "error": str(exc)}
        writer.write((json.dumps(response) + "\n").encode())
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    async def _dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        self.last_activity_monotonic = time.monotonic()
        command = request.get("command")
        self._apply_runtime_auto_compact_overrides(request)
        if command == "status":
            state = self.state_store.load()
            return {"ok": True, "running": True, "state": state.to_dict()}
        if command == "listen":
            blockers = self._listen_blockers()
            if self._listener_still_stopping():
                blockers.append("microphone listener is still stopping; try again shortly")
            if blockers:
                self.state_store.update(listening=False)
                return {"ok": False, "error": "voice runtime is not ready", "blockers": blockers}
            self.listen_overrides = self._request_overrides(request)
            state = self.state_store.update(listening=True)
            self.state_store.append_event("listening_started", device=str(self.config.get("audio.device", "default")))
            if self.listen_task is None or self.listen_task.done():
                stop_event = threading.Event()
                self.listen_stop_event = stop_event
                self.listen_task = asyncio.create_task(self._listen_loop(stop_event, dict(self.listen_overrides)))
            return {"ok": True, "state": state.to_dict()}
        if command == "pause":
            state = await self._stop_listening(record_event=True)
            return {"ok": True, "state": state.to_dict()}
        if command == "bind":
            state = self.state_store.update(
                thread_id=request.get("thread_id", ""),
                session_id="",
                cwd=request.get("cwd", "."),
                active_turn_id="",
                queued_inputs=[],
            )
            return {"ok": True, "state": state.to_dict()}
        if command == "session-new":
            return await self._start_new_session(request)
        if command in {"text", "steer"}:
            self._mark_codex_input_activity("cxv_text")
            config = self._effective_config(request)
            result = await asyncio.to_thread(self._codex().deliver_text, str(request.get("text", "")), command == "steer", config)
            return {"ok": True, "result": result.__dict__}
        if command == "voice-test-audio":
            wav_path = Path(str(request.get("wav", "")))
            send = bool(request.get("send", False))
            result = await asyncio.to_thread(self._run_voice_turn_from_wav, wav_path, send, self._request_overrides(request))
            self.state_store.append_event(
                "voice_test_audio",
                status=result.status,
                transcript=result.transcript,
                wav_path=str(result.wav_path or ""),
                reason=result.reason,
                sent=bool(send and result.delivered),
            )
            return {"ok": result.delivered, "result": self._voice_result(result)}
        if command == "interrupt":
            result = await asyncio.to_thread(self._codex().interrupt)
            return {"ok": True, "result": result.__dict__}
        if command == "shutdown":
            asyncio.get_running_loop().call_soon(asyncio.get_running_loop().stop)
            return {"ok": True}
        raise ValueError(f"unknown daemon command: {command}")

    async def _idle_monitor(self, stop_event: asyncio.Event, poll_interval: float = 5.0) -> None:
        while not stop_event.is_set():
            timeout = self._idle_timeout_sec()
            if timeout <= 0:
                return
            if self._should_exit_for_idle(time.monotonic()):
                self.state_store.append_event("daemon_idle_timeout", idle_timeout_minutes=float(self.config.get("server.idle_timeout_minutes", 0)))
                stop_event.set()
                return
            await asyncio.sleep(min(poll_interval, timeout))

    def _idle_timeout_sec(self) -> float:
        return max(0.0, float(self.config.get("server.idle_timeout_minutes", 0)) * 60.0)

    def _should_exit_for_idle(self, now: float) -> bool:
        timeout = self._idle_timeout_sec()
        if timeout <= 0:
            return False
        state = self.state_store.load()
        if state.listening or state.active_turn_id or state.queued_inputs:
            self.last_activity_monotonic = now
            return False
        return now - self.last_activity_monotonic >= timeout

    def _codex(self) -> CodexAppServer:
        if self.codex is None:
            self.codex = CodexAppServer(self.config, self.state_store, on_event=self._codex_event_from_thread)
            self.codex.start()
        return self.codex

    async def _start_new_session(self, request: dict[str, Any]) -> dict[str, Any]:
        config = self._effective_config(request)
        configured_thread_id, configured_source = configured_resume_thread_id(config)
        if configured_thread_id:
            return {
                "ok": False,
                "error": f"{configured_source} is set; unset it before creating a reusable saved session",
                "configured_thread_id": configured_thread_id,
                "configured_source": configured_source,
            }
        state = self.state_store.load()
        force = bool(request.get("force", False))
        if state.active_turn_id and not force:
            return {
                "ok": False,
                "error": "active turn is running; use --force to interrupt and start a new session",
                "active_turn_id": state.active_turn_id,
            }
        if state.active_turn_id and force:
            try:
                await asyncio.to_thread(self._codex().interrupt)
            except Exception as exc:
                self.state_store.append_event("session_interrupt_failed", error=str(exc), turn_id=state.active_turn_id)
        self.state_store.update(active_turn_id="", queued_inputs=[])
        state = await asyncio.to_thread(self._codex().start_new_thread, config)
        return {"ok": True, "state": state.to_dict()}

    def _listen_blockers(self) -> list[str]:
        checks = [
            audio_readiness(self.config, probe_stream=True),
            vad_readiness(),
            wake_readiness(self.config),
        ]
        return [check.reason for check in checks if not check.ok]

    async def _listen_loop(self, stop_event: threading.Event, overrides: dict[str, Any]) -> None:
        try:
            while self.state_store.load().listening and not stop_event.is_set():
                result = await asyncio.to_thread(self._run_voice_turn, stop_event, overrides)
                if stop_event.is_set() or not self.state_store.load().listening:
                    break
                self.state_store.append_event(
                    "voice_turn",
                    status=result.status,
                    transcript=result.transcript,
                    wav_path=str(result.wav_path or ""),
                    reason=result.reason,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.state_store.append_event("voice_error", error=str(exc))
            self.state_store.update(listening=False)
        finally:
            if self.listen_stop_event is stop_event:
                self.listen_stop_event = None
            if self.listen_task is asyncio.current_task():
                self.listen_task = None

    async def _stop_listening(self, record_event: bool) -> Any:
        state = self.state_store.update(listening=False)
        if record_event:
            self.state_store.append_event("listening_paused")
        self.listen_overrides = {}
        if self.listen_stop_event is not None:
            self.listen_stop_event.set()
        task = self.listen_task
        if task is not None and not task.done():
            done, _pending = await asyncio.wait({task}, timeout=2.0)
            if not done:
                self.state_store.append_event("listening_stop_pending", timeout_sec=2.0)
        if self.listen_task is not None and self.listen_task.done():
            self.listen_task = None
        if self.listen_task is None:
            self.listen_stop_event = None
        return state

    def _listener_still_stopping(self) -> bool:
        return (
            self.listen_task is not None
            and not self.listen_task.done()
            and self.listen_stop_event is not None
            and self.listen_stop_event.is_set()
        )

    def _run_voice_turn(self, stop_event: threading.Event, overrides: dict[str, Any]):
        config = self.config.with_overrides(overrides)
        return self._voice_pipeline(send=True, config=config).run_once(MicCapture(config, stop_event=stop_event).frames())

    def _run_voice_turn_from_wav(self, wav_path: Path, send: bool, overrides: dict[str, Any] | None = None):
        config = self.config.with_overrides(overrides)
        return self._voice_pipeline(send=send, config=config).run_once(wav_frames(config, wav_path))

    def _voice_pipeline(self, send: bool, config: Config | None = None) -> VoicePipeline:
        config = config or self.config
        delivery_config = self._voice_delivery_config(config)
        deliver_text = (lambda text: self._deliver_voice_text(text, delivery_config)) if send else (lambda _text: None)
        pipeline = VoicePipeline(
            config,
            wake=self._wake_detector(config) if send else OpenWakeWordDetector(config),
            vad=SileroVad(config),
            stt=MacParakeetStt(config),
            deliver_text=deliver_text,
            event_sink=lambda event, fields: self.state_store.append_event(event, **fields),
        )
        return pipeline

    def _wake_detector(self, config: Config) -> OpenWakeWordDetector:
        key = (
            str(config.get("wake.model_path", "")),
            str(config.get("wake.word", "scarlett")),
            float(config.get("wake.sensitivity", 0.5)),
            int(config.get("wake.refractory_ms", 1200)),
        )
        if self.live_wake is None or self.live_wake_key != key:
            self.live_wake = OpenWakeWordDetector(config)
            self.live_wake_key = key
        return self.live_wake

    @staticmethod
    def _voice_delivery_config(config: Config) -> Config:
        if config.get("wake.allow_barge_in", True):
            return config
        return config.with_overrides({"delivery": {"when_active": "queue"}})

    def _effective_config(self, request: dict[str, Any]) -> Config:
        return self.config.with_overrides(self._request_overrides(request))

    def _deliver_voice_text(self, text: str, config: Config) -> Any:
        self._mark_codex_input_activity("cxv_voice")
        return self._codex().deliver_text(text, config=config)

    def _codex_event_from_thread(self, method: str, params: dict[str, Any]) -> None:
        loop = self.loop
        if loop is None or loop.is_closed():
            return
        loop.call_soon_threadsafe(self._handle_codex_event, method, params)

    def _handle_codex_event(self, method: str, params: dict[str, Any]) -> None:
        thread_id = self._event_thread_id(params)
        if method == "thread/tokenUsage/updated":
            snapshot = self._token_usage_snapshot(params)
            if snapshot is not None:
                self.latest_token_usage[snapshot.thread_id] = snapshot
                self._maybe_schedule_auto_compact(snapshot.thread_id, snapshot.turn_id, "token_usage_updated")
            return
        if not thread_id or not self._is_bound_thread(thread_id):
            return
        if method == "turn/started":
            self._mark_codex_input_activity("turn_started", thread_id=thread_id)
            return
        if method == "turn/completed":
            turn_id = self._event_turn_id(params) or self.state_store.load().active_turn_id
            if turn_id:
                self.last_completed_turn[thread_id] = turn_id
            self._maybe_schedule_auto_compact(thread_id, turn_id, "turn_completed")

    def _maybe_schedule_auto_compact(self, thread_id: str, turn_id: str, source: str) -> None:
        if not thread_id or not self._is_bound_thread(thread_id):
            return
        config = self._auto_compact_config()
        if not config.get("auto_compact.enabled", True):
            return
        snapshot = self.latest_token_usage.get(thread_id)
        if snapshot is None:
            return
        completed_turn_id = self.last_completed_turn.get(thread_id, "")
        if not completed_turn_id:
            return
        if snapshot.turn_id and snapshot.turn_id != completed_turn_id:
            return
        if turn_id and turn_id != completed_turn_id:
            return
        if turn_id and snapshot.turn_id and snapshot.turn_id != turn_id:
            return
        threshold = self._auto_compact_threshold_ratio(config)
        if snapshot.usage_ratio < threshold:
            return
        state = self.state_store.load()
        if state.active_turn_id or state.queued_inputs:
            return
        idle_delay_sec = self._auto_compact_idle_delay_sec(config)
        cooldown_sec = self._auto_compact_cooldown_sec(config)
        now = time.monotonic()
        last_compact = self.last_auto_compact_monotonic.get(thread_id, 0.0)
        if cooldown_sec > 0 and last_compact and now - last_compact < cooldown_sec:
            self.state_store.append_event(
                "auto_compact_skipped",
                thread_id=thread_id,
                turn_id=snapshot.turn_id,
                reason="cooldown",
                cooldown_sec=cooldown_sec,
                usage_ratio=snapshot.usage_ratio,
                threshold_ratio=threshold,
            )
            return
        self._cancel_auto_compact_task("rescheduled")
        activity_seq = self.auto_compact_activity_seq
        self.auto_compact_task = asyncio.create_task(
            self._run_auto_compact_after_idle(snapshot, threshold, idle_delay_sec, cooldown_sec, activity_seq, source)
        )
        self.state_store.append_event(
            "auto_compact_scheduled",
            thread_id=thread_id,
            turn_id=snapshot.turn_id,
            source=source,
            idle_delay_sec=idle_delay_sec,
            usage_ratio=snapshot.usage_ratio,
            threshold_ratio=threshold,
            total_tokens=snapshot.total_tokens,
            model_context_window=snapshot.model_context_window,
        )

    async def _run_auto_compact_after_idle(
        self,
        snapshot: TokenUsageSnapshot,
        threshold_ratio: float,
        idle_delay_sec: float,
        cooldown_sec: float,
        activity_seq: int,
        source: str,
    ) -> None:
        try:
            await asyncio.sleep(idle_delay_sec)
            if self.auto_compact_task is asyncio.current_task():
                self.auto_compact_task = None
            if activity_seq != self.auto_compact_activity_seq:
                self._record_auto_compact_skip(snapshot, "new_input", threshold_ratio)
                return
            state = self.state_store.load()
            if state.thread_id != snapshot.thread_id:
                self._record_auto_compact_skip(snapshot, "thread_changed", threshold_ratio)
                return
            if state.active_turn_id:
                self._record_auto_compact_skip(snapshot, "active_turn", threshold_ratio)
                return
            if state.queued_inputs:
                self._record_auto_compact_skip(snapshot, "queued_input", threshold_ratio)
                return
            latest = self.latest_token_usage.get(snapshot.thread_id, snapshot)
            if latest.usage_ratio < threshold_ratio:
                self._record_auto_compact_skip(latest, "below_threshold", threshold_ratio)
                return
            now = time.monotonic()
            last_compact = self.last_auto_compact_monotonic.get(snapshot.thread_id, 0.0)
            if cooldown_sec > 0 and last_compact and now - last_compact < cooldown_sec:
                self._record_auto_compact_skip(snapshot, "cooldown", threshold_ratio)
                return
            self.state_store.append_event(
                "auto_compact_started",
                thread_id=snapshot.thread_id,
                turn_id=snapshot.turn_id,
                source=source,
                usage_ratio=latest.usage_ratio,
                threshold_ratio=threshold_ratio,
                total_tokens=latest.total_tokens,
                model_context_window=latest.model_context_window,
            )
            await asyncio.to_thread(self._codex().compact_thread, snapshot.thread_id)
            self.last_auto_compact_monotonic[snapshot.thread_id] = time.monotonic()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.state_store.append_event(
                "auto_compact_failed",
                thread_id=snapshot.thread_id,
                turn_id=snapshot.turn_id,
                error=str(exc),
                usage_ratio=snapshot.usage_ratio,
                threshold_ratio=threshold_ratio,
            )
        finally:
            if self.auto_compact_task is asyncio.current_task():
                self.auto_compact_task = None

    def _record_auto_compact_skip(self, snapshot: TokenUsageSnapshot, reason: str, threshold_ratio: float) -> None:
        self.state_store.append_event(
            "auto_compact_skipped",
            thread_id=snapshot.thread_id,
            turn_id=snapshot.turn_id,
            reason=reason,
            usage_ratio=snapshot.usage_ratio,
            threshold_ratio=threshold_ratio,
            total_tokens=snapshot.total_tokens,
            model_context_window=snapshot.model_context_window,
        )

    def _mark_codex_input_activity(self, source: str, thread_id: str = "") -> None:
        self.auto_compact_activity_seq += 1
        self._cancel_auto_compact_task(source, thread_id=thread_id)

    def _cancel_auto_compact_task(self, source: str, thread_id: str = "") -> None:
        task = self.auto_compact_task
        if task is None or task.done():
            self.auto_compact_task = None
            return
        task.cancel()
        self.state_store.append_event("auto_compact_cancelled", source=source, thread_id=thread_id)

    def _apply_runtime_auto_compact_overrides(self, request: dict[str, Any]) -> None:
        overrides = self._request_overrides(request)
        auto_compact = overrides.get("auto_compact")
        if isinstance(auto_compact, dict):
            self.auto_compact_overrides.update(auto_compact)

    def _auto_compact_config(self) -> Config:
        if not self.auto_compact_overrides:
            return self.config
        return self.config.with_overrides({"auto_compact": dict(self.auto_compact_overrides)})

    @staticmethod
    def _auto_compact_threshold_ratio(config: Config) -> float:
        return max(0.0, float(config.get("auto_compact.threshold_ratio", 0.55)))

    @staticmethod
    def _auto_compact_idle_delay_sec(config: Config) -> float:
        return max(0.0, float(config.get("auto_compact.idle_delay_sec", 45.0)))

    @staticmethod
    def _auto_compact_cooldown_sec(config: Config) -> float:
        return max(0.0, float(config.get("auto_compact.cooldown_sec", 300.0)))

    def _token_usage_snapshot(self, params: dict[str, Any]) -> TokenUsageSnapshot | None:
        thread_id = self._event_thread_id(params)
        if not thread_id or not self._is_bound_thread(thread_id):
            return None
        usage = dict(params.get("tokenUsage") or {})
        total = dict(usage.get("total") or {})
        total_tokens = _safe_int(total.get("totalTokens"))
        model_context_window = _safe_int(usage.get("modelContextWindow"))
        if model_context_window <= 0:
            return None
        return TokenUsageSnapshot(
            thread_id=thread_id,
            turn_id=str(params.get("turnId", "")),
            total_tokens=total_tokens,
            model_context_window=model_context_window,
            usage_ratio=float(total_tokens / model_context_window),
        )

    def _is_bound_thread(self, thread_id: str) -> bool:
        return bool(thread_id) and thread_id == self.state_store.load().thread_id

    @staticmethod
    def _event_thread_id(params: dict[str, Any]) -> str:
        return _id_param(params, "threadId", "thread")

    @staticmethod
    def _event_turn_id(params: dict[str, Any]) -> str:
        return _id_param(params, "turnId", "turn")

    @staticmethod
    def _request_overrides(request: dict[str, Any]) -> dict[str, Any]:
        overrides = request.get("overrides", {})
        return dict(overrides) if isinstance(overrides, dict) else {}

    @staticmethod
    def _voice_result(result) -> dict[str, Any]:
        return {
            "status": result.status,
            "wav_path": str(result.wav_path or ""),
            "transcript": result.transcript,
            "delivered": result.delivered,
            "reason": result.reason,
        }


def _id_param(params: dict[str, Any], key: str, nested_key: str) -> str:
    value = params.get(key)
    if value:
        return str(value)
    nested = params.get(nested_key)
    if isinstance(nested, dict) and nested.get("id"):
        return str(nested["id"])
    return ""


def _safe_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


async def send_request(config: Config, payload: dict[str, Any]) -> dict[str, Any]:
    socket_path = expand_path(str(config.get("server.socket_path")))
    reader, writer = await asyncio.open_unix_connection(str(socket_path))
    writer.write((json.dumps(payload) + "\n").encode())
    await writer.drain()
    line = await reader.readline()
    writer.close()
    await writer.wait_closed()
    return json.loads(line.decode())


def is_running(config: Config) -> bool:
    pid_path = expand_path(str(config.get("server.pid_path")))
    socket_path = expand_path(str(config.get("server.socket_path")))
    if not pid_path.exists() or not socket_path.exists():
        return False
    try:
        os.kill(int(pid_path.read_text().strip()), 0)
    except (OSError, ValueError):
        return False
    return True


def start_background(config: Config) -> int:
    if is_running(config):
        pid_path = expand_path(str(config.get("server.pid_path")))
        return int(pid_path.read_text().strip())
    cmd = [sys.executable, "-m", "codex_voice_steer.cli", "serve"]
    log_file = expand_path(str(config.get("server.log_file")))
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log = log_file.open("a")
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
    deadline = time.time() + 8
    while time.time() < deadline:
        if is_running(config):
            return proc.pid
        time.sleep(0.1)
    raise RuntimeError("cxv daemon did not become ready")


def stop_background(config: Config) -> bool:
    if not is_running(config):
        StateStore(expand_path(str(config.get("server.state_db")))).update(active_turn_id="", listening=False)
        _terminate_stale_serve_processes()
        return False
    pid_path = expand_path(str(config.get("server.pid_path")))
    pid = int(pid_path.read_text().strip())
    os.kill(pid, signal.SIGTERM)
    deadline = time.time() + 5
    while time.time() < deadline:
        if not is_running(config):
            StateStore(expand_path(str(config.get("server.state_db")))).update(active_turn_id="", listening=False)
            _terminate_stale_serve_processes(exclude={pid})
            return True
        time.sleep(0.1)
    os.kill(pid, signal.SIGKILL)
    StateStore(expand_path(str(config.get("server.state_db")))).update(active_turn_id="", listening=False)
    _terminate_stale_serve_processes(exclude={pid})
    return True


def _terminate_stale_serve_processes(exclude: set[int] | None = None) -> None:
    exclude = set(exclude or set())
    exclude.add(os.getpid())
    for pid in _stale_serve_pids(exclude=exclude):
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            continue
    deadline = time.time() + 2
    while time.time() < deadline:
        remaining = [pid for pid in _stale_serve_pids(exclude=exclude) if _pid_alive(pid)]
        if not remaining:
            return
        time.sleep(0.1)
    for pid in _stale_serve_pids(exclude=exclude):
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


def _stale_serve_pids(exclude: set[int] | None = None) -> list[int]:
    exclude = set(exclude or set())
    try:
        proc = subprocess.run(
            ["pgrep", "-f", r"python.*-m codex_voice_steer\.cli serve"],
            text=True,
            capture_output=True,
            timeout=2,
            check=False,
        )
    except Exception:
        return []
    if proc.returncode not in {0, 1}:
        return []
    pids: list[int] = []
    for line in proc.stdout.splitlines():
        try:
            pid = int(line.strip())
        except ValueError:
            continue
        if pid not in exclude:
            pids.append(pid)
    return pids


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


async def ensure_daemon(config: Config, no_start: bool = False) -> None:
    if is_running(config):
        return
    if no_start:
        raise RuntimeError("cxv daemon is not running")
    start_background(config)


def run_serve() -> None:
    asyncio.run(CxvDaemon(load_config()).serve())
