from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .codex_app_server import CodexAppServer
from .config import Config, load_config
from .audio import MicCapture, audio_readiness, wav_frames
from .paths import expand_path
from .state import StateStore
from .stt import MacParakeetStt
from .vad import SileroVad, vad_readiness
from .voice_pipeline import VoicePipeline
from .wake import OpenWakeWordDetector, wake_readiness


class CxvDaemon:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.socket_path = expand_path(str(config.get("server.socket_path")))
        self.pid_path = expand_path(str(config.get("server.pid_path")))
        self.state_store = StateStore(expand_path(str(config.get("server.state_db"))))
        self.codex: CodexAppServer | None = None
        self.listen_task: asyncio.Task[None] | None = None
        self.listen_overrides: dict[str, Any] = {}
        self.live_wake: OpenWakeWordDetector | None = None
        self.live_wake_key: tuple[object, ...] | None = None
        self.last_activity_monotonic = time.monotonic()

    async def serve(self) -> None:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        self.pid_path.parent.mkdir(parents=True, exist_ok=True)
        if self.socket_path.exists():
            self.socket_path.unlink()
        self.pid_path.write_text(str(os.getpid()))
        server = await asyncio.start_unix_server(self._handle_client, path=str(self.socket_path))
        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
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
        if command == "status":
            state = self.state_store.load()
            return {"ok": True, "running": True, "state": state.to_dict()}
        if command == "listen":
            blockers = self._listen_blockers()
            if blockers:
                self.state_store.update(listening=False)
                return {"ok": False, "error": "voice runtime is not ready", "blockers": blockers}
            self.listen_overrides = self._request_overrides(request)
            state = self.state_store.update(listening=True)
            if self.listen_task is None or self.listen_task.done():
                self.listen_task = asyncio.create_task(self._listen_loop())
            return {"ok": True, "state": state.to_dict()}
        if command == "pause":
            state = self.state_store.update(listening=False)
            self.listen_overrides = {}
            if self.listen_task is not None:
                self.listen_task.cancel()
                self.listen_task = None
            return {"ok": True, "state": state.to_dict()}
        if command == "bind":
            state = self.state_store.update(thread_id=request.get("thread_id", ""), cwd=request.get("cwd", "."))
            return {"ok": True, "state": state.to_dict()}
        if command in {"text", "steer"}:
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
            self.codex = CodexAppServer(self.config, self.state_store)
            self.codex.start()
        return self.codex

    def _listen_blockers(self) -> list[str]:
        checks = [
            audio_readiness(self.config, probe_stream=True),
            vad_readiness(),
            wake_readiness(self.config),
        ]
        return [check.reason for check in checks if not check.ok]

    async def _listen_loop(self) -> None:
        try:
            while self.state_store.load().listening:
                result = await asyncio.to_thread(self._run_voice_turn)
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

    def _run_voice_turn(self):
        config = self.config.with_overrides(self.listen_overrides)
        return self._voice_pipeline(send=True, config=config).run_once(MicCapture(config).frames())

    def _run_voice_turn_from_wav(self, wav_path: Path, send: bool, overrides: dict[str, Any] | None = None):
        config = self.config.with_overrides(overrides)
        return self._voice_pipeline(send=send, config=config).run_once(wav_frames(config, wav_path))

    def _voice_pipeline(self, send: bool, config: Config | None = None) -> VoicePipeline:
        config = config or self.config
        delivery_config = self._voice_delivery_config(config)
        deliver_text = (lambda text: self._codex().deliver_text(text, config=delivery_config)) if send else (lambda _text: None)
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
        return False
    pid_path = expand_path(str(config.get("server.pid_path")))
    pid = int(pid_path.read_text().strip())
    os.kill(pid, signal.SIGTERM)
    deadline = time.time() + 5
    while time.time() < deadline:
        if not is_running(config):
            StateStore(expand_path(str(config.get("server.state_db")))).update(active_turn_id="", listening=False)
            return True
        time.sleep(0.1)
    os.kill(pid, signal.SIGKILL)
    StateStore(expand_path(str(config.get("server.state_db")))).update(active_turn_id="", listening=False)
    return True


async def ensure_daemon(config: Config, no_start: bool = False) -> None:
    if is_running(config):
        return
    if no_start:
        raise RuntimeError("cxv daemon is not running")
    start_background(config)


def run_serve() -> None:
    asyncio.run(CxvDaemon(load_config()).serve())
