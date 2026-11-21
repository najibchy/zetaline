"""Control socket: newline-delimited JSON over a Unix domain socket.

Request:  {"id": 1, "cmd": "status", "args": {}}
Response: {"id": 1, "ok": true, "result": ...} | {"id": 1, "ok": false, "error": "..."}

Only commands in the explicit allowlist can be invoked. The socket is mode 0600.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from zetaline.runtime import Platform

MAX_LINE = 1_048_576  # 1 MiB


def default_socket_path() -> Path:
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if runtime_dir and Path(runtime_dir).is_dir():
        return Path(runtime_dir) / "zetaline.sock"
    return Path(f"/tmp/zetaline-{os.getuid()}.sock")


class ControlError(Exception):
    """The daemon returned ok=false."""


class DaemonNotRunning(ConnectionError):
    """No daemon is listening on the socket."""


class ControlServer:
    def __init__(self, platform: Platform, path: Path | str | None = None) -> None:
        self.platform = platform
        self.path = Path(path) if path is not None else default_socket_path()
        self.shutdown_event = asyncio.Event()
        self._server: asyncio.AbstractServer | None = None
        self._handlers: dict[str, Callable[..., Any]] = {
            "ping": lambda: "pong",
            "status": platform.status,
            "positions": platform.positions,
            "prices": platform.prices,
            "latency": platform.latency_summary,
            "fills": lambda n=20: platform.recent_fills(int(n)),
            "rejects": lambda n=20: platform.recent_rejects(int(n)),
            "events": lambda n=20: platform.recent_events(int(n)),
            "kill": lambda reason="manual": platform.kill(str(reason)),
            "resume": platform.resume,
            "set_limits": lambda **kw: _pairs(platform.set_limits(**kw)),
            "set_strategy_params": lambda name, **kw: _pairs(
                platform.set_strategy_params(name, **kw)
            ),
            "set_strategy_enabled": lambda name, enabled: platform.set_strategy_enabled(
                name, bool(enabled)
            ),
            "shutdown": self._request_shutdown,
        }

    def _request_shutdown(self) -> str:
        self.shutdown_event.set()
        return "shutting down"

    async def start(self) -> None:
        await self._remove_stale_socket()
        self._server = await asyncio.start_unix_server(
            self._handle, path=str(self.path), limit=MAX_LINE
        )
        os.chmod(self.path, 0o600)

    async def _remove_stale_socket(self) -> None:
        if not self.path.exists():
            return
        try:
            _, writer = await asyncio.open_unix_connection(str(self.path))
        except (ConnectionRefusedError, FileNotFoundError):
            self.path.unlink(missing_ok=True)  # stale: nobody is listening
            return
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()
        raise RuntimeError(f"a daemon is already listening on {self.path}")

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        self.path.unlink(missing_ok=True)

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                try:
                    line = await reader.readline()
                except (ValueError, asyncio.LimitOverrunError):
                    await self._send(writer, {"id": None, "ok": False, "error": "line too long"})
                    break
                if not line:
                    break
                await self._send(writer, self.dispatch(line))
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    @staticmethod
    async def _send(writer: asyncio.StreamWriter, message: dict[str, Any]) -> None:
        writer.write(json.dumps(message, default=str).encode() + b"\n")
        await writer.drain()

    def dispatch(self, line: bytes) -> dict[str, Any]:
        """Parse one request line and run it. Never raises."""
        req_id: Any = None
        try:
            req = json.loads(line)
            if not isinstance(req, dict):
                raise TypeError("request must be a JSON object")
            req_id = req.get("id")
            cmd = req.get("cmd")
            args = req.get("args") or {}
            if not isinstance(args, dict):
                raise TypeError("args must be an object")
            handler = self._handlers.get(cmd) if isinstance(cmd, str) else None
            if handler is None:
                raise ValueError(f"unknown command: {cmd!r}")
            return {"id": req_id, "ok": True, "result": handler(**args)}
        except json.JSONDecodeError:
            return {"id": req_id, "ok": False, "error": "invalid JSON"}
        except (ValueError, TypeError, KeyError) as exc:
            return {"id": req_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _pairs(applied: dict[str, tuple[Any, Any]]) -> dict[str, list[Any]]:
    """JSON-friendly {field: [old, new]}."""
    return {k: [old, new] for k, (old, new) in applied.items()}


class ControlClient:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else default_socket_path()
        self._next_id = 0

    async def call(self, cmd: str, **args: Any) -> Any:
        try:
            reader, writer = await asyncio.open_unix_connection(str(self.path), limit=MAX_LINE)
        except (FileNotFoundError, ConnectionRefusedError) as exc:
            raise DaemonNotRunning(f"no daemon listening on {self.path}") from exc
        try:
            self._next_id += 1
            request = {"id": self._next_id, "cmd": cmd, "args": args}
            writer.write(json.dumps(request).encode() + b"\n")
            await writer.drain()
            line = await reader.readline()
            if not line:
                raise ControlError("connection closed by daemon")
            response = json.loads(line)
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
        if not response.get("ok"):
            raise ControlError(response.get("error", "unknown error"))
        return response.get("result")
