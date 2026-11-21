import asyncio
import json
import os
import stat

import pytest

from zetaline.control import (
    ControlClient,
    ControlError,
    ControlServer,
    DaemonNotRunning,
    default_socket_path,
)
from zetaline.runtime import Platform, PlatformConfig


def make_platform() -> Platform:
    return Platform(
        PlatformConfig(rate_hz=500.0, volatility=0.002, fast=3, slow=6, seed=1)
    )


@pytest.fixture
async def daemon(tmp_path):
    platform = make_platform()
    await platform.start()
    server = ControlServer(platform, tmp_path / "z.sock")
    await server.start()
    yield platform, server
    await server.stop()
    await platform.stop()


def req(cmd, args=None, id_=1) -> bytes:
    return json.dumps({"id": id_, "cmd": cmd, "args": args or {}}).encode()


# ---- dispatch (no socket) ---------------------------------------------------
def test_dispatch_ping_and_status():
    server = ControlServer(make_platform(), "/tmp/unused.sock")
    assert server.dispatch(req("ping")) == {"id": 1, "ok": True, "result": "pong"}
    st = server.dispatch(req("status", id_=2))
    assert st["ok"] and st["id"] == 2 and "portfolio" in st["result"]


def test_dispatch_invalid_json():
    server = ControlServer(make_platform(), "/tmp/unused.sock")
    out = server.dispatch(b"not json")
    assert out["ok"] is False and out["error"] == "invalid JSON" and out["id"] is None


@pytest.mark.parametrize("raw", [b"[1, 2]", b'"str"', b"42"])
def test_dispatch_non_object_rejected(raw):
    out = ControlServer(make_platform(), "/tmp/unused.sock").dispatch(raw)
    assert out["ok"] is False and "TypeError" in out["error"]


def test_dispatch_args_must_be_object():
    out = ControlServer(make_platform(), "/tmp/unused.sock").dispatch(
        b'{"id": 3, "cmd": "ping", "args": [1]}'
    )
    assert out["ok"] is False and out["id"] == 3


@pytest.mark.parametrize("cmd", ["bogus", "__class__", "platform", "", None, 5])
def test_dispatch_unknown_or_non_allowlisted_command(cmd):
    server = ControlServer(make_platform(), "/tmp/unused.sock")
    out = server.dispatch(json.dumps({"id": 1, "cmd": cmd}).encode())
    assert out["ok"] is False and "unknown command" in out["error"]


def test_dispatch_bad_arguments_become_errors():
    server = ControlServer(make_platform(), "/tmp/unused.sock")
    assert server.dispatch(req("set_limits", {"max_loss": -1}))["ok"] is False
    assert server.dispatch(req("set_limits", {"nope": 1}))["ok"] is False
    assert server.dispatch(req("set_strategy_params", {"name": "missing"}))["ok"] is False
    assert server.dispatch(req("fills", {"n": "x"}))["ok"] is False
    assert server.dispatch(req("ping", {"unexpected": 1}))["ok"] is False


def test_dispatch_set_limits_returns_json_pairs():
    server = ControlServer(make_platform(), "/tmp/unused.sock")
    out = server.dispatch(req("set_limits", {"max_loss": 1_500.0}))
    assert out["result"] == {"max_loss": [2_000.0, 1_500.0]}


def test_dispatch_shutdown_sets_event():
    server = ControlServer(make_platform(), "/tmp/unused.sock")
    assert not server.shutdown_event.is_set()
    assert server.dispatch(req("shutdown"))["ok"]
    assert server.shutdown_event.is_set()


def test_default_socket_path_is_user_scoped():
    assert default_socket_path().name.startswith("zetaline")


# ---- real socket --------------------------------------------------------------
async def test_client_without_daemon(tmp_path):
    with pytest.raises(DaemonNotRunning):
        await ControlClient(tmp_path / "missing.sock").call("ping")


async def test_socket_mode_is_0600(daemon):
    _, server = daemon
    assert stat.S_IMODE(os.stat(server.path).st_mode) == 0o600


async def test_round_trip_commands(daemon):
    _, server = daemon
    c = ControlClient(server.path)
    assert await c.call("ping") == "pong"
    await asyncio.sleep(0.2)
    st = await c.call("status")
    assert st["running"] and st["ticks"] > 0
    assert sorted(await c.call("prices")) == sorted(st["symbols"])
    assert await c.call("set_limits", max_loss=1_500.0) == {"max_loss": [2_000.0, 1_500.0]}
    assert isinstance(await c.call("fills", n=5), list)


async def test_kill_and_resume_over_socket(daemon):
    _, server = daemon
    c = ControlClient(server.path)
    await c.call("kill", reason="test")
    assert (await c.call("status"))["kill_switch"] is True
    assert await c.call("resume") is True
    assert (await c.call("status"))["kill_switch"] is False


async def test_error_response_raises_control_error(daemon):
    _, server = daemon
    with pytest.raises(ControlError):
        await ControlClient(server.path).call("set_limits", max_loss=-1)


async def test_bad_json_keeps_connection_open(daemon):
    _, server = daemon
    reader, writer = await asyncio.open_unix_connection(str(server.path))
    writer.write(b"garbage\n")
    await writer.drain()
    assert json.loads(await reader.readline())["ok"] is False
    writer.write(b'{"id": 9, "cmd": "ping"}\n')
    await writer.drain()
    assert json.loads(await reader.readline()) == {"id": 9, "ok": True, "result": "pong"}
    writer.close()


async def test_second_daemon_refused_on_live_socket(daemon):
    platform, server = daemon
    other = ControlServer(platform, server.path)
    with pytest.raises(RuntimeError, match="already listening"):
        await other.start()
    # The live daemon's socket must survive the refused attempt.
    assert await ControlClient(server.path).call("ping") == "pong"


async def test_stale_socket_file_is_replaced(tmp_path):
    path = tmp_path / "stale.sock"
    path.write_text("")  # a leftover file nobody is listening on
    platform = make_platform()
    server = ControlServer(platform, path)
    await server.start()
    assert await ControlClient(path).call("ping") == "pong"
    await server.stop()
    assert not path.exists()


async def test_stop_removes_socket(tmp_path):
    path = tmp_path / "z.sock"
    server = ControlServer(make_platform(), path)
    await server.start()
    assert path.exists()
    await server.stop()
    assert not path.exists()
