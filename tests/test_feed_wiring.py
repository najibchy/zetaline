import asyncio
import json

import pytest
from websockets.asyncio.server import serve

from zetaline.control import ControlClient, DaemonNotRunning
from zetaline.daemon import main, run_daemon
from zetaline.runtime import Platform, PlatformConfig


def msg(symbol: str = "BTCUSDT", bid: str = "100.00", ask: str = "100.10") -> str:
    data = {"u": 1, "s": symbol, "b": bid, "B": "1.0", "a": ask, "A": "2.0"}
    return json.dumps({"stream": f"{symbol.lower()}@bookTicker", "data": data})


@pytest.fixture
async def fake_server():
    servers = []

    async def start(handler) -> str:
        server = await serve(handler, "127.0.0.1", 0)
        servers.append(server)
        return f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/stream"

    yield start
    for server in servers:
        server.close()
        await server.wait_closed()


def live_config(url: str) -> PlatformConfig:
    return PlatformConfig(feed="live", prices={"BTCUSDT": 1.0}, feed_url=url)


class CrashingFeed:
    symbols = ("X",)
    ticks_published = 0

    async def run(self):
        raise RuntimeError("boom")


# ---- config ---------------------------------------------------------------------
def test_config_rejects_unknown_feed():
    with pytest.raises(ValueError, match="feed must be"):
        PlatformConfig(feed="bogus")


def test_config_feed_defaults():
    cfg = PlatformConfig()
    assert cfg.feed == "sim" and cfg.testnet is False and cfg.feed_url is None


def test_live_platform_rejects_bad_symbol():
    with pytest.raises(ValueError, match="invalid symbol"):
        Platform(PlatformConfig(feed="live", prices={"btc/usdt": 1.0}))


# ---- platform ---------------------------------------------------------------------
async def test_sim_status_has_feed_block():
    p = Platform(PlatformConfig(rate_hz=500.0, volatility=0.002, seed=1))
    assert p.status()["feed"]["alive"] is False
    await p.start()
    await asyncio.sleep(0.1)
    feed = p.status()["feed"]
    assert feed["kind"] == "sim" and feed["alive"] is True
    assert feed["ticks_published"] > 0 and feed["error"] == ""
    await p.stop()
    assert p.status()["feed"]["alive"] is False


async def test_live_platform_consumes_fake_feed(fake_server):
    async def handler(ws):
        for i in range(30):
            await ws.send(msg(bid=f"{100 + i * 0.01:.2f}", ask=f"{100.1 + i * 0.01:.2f}"))
        await ws.wait_closed()

    p = Platform(live_config(await fake_server(handler)))
    assert p.feed.symbols == ["BTCUSDT"]
    await p.start()
    await asyncio.sleep(0.4)
    s = p.status()
    assert s["feed"]["kind"] == "live"
    assert s["feed"]["connected"] is True and s["feed"]["alive"] is True
    assert s["feed"]["messages"] == 30 and s["feed"]["parse_errors"] == 0
    assert s["ticks"] == 30
    assert p.prices()["BTCUSDT"]["mid"] == pytest.approx(100.34, abs=1e-6)
    json.dumps(s)  # the status, including feed stats, must stay JSON-safe
    await p.stop()
    s = p.status()
    assert s["feed"]["alive"] is False and s["feed"]["connected"] is False
    assert s["intents"] == s["approved"] + s["rejected"]
    assert s["fills"] == s["approved"]


async def test_live_platform_survives_refused_endpoint():
    p = Platform(live_config("ws://127.0.0.1:9/ws"))
    await p.start()
    await asyncio.sleep(0.3)
    feed = p.status()["feed"]
    assert feed["alive"] is True and feed["connected"] is False
    assert feed["failures"] >= 1 and "ConnectionRefusedError" in feed["last_error"]
    assert feed["error"] == ""  # retrying is not a crash
    await p.stop()


async def test_feed_crash_is_recorded_not_raised():
    p = Platform(PlatformConfig())
    p.feed = CrashingFeed()
    await p.start()
    await asyncio.sleep(0.05)
    feed = p.status()["feed"]
    assert feed["alive"] is False and "RuntimeError: boom" in feed["error"]
    assert "feed_error" in [e["type"] for e in p.recent_events()]
    await p.stop()  # must not raise
    assert "boom" in p.feed_error


# ---- daemon ------------------------------------------------------------------------
def test_daemon_rejects_live_only_flags_on_sim():
    with pytest.raises(SystemExit) as exc:
        main(["--testnet"])
    assert exc.value.code == 2


def test_daemon_bad_symbol_exits_1(tmp_path, capsys):
    code = main(["--feed", "live", "--symbols", "btc/usdt", "--socket", str(tmp_path / "d.sock")])
    assert code == 1
    assert "invalid symbol" in capsys.readouterr().err


async def test_run_daemon_serves_and_shuts_down(tmp_path):
    sock = tmp_path / "d.sock"
    task = asyncio.create_task(run_daemon(PlatformConfig(rate_hz=200.0, seed=1), str(sock)))
    client = ControlClient(sock)
    for _ in range(150):
        try:
            if await client.call("ping") == "pong":
                break
        except DaemonNotRunning:
            await asyncio.sleep(0.02)
    else:
        pytest.fail("daemon never answered")
    assert (await client.call("status"))["feed"]["kind"] == "sim"
    await client.call("shutdown")
    await asyncio.wait_for(task, timeout=5)
    assert not sock.exists()
