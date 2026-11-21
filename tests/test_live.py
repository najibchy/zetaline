import asyncio
import json
import time

import pytest
from websockets.asyncio.server import serve

from zetaline.core.events import EventBus, Topic
from zetaline.feed.live import LiveFeed


def msg(symbol: str = "BTCUSDT", bid: str = "100.00", ask: str = "100.10") -> str:
    data = {"u": 1, "s": symbol, "b": bid, "B": "1.0", "a": ask, "A": "2.0"}
    return json.dumps({"stream": f"{symbol.lower()}@bookTicker", "data": data})


@pytest.fixture
async def fake_server():
    servers = []

    async def start(handler) -> str:
        server = await serve(handler, "127.0.0.1", 0)
        servers.append(server)
        port = server.sockets[0].getsockname()[1]
        return f"ws://127.0.0.1:{port}/stream"

    yield start
    for server in servers:
        server.close()
        await server.wait_closed()


def make(bus: EventBus, url: str, **kw) -> LiveFeed:
    defaults = {"backoff_initial": 0.02, "backoff_max": 0.05}
    return LiveFeed(bus, ["BTCUSDT"], url=url, **{**defaults, **kw})


async def run(feed: LiveFeed, max_ticks: int, timeout: float = 5.0) -> None:
    await asyncio.wait_for(feed.run(max_ticks=max_ticks), timeout=timeout)


# ---- streaming ------------------------------------------------------------------
async def test_publishes_ticks_with_receive_time(fake_server):
    async def handler(ws):
        for bid in ("100.00", "101.00", "102.00"):
            await ws.send(msg(bid=bid, ask=str(float(bid) + 0.1)))
        await ws.wait_closed()

    url = await fake_server(handler)
    bus = EventBus()
    sub = bus.subscribe(Topic.TICK)
    feed = make(bus, url)
    before = time.perf_counter_ns()
    await run(feed, 3)
    ticks = [sub.get_nowait() for _ in range(3)]
    assert [t.bid for t in ticks] == [100.0, 101.0, 102.0]
    assert all(before <= t.recv_ns <= time.perf_counter_ns() for t in ticks)
    assert feed.ticks_published == 3 and feed.connects == 1
    assert feed.running is False and feed.connected is False


async def test_bad_and_foreign_messages_are_counted_not_fatal(fake_server):
    async def handler(ws):
        await ws.send("garbage")                                  # parse error
        await ws.send('{"result":null,"id":1}')                   # ack: silently skipped
        await ws.send(msg("ETHUSDT"))                             # symbol we did not ask for
        await ws.send(msg(bid="101.00", ask="100.00"))            # crossed book: parse error
        await ws.send(msg())                                      # valid
        await ws.wait_closed()

    url = await fake_server(handler)
    feed = make(EventBus(), url)
    await run(feed, 1)
    assert feed.ticks_published == 1
    assert feed.messages == 5
    assert feed.parse_errors == 2
    assert feed.ignored == 1
    assert feed.failures == 0


# ---- resilience ------------------------------------------------------------------
async def test_reconnects_after_server_closes(fake_server):
    conns = []

    async def handler(ws):
        conns.append(1)
        await ws.send(msg())
        if len(conns) == 1:
            return  # close the first connection
        await ws.wait_closed()

    url = await fake_server(handler)
    feed = make(EventBus(), url)
    await run(feed, 2)
    assert feed.connects == 2
    assert feed.failures == 1
    assert feed.ticks_published == 2


async def test_idle_watchdog_forces_reconnect(fake_server):
    conns = []

    async def handler(ws):
        conns.append(1)
        if len(conns) == 1:
            await ws.wait_closed()  # connected but silent
            return
        await ws.send(msg())
        await ws.wait_closed()

    url = await fake_server(handler)
    feed = make(EventBus(), url, idle_timeout=0.2)
    await run(feed, 1)
    assert feed.connects == 2
    assert feed.failures == 1
    assert "IdleTimeout" in feed.last_error


async def test_planned_rotation_reconnects_without_failure(fake_server):
    async def handler(ws):
        await ws.send(msg())
        await ws.wait_closed()

    url = await fake_server(handler)
    feed = make(EventBus(), url, max_session_s=0.3, idle_timeout=30.0)
    await run(feed, 2)
    assert feed.rotations == 1
    assert feed.connects == 2
    assert feed.failures == 0


async def test_server_shutdown_event_reconnects(fake_server):
    conns = []

    async def handler(ws):
        conns.append(1)
        await ws.send(msg())
        if len(conns) == 1:
            await ws.send('{"e":"serverShutdown","E":1}')
        await ws.wait_closed()

    url = await fake_server(handler)
    feed = make(EventBus(), url)
    await run(feed, 2)
    assert feed.shutdown_events == 1
    assert feed.connects == 2
    assert feed.failures == 0


async def test_cancellation_is_clean(fake_server):
    async def handler(ws):
        await ws.wait_closed()

    url = await fake_server(handler)
    feed = make(EventBus(), url)
    task = asyncio.create_task(feed.run())
    for _ in range(100):
        if feed.connected:
            break
        await asyncio.sleep(0.02)
    assert feed.connected and feed.running
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert feed.running is False and feed.connected is False


async def test_refused_endpoint_counts_failures_and_keeps_trying():
    feed = LiveFeed(
        EventBus(), ["BTCUSDT"], url="ws://127.0.0.1:9/ws",
        backoff_initial=0.02, backoff_max=0.05,
    )
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(feed.run(), timeout=0.5)
    assert feed.failures >= 2
    assert feed.connects == 0
    assert feed.running is False
    assert "ConnectionRefusedError" in feed.last_error


def test_backoff_grows_and_is_capped():
    feed = LiveFeed(EventBus(), ["BTCUSDT"], backoff_initial=1.0, backoff_max=8.0)
    for attempt, cap in [(1, 1.0), (2, 2.0), (3, 4.0), (4, 8.0), (5, 8.0), (20, 8.0)]:
        for _ in range(50):
            assert 0.5 * cap <= feed._backoff(attempt) <= cap


# ---- construction and stats --------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs",
    [
        {"open_timeout": 0},
        {"idle_timeout": -1},
        {"max_session_s": 0},
        {"backoff_initial": 0},
        {"backoff_max": 0.5},  # below the default backoff_initial of 1.0
    ],
)
def test_constructor_rejects_bad_settings(kwargs):
    with pytest.raises(ValueError):
        LiveFeed(EventBus(), ["BTCUSDT"], **kwargs)


@pytest.mark.parametrize("symbols", [[], ["btc/usdt"]])
def test_constructor_rejects_bad_symbols(symbols):
    with pytest.raises(ValueError):
        LiveFeed(EventBus(), symbols)


def test_symbols_are_uppercased():
    assert LiveFeed(EventBus(), ["btcusdt", "ethusdt"]).symbols == ["BTCUSDT", "ETHUSDT"]


async def test_stats_shape_before_and_after(fake_server):
    async def handler(ws):
        await ws.send(msg())
        await ws.wait_closed()

    url = await fake_server(handler)
    feed = make(EventBus(), url)
    before = feed.stats()
    assert before["connected"] is False and before["last_message_age_s"] is None
    assert before["reconnects"] == 0
    await run(feed, 1)
    after = feed.stats()
    assert after["messages"] == 1 and after["ticks_published"] == 1
    assert after["last_message_age_s"] is not None
    assert {"url", "failures", "rotations", "parse_errors", "ignored", "last_error"} <= set(after)
