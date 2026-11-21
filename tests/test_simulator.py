import asyncio
import time

import pytest

from zetaline.core.events import EventBus, Topic
from zetaline.feed.book import OrderBook
from zetaline.feed.simulator import DEFAULT_PRICES, MarketSimulator


def mids(seed: int, n: int = 5) -> list[float]:
    sim = MarketSimulator(EventBus(), seed=seed)
    return [sim.next_tick("BTCUSDT").mid for _ in range(n)]


def test_same_seed_is_reproducible():
    assert mids(7) == mids(7)


def test_different_seed_differs():
    assert mids(7) != mids(8)


def test_tick_is_valid():
    sim = MarketSimulator(EventBus(), seed=1)
    for _ in range(200):
        for symbol in sim.symbols:
            t = sim.next_tick(symbol)
            assert t.bid > 0
            assert t.bid < t.ask
            assert t.bid_size > 0 and t.ask_size > 0


def test_spread_matches_spread_bps():
    sim = MarketSimulator(EventBus(), seed=1, spread_bps=2.0)
    t = sim.next_tick("BTCUSDT")
    assert t.spread / t.mid == pytest.approx(2.0 / 10_000, rel=1e-9)


def test_zero_volatility_keeps_price_flat():
    sim = MarketSimulator(EventBus(), prices={"X": 100.0}, volatility=0.0)
    assert [sim.next_tick("X").mid for _ in range(3)] == [100.0, 100.0, 100.0]


def test_default_symbols():
    sim = MarketSimulator(EventBus())
    assert sim.symbols == list(DEFAULT_PRICES)


def test_step_publishes_one_tick_per_symbol():
    bus = EventBus()
    sub = bus.subscribe(Topic.TICK)
    sim = MarketSimulator(bus)
    assert sim.step() == 3
    assert sub.qsize() == 3
    assert sim.ticks_published == 3


async def test_run_stops_after_max_ticks():
    bus = EventBus()
    sub = bus.subscribe(Topic.TICK)
    sim = MarketSimulator(bus, rate_hz=0)
    await sim.run(max_ticks=5)
    assert sim.ticks_published == 15
    assert sub.qsize() == 15
    assert sim.running is False


async def test_running_flag_is_true_while_running():
    sim = MarketSimulator(EventBus(), rate_hz=0)
    task = asyncio.create_task(sim.run())
    await asyncio.sleep(0.01)
    assert sim.running is True
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert sim.running is False


async def test_rate_limit_is_respected():
    sim = MarketSimulator(EventBus(), prices={"X": 100.0}, rate_hz=200)
    start = time.perf_counter()
    await sim.run(max_ticks=10)
    elapsed = time.perf_counter() - start
    # 10 rounds at 200 Hz is about 45 ms (the first round has no delay). Generous upper bound.
    assert elapsed >= 0.03
    assert elapsed < 0.5


async def test_book_consumes_feed():
    bus = EventBus()
    sub = bus.subscribe(Topic.TICK)
    book = OrderBook()
    sim = MarketSimulator(bus, rate_hz=0)
    await sim.run(max_ticks=3)
    while sub.qsize():
        book.update(sub.get_nowait())
    assert book.symbols() == sorted(DEFAULT_PRICES)
    assert book.updates == 9


def test_invalid_arguments():
    with pytest.raises(ValueError):
        MarketSimulator(EventBus(), rate_hz=-1)
    with pytest.raises(ValueError):
        MarketSimulator(EventBus(), volatility=-0.1)
    with pytest.raises(ValueError):
        MarketSimulator(EventBus(), spread_bps=-1)
    with pytest.raises(ValueError):
        MarketSimulator(EventBus(), prices={})
