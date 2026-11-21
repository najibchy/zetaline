import asyncio

import pytest

from zetaline.core.events import EventBus, Topic
from zetaline.core.models import Fill, Side, Tick
from zetaline.execution.paper import PaperExecutor
from zetaline.execution.portfolio import Portfolio
from zetaline.feed.book import OrderBook
from zetaline.feed.simulator import MarketSimulator
from zetaline.risk.manager import RiskManager
from zetaline.strategy.base import Strategy
from zetaline.strategy.momentum import MomentumStrategy
from zetaline.strategy.runner import StrategyRunner


def tick(price: float = 100.0, symbol: str = "BTC") -> Tick:
    return Tick(symbol, bid=price - 0.01, ask=price + 0.01, bid_size=1, ask_size=1)


class AlwaysBuy(Strategy):
    def on_tick(self, tick, position_qty):
        return self._order(tick, Side.BUY, 1)


class Broken(Strategy):
    def on_tick(self, tick, position_qty):
        raise RuntimeError("boom")


class Recorder(Strategy):
    def __init__(self, name: str = "rec") -> None:
        super().__init__(name)
        self.seen: list[float] = []

    def on_tick(self, tick, position_qty):
        self.seen.append(position_qty)


def make(strategies=None):
    bus, book, pf = EventBus(), OrderBook(), Portfolio(100_000.0)
    runner = StrategyRunner(bus, book, pf, strategies)
    return bus, book, pf, runner


def test_on_tick_updates_book_and_counts():
    _, book, _, runner = make()
    assert runner.on_tick(tick(100.0)) == 0
    assert runner.ticks == 1
    assert book.mid("BTC") == pytest.approx(100.0)


def test_publishes_intent_for_strategy_order():
    bus, _, _, runner = make([AlwaysBuy("a")])
    intents = bus.subscribe(Topic.INTENT)
    assert runner.on_tick(tick()) == 1
    assert runner.intents == 1
    order = intents.get_nowait()
    assert order.strategy == "a" and order.side is Side.BUY


def test_disabled_strategy_is_skipped():
    bus, _, _, runner = make([AlwaysBuy("a")])
    intents = bus.subscribe(Topic.INTENT)
    runner.set_enabled("a", False)
    assert runner.on_tick(tick()) == 0
    assert intents.qsize() == 0
    runner.set_enabled("a", True)
    assert runner.on_tick(tick()) == 1


def test_symbol_filter_respected():
    _, _, _, runner = make([AlwaysBuy("a", symbols=["ETH"])])
    assert runner.on_tick(tick(symbol="BTC")) == 0
    assert runner.on_tick(tick(symbol="ETH")) == 1


def test_strategy_receives_current_position():
    rec = Recorder()
    _, _, pf, runner = make([rec])
    runner.on_tick(tick())
    pf.apply_fill(Fill(1, "BTC", Side.BUY, 3.0, 100.0))
    runner.on_tick(tick())
    assert rec.seen == [0.0, 3.0]


def test_faulty_strategy_is_isolated_and_disabled():
    bus, _, _, runner = make([Broken("broken"), AlwaysBuy("good")])
    intents = bus.subscribe(Topic.INTENT)
    control = bus.subscribe(Topic.CONTROL)
    assert runner.on_tick(tick()) == 1  # the healthy strategy still ran
    assert runner.strategy_errors == 1
    assert runner.get("broken").enabled is False
    assert runner.get("good").enabled is True
    event = control.get_nowait()
    assert event["type"] == "strategy_error" and event["strategy"] == "broken"
    assert "RuntimeError: boom" in event["error"]
    runner.on_tick(tick())  # broken is now skipped: no new error
    assert runner.strategy_errors == 1
    assert intents.qsize() == 2


def test_duplicate_name_refused():
    _, _, _, runner = make([AlwaysBuy("a")])
    with pytest.raises(ValueError, match="already registered"):
        runner.add(AlwaysBuy("a"))


def test_remove_and_get():
    a = AlwaysBuy("a")
    _, _, _, runner = make([a])
    assert runner.get("a") is a
    assert runner.remove("a") is a
    assert runner.names() == []
    assert runner.on_tick(tick()) == 0


def test_unknown_strategy_raises():
    _, _, _, runner = make()
    with pytest.raises(KeyError):
        runner.get("nope")
    with pytest.raises(KeyError):
        runner.remove("nope")
    with pytest.raises(KeyError):
        runner.set_enabled("nope", True)


def test_names_sorted():
    _, _, _, runner = make([AlwaysBuy("b"), AlwaysBuy("a"), AlwaysBuy("c")])
    assert runner.names() == ["a", "b", "c"]


async def test_run_consumes_ticks_from_bus():
    bus, book, _, runner = make([AlwaysBuy("a")])
    intents = bus.subscribe(Topic.INTENT)
    bus.publish(Topic.TICK, tick(100.0))
    bus.publish(Topic.TICK, tick(101.0))
    task = asyncio.create_task(runner.run())
    await asyncio.sleep(0.01)
    runner.stop()
    await asyncio.wait_for(task, timeout=1)
    assert runner.ticks == 2
    assert intents.qsize() == 2
    assert book.mid("BTC") == pytest.approx(101.0)


async def test_simulator_to_portfolio_end_to_end():
    """Every real component on one bus: sim -> runner -> risk -> executor -> portfolio."""
    bus, book, pf = EventBus(), OrderBook(), Portfolio(100_000.0)
    runner = StrategyRunner(
        bus, book, pf,
        [MomentumStrategy(fast=3, slow=6, trade_qty=10.0, threshold_bps=1.0)],
    )
    rm = RiskManager(bus, book, pf)
    ex = PaperExecutor(bus, book, pf, fee_bps=5.0)
    sim = MarketSimulator(bus, prices={"X": 100.0}, seed=7, rate_hz=0, volatility=0.01)

    tasks = [
        asyncio.create_task(runner.run()),
        asyncio.create_task(rm.run()),
        asyncio.create_task(ex.run()),
    ]
    await sim.run(max_ticks=200)
    await asyncio.sleep(0.1)  # let the pipeline drain
    runner.stop()
    rm.stop()
    ex.stop()
    await asyncio.wait_for(asyncio.gather(*tasks), timeout=2)

    assert sim.ticks_published == 200
    assert runner.ticks == 200
    assert bus.stats()["dropped"]["tick"] == 0
    # Every intent was either approved or rejected by risk.
    assert runner.intents == rm.approved + rm.rejected
    # Every approved order was filled (the book always had data).
    assert ex.rejects == 0
    assert ex.fills == rm.approved
    assert pf.fills == ex.fills
    assert ex.fills > 0  # a volatile random walk must trigger the crossover
    # Accounting identity: equity - starting cash == realized + unrealized.
    assert pf.total_pnl(book) == pytest.approx(
        pf.realized_pnl + pf.unrealized_pnl(book), abs=1e-6
    )
