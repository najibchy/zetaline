"""End-to-end: strategy intent -> risk manager -> paper executor -> portfolio."""
import asyncio

import pytest

from zetaline.core.events import EventBus, Topic
from zetaline.core.models import Order, Side, Tick
from zetaline.execution.paper import PaperExecutor
from zetaline.execution.portfolio import Portfolio
from zetaline.feed.book import OrderBook
from zetaline.risk.manager import RiskManager


class Pipeline:
    def __init__(self) -> None:
        self.bus = EventBus()
        self.book = OrderBook()
        self.pf = Portfolio(100_000.0)
        self.rm = RiskManager(self.bus, self.book, self.pf)
        self.ex = PaperExecutor(self.bus, self.book, self.pf, fee_bps=5.0)
        self.fills = self.bus.subscribe(Topic.FILL)
        self.rejects = self.bus.subscribe(Topic.REJECT)
        self.tasks: list[asyncio.Task] = []

    def quote(self, bid: float, ask: float, symbol: str = "BTC") -> None:
        self.book.update(Tick(symbol, bid=bid, ask=ask, bid_size=1, ask_size=1))

    def start(self) -> None:
        self.tasks = [
            asyncio.create_task(self.rm.run()),
            asyncio.create_task(self.ex.run()),
        ]

    async def settle(self) -> None:
        await asyncio.sleep(0.02)

    async def stop(self) -> None:
        self.rm.stop()
        self.ex.stop()
        await asyncio.wait_for(asyncio.gather(*self.tasks), timeout=1)

    def intend(self, side: Side, qty: float) -> None:
        self.bus.publish(Topic.INTENT, Order("BTC", side, qty, strategy="test"))


async def test_intent_flows_to_fill_and_portfolio():
    p = Pipeline()
    p.quote(99.0, 101.0)
    p.start()
    p.intend(Side.BUY, 10)
    await p.settle()
    await p.stop()

    assert p.rm.approved == 1 and p.rm.rejected == 0
    assert p.ex.fills == 1
    fill = p.fills.get_nowait()
    assert fill.price == 101.0
    assert p.pf.position("BTC").qty == 10
    assert p.pf.cash == pytest.approx(100_000 - 1_010.0 - fill.fee)


async def test_risk_rejection_never_reaches_executor():
    p = Pipeline()
    p.quote(99.0, 101.0)
    p.start()
    p.intend(Side.BUY, 2_000)  # exceeds max order qty
    await p.settle()
    await p.stop()

    assert p.rm.rejected == 1
    assert p.ex.fills == 0 and p.ex.rejects == 0
    assert p.pf.fills == 0
    rejection = p.rejects.get_nowait()
    assert rejection.source == RiskManager.SOURCE


async def test_kill_switch_blocks_new_risk_but_allows_exit():
    p = Pipeline()
    p.quote(99.0, 101.0)
    p.start()
    p.intend(Side.BUY, 10)
    await p.settle()
    assert p.pf.position("BTC").qty == 10

    p.rm.trip_kill_switch("test")
    p.intend(Side.BUY, 1)    # blocked
    p.intend(Side.SELL, 10)  # reduction, allowed
    await p.settle()
    await p.stop()

    assert p.rm.rejected == 1
    assert p.ex.fills == 2
    assert p.pf.positions() == {}


async def test_max_loss_trips_kill_switch_end_to_end():
    p = Pipeline()
    p.quote(99.0, 101.0)
    p.start()
    p.intend(Side.BUY, 50)  # 5,050 notional, fills at 101
    await p.settle()
    assert p.pf.position("BTC").qty == 50

    p.quote(49.0, 51.0)  # mid 50: pnl about -2,550, beyond the 2,000 limit
    p.intend(Side.BUY, 1)
    await p.settle()
    assert p.rm.kill_switch is True
    assert "max loss" in p.rm.kill_reason

    p.intend(Side.SELL, 50)  # exit still allowed, fills at the bid of 49
    await p.settle()
    await p.stop()

    assert p.pf.positions() == {}
    assert p.ex.fills == 2
    assert p.pf.realized_pnl < -2_000
