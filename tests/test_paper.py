import asyncio

import pytest

from zetaline.core.events import EventBus, Topic
from zetaline.core.models import Fill, Order, OrderType, Rejection, Side, Tick
from zetaline.execution.paper import PaperExecutor
from zetaline.execution.portfolio import Portfolio
from zetaline.feed.book import OrderBook


def make(fee_bps: float = 5.0):
    bus, book, pf = EventBus(), OrderBook(), Portfolio(100_000)
    ex = PaperExecutor(bus, book, pf, fee_bps=fee_bps)
    return bus, book, pf, ex


def quote(book: OrderBook, bid: float = 99.0, ask: float = 101.0, symbol: str = "BTC") -> None:
    book.update(Tick(symbol, bid=bid, ask=ask, bid_size=1, ask_size=1))


def test_negative_fee_rejected():
    with pytest.raises(ValueError):
        PaperExecutor(EventBus(), OrderBook(), Portfolio(1_000), fee_bps=-1)


def test_reject_when_no_market_data():
    bus, _, pf, ex = make()
    rejs = bus.subscribe(Topic.REJECT)
    result = ex.execute(Order("BTC", Side.BUY, 1))
    assert isinstance(result, Rejection)
    assert "no market data" in result.reason
    assert result.source == PaperExecutor.SOURCE
    assert ex.rejects == 1 and ex.fills == 0
    assert rejs.qsize() == 1
    assert pf.fills == 0


def test_buy_fills_at_ask_with_fee():
    _, book, pf, ex = make(fee_bps=5)
    quote(book)
    result = ex.execute(Order("BTC", Side.BUY, 2))
    assert isinstance(result, Fill)
    assert result.price == 101.0
    assert result.fee == pytest.approx(2 * 101.0 * 5 / 10_000)
    assert pf.position("BTC").qty == 2
    assert pf.cash == pytest.approx(100_000 - 202.0 - result.fee)


def test_sell_fills_at_bid():
    _, book, pf, ex = make()
    quote(book)
    result = ex.execute(Order("BTC", Side.SELL, 1))
    assert result.price == 99.0
    assert pf.position("BTC").qty == -1


def test_marketable_limit_buy_fills_at_ask():
    _, book, _, ex = make()
    quote(book)
    result = ex.execute(Order("BTC", Side.BUY, 1, OrderType.LIMIT, limit_price=105.0))
    assert isinstance(result, Fill)
    assert result.price == 101.0


def test_non_marketable_limit_buy_rejected():
    _, book, pf, ex = make()
    quote(book)
    result = ex.execute(Order("BTC", Side.BUY, 1, OrderType.LIMIT, limit_price=100.0))
    assert isinstance(result, Rejection)
    assert "not marketable" in result.reason
    assert pf.fills == 0


def test_non_marketable_limit_sell_rejected():
    _, book, _, ex = make()
    quote(book)
    result = ex.execute(Order("BTC", Side.SELL, 1, OrderType.LIMIT, limit_price=100.0))
    assert isinstance(result, Rejection)


def test_marketable_limit_sell_fills_at_bid():
    _, book, _, ex = make()
    quote(book)
    result = ex.execute(Order("BTC", Side.SELL, 1, OrderType.LIMIT, limit_price=98.0))
    assert isinstance(result, Fill)
    assert result.price == 99.0


def test_fill_is_published_to_bus():
    bus, book, _, ex = make()
    fills = bus.subscribe(Topic.FILL)
    quote(book)
    result = ex.execute(Order("BTC", Side.BUY, 1))
    assert fills.get_nowait() is result


def test_round_trip_pays_spread_and_fees():
    _, book, pf, ex = make(fee_bps=5)
    quote(book)
    ex.execute(Order("BTC", Side.BUY, 1))
    ex.execute(Order("BTC", Side.SELL, 1))
    assert pf.positions() == {}
    expected = -(101.0 - 99.0) - (101.0 + 99.0) * 5 / 10_000
    assert pf.total_pnl(book) == pytest.approx(expected)
    assert pf.total_pnl(book) == pytest.approx(pf.realized_pnl + pf.unrealized_pnl(book))


async def test_run_consumes_orders_from_bus():
    bus, book, pf, ex = make()
    quote(book)
    bus.publish(Topic.ORDER, Order("BTC", Side.BUY, 1))
    bus.publish(Topic.ORDER, Order("BTC", Side.BUY, 1))
    task = asyncio.create_task(ex.run())
    await asyncio.sleep(0.01)
    ex.stop()
    await asyncio.wait_for(task, timeout=1)
    assert ex.fills == 2
    assert pf.position("BTC").qty == 2


async def test_orders_published_before_run_are_not_lost():
    bus, book, _, ex = make()
    quote(book)
    bus.publish(Topic.ORDER, Order("BTC", Side.BUY, 1))
    task = asyncio.create_task(ex.run())
    await asyncio.sleep(0.01)
    ex.stop()
    await asyncio.wait_for(task, timeout=1)
    assert ex.fills == 1
