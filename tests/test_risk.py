import asyncio

import pytest

from zetaline.core.events import EventBus, Topic
from zetaline.core.models import Fill, Order, Rejection, Side, Tick
from zetaline.execution.portfolio import Portfolio
from zetaline.feed.book import OrderBook
from zetaline.risk.limits import RiskLimits
from zetaline.risk.manager import RiskManager


def make(cash: float = 100_000.0, limits: RiskLimits | None = None, clock=None):
    bus, book, pf = EventBus(), OrderBook(), Portfolio(cash)
    kwargs = {"clock_ns": clock} if clock is not None else {}
    rm = RiskManager(bus, book, pf, limits=limits, **kwargs)
    return bus, book, pf, rm


def quote(book: OrderBook, symbol: str = "BTC", bid: float = 99.0, ask: float = 101.0,
          ts_ns: int | None = None) -> None:
    kwargs = {} if ts_ns is None else {"ts_ns": ts_ns}
    book.update(Tick(symbol, bid=bid, ask=ask, bid_size=1, ask_size=1, **kwargs))


def buy_fill(pf: Portfolio, qty: float, price: float, symbol: str = "BTC") -> None:
    pf.apply_fill(Fill(1, symbol, Side.BUY, qty, price))


def test_negative_fee_buffer_rejected():
    with pytest.raises(ValueError):
        RiskManager(EventBus(), OrderBook(), Portfolio(1_000), fee_buffer_bps=-1)


def test_approve_small_buy():
    _, book, _, rm = make()
    quote(book)
    assert rm.check(Order("BTC", Side.BUY, 10)) is None


def test_reject_no_market_data():
    _, _, _, rm = make()
    assert "no market data" in rm.check(Order("BTC", Side.BUY, 1))


def test_stale_data_uses_injected_clock():
    now = [1_000_000_000]
    _, book, _, rm = make(clock=lambda: now[0])
    quote(book, ts_ns=0)
    assert rm.check(Order("BTC", Side.BUY, 1)) is None  # 1s old, limit 2s
    now[0] = 3_000_000_000
    assert "stale" in rm.check(Order("BTC", Side.BUY, 1))


def test_reject_order_qty():
    _, book, _, rm = make()
    quote(book)
    assert "qty" in rm.check(Order("BTC", Side.BUY, 2_000))


def test_reject_order_notional():
    _, book, _, rm = make()
    quote(book)
    assert "order notional" in rm.check(Order("BTC", Side.BUY, 100))  # 10,100 > 10,000


def test_reject_position_limit():
    _, book, pf, rm = make()
    quote(book)
    buy_fill(pf, 240, 100.0)
    assert "position notional" in rm.check(Order("BTC", Side.BUY, 20))


def test_reject_gross_exposure():
    limits = RiskLimits(
        max_order_notional=10_000.0,
        max_position_notional=20_000.0,
        max_gross_exposure=25_000.0,
    )
    _, book, pf, rm = make(limits=limits)
    quote(book, "BTC", 99.0, 101.0)
    quote(book, "ETH", 49.0, 51.0)
    buy_fill(pf, 200, 100.0, "BTC")  # exposure 20,000
    # ETH: 120 * mid 50 = 6,000 -> gross 26,000 > 25,000
    assert "gross exposure" in rm.check(Order("ETH", Side.BUY, 120))
    assert rm.check(Order("ETH", Side.BUY, 50)) is None  # 22,500 is fine


def test_reject_insufficient_cash():
    _, book, _, rm = make(cash=1_000.0)
    quote(book)
    assert rm.check(Order("BTC", Side.BUY, 5)) is None
    assert "insufficient cash" in rm.check(Order("BTC", Side.BUY, 10))


def test_short_sell_needs_no_cash():
    _, book, _, rm = make(cash=1_000.0)
    quote(book)
    assert rm.check(Order("BTC", Side.SELL, 10)) is None


def test_reduction_bypasses_position_and_exposure_limits():
    limits = RiskLimits(
        max_order_notional=1_000.0,
        max_position_notional=1_000.0,
        max_gross_exposure=1_000.0,
    )
    _, book, pf, rm = make(limits=limits)
    quote(book)
    buy_fill(pf, 240, 100.0)  # 24,000 held, far above the 1,000 limits
    assert rm.check(Order("BTC", Side.SELL, 10)) is None
    assert "position notional" in rm.check(Order("BTC", Side.BUY, 5))


def test_kill_switch_blocks_new_risk():
    _, book, _, rm = make()
    quote(book)
    rm.trip_kill_switch("manual")
    reason = rm.check(Order("BTC", Side.BUY, 1))
    assert "kill switch" in reason and "manual" in reason


def test_kill_switch_allows_reduction():
    _, book, pf, rm = make()
    quote(book)
    buy_fill(pf, 100, 100.0)
    rm.trip_kill_switch("manual")
    assert rm.check(Order("BTC", Side.SELL, 10)) is None
    assert rm.check(Order("BTC", Side.SELL, 100)) is None  # full close is a reduction


def test_kill_switch_blocks_flip():
    _, book, pf, rm = make()
    quote(book)
    buy_fill(pf, 100, 100.0)
    rm.trip_kill_switch("manual")
    assert "kill switch" in rm.check(Order("BTC", Side.SELL, 150))


def test_trip_publishes_control_once():
    bus, _, _, rm = make()
    control = bus.subscribe(Topic.CONTROL)
    rm.trip_kill_switch("first")
    rm.trip_kill_switch("second")
    assert control.qsize() == 1
    event = control.get_nowait()
    assert event["type"] == "kill_switch" and event["active"] is True
    assert event["reason"] == "first"
    assert rm.kill_reason == "first"


def test_reset_publishes_control():
    bus, _, _, rm = make()
    rm.trip_kill_switch("x")
    control = bus.subscribe(Topic.CONTROL)
    rm.reset_kill_switch()
    assert rm.kill_switch is False and rm.kill_reason == ""
    assert control.get_nowait()["active"] is False


def test_reset_is_noop_when_inactive():
    bus, _, _, rm = make()
    control = bus.subscribe(Topic.CONTROL)
    rm.reset_kill_switch()
    assert control.qsize() == 0


def test_max_loss_auto_trips_kill_switch():
    bus, book, pf, rm = make()
    control = bus.subscribe(Topic.CONTROL)
    buy_fill(pf, 100, 100.0)
    quote(book, bid=69.0, ask=71.0)  # mid 70 -> pnl -3,000 <= -2,000
    reason = rm.check(Order("BTC", Side.BUY, 1))
    assert "kill switch" in reason and "max loss" in reason
    assert rm.kill_switch is True
    assert control.qsize() == 1


def test_reset_retrips_if_loss_persists():
    _, book, pf, rm = make()
    buy_fill(pf, 100, 100.0)
    quote(book, bid=69.0, ask=71.0)
    rm.check(Order("BTC", Side.BUY, 1))
    rm.reset_kill_switch()
    assert rm.kill_switch is False
    assert "kill switch" in rm.check(Order("BTC", Side.BUY, 1))
    assert rm.kill_switch is True


def test_process_approved_goes_to_order_topic():
    bus, book, _, rm = make()
    orders, rejects = bus.subscribe(Topic.ORDER), bus.subscribe(Topic.REJECT)
    quote(book)
    order = Order("BTC", Side.BUY, 1)
    assert rm.process(order) is order
    assert orders.get_nowait() is order
    assert rejects.qsize() == 0
    assert rm.approved == 1 and rm.rejected == 0


def test_process_rejected_goes_to_reject_topic():
    bus, _, _, rm = make()
    orders, rejects = bus.subscribe(Topic.ORDER), bus.subscribe(Topic.REJECT)
    result = rm.process(Order("BTC", Side.BUY, 1))  # no market data
    assert isinstance(result, Rejection)
    assert result.source == RiskManager.SOURCE
    assert orders.qsize() == 0
    assert rejects.get_nowait() is result
    assert rm.approved == 0 and rm.rejected == 1


def test_runtime_limit_update_takes_effect():
    _, book, _, rm = make()
    quote(book)
    assert rm.check(Order("BTC", Side.BUY, 10)) is None
    rm.limits.update(max_order_qty=5.0)
    assert "qty" in rm.check(Order("BTC", Side.BUY, 10))


async def test_run_consumes_intents():
    bus, book, _, rm = make()
    orders = bus.subscribe(Topic.ORDER)
    quote(book)
    task = asyncio.create_task(rm.run())
    bus.publish(Topic.INTENT, Order("BTC", Side.BUY, 1))
    bus.publish(Topic.INTENT, Order("BTC", Side.BUY, 2_000))  # rejected
    await asyncio.sleep(0.01)
    rm.stop()
    await asyncio.wait_for(task, timeout=1)
    assert rm.approved == 1 and rm.rejected == 1
    assert orders.qsize() == 1


async def test_intents_before_run_are_not_lost():
    bus, book, _, rm = make()
    quote(book)
    bus.publish(Topic.INTENT, Order("BTC", Side.BUY, 1))
    task = asyncio.create_task(rm.run())
    await asyncio.sleep(0.01)
    rm.stop()
    await asyncio.wait_for(task, timeout=1)
    assert rm.approved == 1
