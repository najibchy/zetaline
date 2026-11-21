import pytest

from zetaline.core.events import EventBus, Topic
from zetaline.core.latency import (
    STAGE_EXECUTE,
    STAGE_RISK_CHECK,
    STAGE_STRATEGY,
    STAGE_TICK_TO_FILL,
    STAGE_TICK_TO_INTENT,
    STAGE_TICK_TO_ORDER,
    LatencyRecorder,
    StageStats,
    percentile,
)
from zetaline.core.models import Order, Side, Tick
from zetaline.execution.paper import PaperExecutor
from zetaline.execution.portfolio import Portfolio
from zetaline.feed.book import OrderBook
from zetaline.risk.manager import RiskManager
from zetaline.strategy.base import Strategy
from zetaline.strategy.runner import StrategyRunner


# ---- percentile -----------------------------------------------------------
def test_percentile_empty_is_zero():
    assert percentile([], 50) == 0


def test_percentile_nearest_rank():
    vals = list(range(1, 101))
    assert percentile(vals, 50) == 50
    assert percentile(vals, 95) == 95
    assert percentile(vals, 99) == 99
    assert percentile(vals, 100) == 100


def test_percentile_single_value():
    assert percentile([7], 1) == 7
    assert percentile([7], 99) == 7


# ---- StageStats -------------------------------------------------------------
def test_invalid_capacity():
    with pytest.raises(ValueError):
        StageStats(0)


def test_summary_basic():
    s = StageStats()
    for i in range(1, 101):
        s.record(i * 1_000)
    out = s.summary()
    assert out["count"] == 100
    assert out["mean_us"] == pytest.approx(50.5)
    assert out["min_us"] == 1.0 and out["max_us"] == 100.0
    assert out["p50_us"] == 50.0 and out["p99_us"] == 99.0


def test_empty_summary_is_zeros():
    out = StageStats().summary()
    assert out["count"] == 0
    assert out["mean_us"] == 0.0 and out["p99_us"] == 0.0


def test_ring_wraps_but_alltime_counters_persist():
    s = StageStats(capacity=10)
    for i in range(1, 26):
        s.record(i * 1_000)
    out = s.summary()
    assert out["count"] == 25
    assert out["min_us"] == 1.0 and out["max_us"] == 25.0  # all-time
    assert sorted(s.window()) == [i * 1_000 for i in range(16, 26)]  # rolling
    assert out["p50_us"] == 20.0


def test_negative_sample_clamped():
    s = StageStats()
    s.record(-5)
    assert s.summary()["max_us"] == 0.0
    assert s.count == 1


# ---- LatencyRecorder ----------------------------------------------------------
def test_recorder_creates_stages_lazily_and_resets():
    rec = LatencyRecorder()
    assert rec.stages() == []
    rec.record("a", 1_000)
    rec.record("b", 2_000)
    assert rec.stages() == ["a", "b"]
    assert rec.summary()["a"]["count"] == 1
    rec.reset()
    assert rec.stages() == [] and rec.summary() == {}


# ---- wiring -----------------------------------------------------------------------
class AlwaysBuy(Strategy):
    def on_tick(self, tick, position_qty):
        return self._order(tick, Side.BUY, 1)


def tick(price: float = 100.0) -> Tick:
    return Tick("BTC", bid=price - 0.01, ask=price + 0.01, bid_size=1, ask_size=1)


def test_runner_records_strategy_and_tick_to_intent():
    bus, book, pf, lat = EventBus(), OrderBook(), Portfolio(100_000.0), LatencyRecorder()
    runner = StrategyRunner(bus, book, pf, [AlwaysBuy("a")], latency=lat)
    runner.on_tick(tick())
    runner.on_tick(tick())
    assert lat.summary()[STAGE_STRATEGY]["count"] == 2
    assert lat.summary()[STAGE_TICK_TO_INTENT]["count"] == 2


def test_runner_without_intent_records_only_strategy():
    class Quiet(Strategy):
        def on_tick(self, tick, position_qty):
            return None

    bus, book, pf, lat = EventBus(), OrderBook(), Portfolio(100_000.0), LatencyRecorder()
    StrategyRunner(bus, book, pf, [Quiet("q")], latency=lat).on_tick(tick())
    assert lat.stages() == [STAGE_STRATEGY]


def test_risk_manager_records_check_and_tick_to_order():
    bus, book, pf, lat = EventBus(), OrderBook(), Portfolio(100_000.0), LatencyRecorder()
    rm = RiskManager(bus, book, pf, latency=lat)
    book.update(tick())
    rm.process(Order("BTC", Side.BUY, 1))
    summary = lat.summary()
    assert summary[STAGE_RISK_CHECK]["count"] == 1
    assert summary[STAGE_TICK_TO_ORDER]["count"] == 1


def test_risk_rejection_records_check_only():
    bus, book, pf, lat = EventBus(), OrderBook(), Portfolio(100_000.0), LatencyRecorder()
    rm = RiskManager(bus, book, pf, latency=lat)
    rm.process(Order("BTC", Side.BUY, 1))  # no market data -> rejected
    assert lat.stages() == [STAGE_RISK_CHECK]


def test_executor_records_execute_and_tick_to_fill():
    bus, book, pf, lat = EventBus(), OrderBook(), Portfolio(100_000.0), LatencyRecorder()
    ex = PaperExecutor(bus, book, pf, latency=lat)
    book.update(tick())
    ex.execute(Order("BTC", Side.BUY, 1))
    summary = lat.summary()
    assert summary[STAGE_EXECUTE]["count"] == 1
    assert summary[STAGE_TICK_TO_FILL]["count"] == 1


def test_components_work_without_recorder():
    bus, book, pf = EventBus(), OrderBook(), Portfolio(100_000.0)
    runner = StrategyRunner(bus, book, pf, [AlwaysBuy("a")])
    rm = RiskManager(bus, book, pf)
    ex = PaperExecutor(bus, book, pf)
    orders = bus.subscribe(Topic.INTENT)
    runner.on_tick(tick())
    assert rm.process(orders.get_nowait()) is not None
    assert ex.execute(Order("BTC", Side.BUY, 1)) is not None
