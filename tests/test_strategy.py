import pytest

from zetaline.core.models import Side, Tick
from zetaline.strategy.base import Strategy
from zetaline.strategy.momentum import MomentumStrategy


def tick(price: float, symbol: str = "BTC") -> Tick:
    return Tick(symbol, bid=price - 0.01, ask=price + 0.01, bid_size=1, ask_size=1)


class AlwaysBuy(Strategy):
    def on_tick(self, tick, position_qty):
        return self._order(tick, Side.BUY, 1)


def drive(strategy: MomentumStrategy, prices, symbol: str = "BTC"):
    """Feed prices, tracking the position like the runner does. Returns (orders, position)."""
    pos, orders = 0.0, []
    for p in prices:
        o = strategy.on_tick(tick(p, symbol), pos)
        if o:
            pos += o.side.sign * o.qty
            orders.append(o)
    return orders, pos


# ---- base class ---------------------------------------------------------
def test_base_is_abstract():
    with pytest.raises(TypeError):
        Strategy("x")


def test_empty_name_rejected():
    with pytest.raises(ValueError):
        AlwaysBuy("")


def test_wants_filters_symbols():
    s = AlwaysBuy("a", symbols=["BTC"])
    assert s.wants("BTC") and not s.wants("ETH")
    assert AlwaysBuy("b").wants("ANYTHING")


def test_order_helper_tags_strategy_and_origin():
    s = AlwaysBuy("a")
    t = tick(100)
    o = s.on_tick(t, 0.0)
    assert o.strategy == "a"
    assert o.origin_ns == t.recv_ns


def test_base_set_params_refuses():
    with pytest.raises(ValueError):
        AlwaysBuy("a").set_params(x=1)
    assert AlwaysBuy("a").params() == {}


def test_enabled_defaults_true():
    assert AlwaysBuy("a").enabled is True


# ---- momentum -----------------------------------------------------------
def make(**kw) -> MomentumStrategy:
    defaults = {"symbols": ["BTC"], "fast": 3, "slow": 6, "trade_qty": 2.0}
    return MomentumStrategy(**{**defaults, **kw})


def test_no_orders_during_warmup():
    s = make()
    orders, _ = drive(s, [100.0, 101.0, 102.0, 103.0, 104.0])  # 5 ticks < slow=6
    assert orders == []


def test_flat_prices_never_trade():
    orders, pos = drive(make(), [100.0] * 50)
    assert orders == [] and pos == 0.0


def test_uptrend_buys_then_downtrend_sells():
    prices = [100.0] * 8 + [100 + 2 * i for i in range(1, 11)] + [120 - 3 * i for i in range(1, 15)]
    orders, pos = drive(make(), prices)
    assert [(o.side, o.qty) for o in orders] == [(Side.BUY, 2.0), (Side.SELL, 2.0)]
    assert pos == 0.0


def test_does_not_reorder_when_already_at_target():
    s = make()
    prices = [100.0] * 8 + [100 + 2 * i for i in range(1, 11)]
    orders, pos = drive(s, prices)
    assert len(orders) == 1 and pos == 2.0


def test_retries_when_position_not_updated():
    """If a risk rejection means the position never changes, the strategy asks again."""
    s = make()
    prices = [100.0] * 8 + [100 + 2 * i for i in range(1, 11)]
    orders = [s.on_tick(tick(p), 0.0) for p in prices]
    assert sum(o is not None for o in orders) > 1


def test_orders_only_the_difference():
    s = make(trade_qty=5.0)
    prices = [100.0] * 8 + [100 + 2 * i for i in range(1, 6)]
    first = None
    for p in prices:
        first = s.on_tick(tick(p), 3.0) or first  # already holding 3
    assert first is not None and first.side is Side.BUY and first.qty == pytest.approx(2.0)


def test_large_deadband_suppresses_trading():
    s = make(threshold_bps=10_000.0)
    prices = [100.0] * 8 + [100 + 2 * i for i in range(1, 11)]
    orders, _ = drive(s, prices)
    assert orders == []


def test_symbols_have_independent_state():
    s = MomentumStrategy(symbols=None, fast=3, slow=6, trade_qty=1.0)
    up = [100.0] * 8 + [100 + 2 * i for i in range(1, 11)]
    flat = [50.0] * len(up)
    pos = {"A": 0.0, "B": 0.0}
    for pa, pb in zip(up, flat, strict=True):
        for sym, p in (("A", pa), ("B", pb)):
            o = s.on_tick(tick(p, sym), pos[sym])
            if o:
                pos[sym] += o.side.sign * o.qty
    assert pos == {"A": 1.0, "B": 0.0}


def test_order_carries_originating_tick_time():
    s = make()
    pos, last_tick, order = 0.0, None, None
    for p in [100.0] * 8 + [100 + 2 * i for i in range(1, 11)]:
        last_tick = tick(p)
        order = s.on_tick(last_tick, pos)
        if order:
            break
    assert order is not None and order.origin_ns == last_tick.recv_ns


# ---- validation -----------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs",
    [
        {"fast": 10, "slow": 5},
        {"fast": 5, "slow": 5},
        {"fast": 0, "slow": 5},
        {"slow": 1},
        {"trade_qty": 0},
        {"trade_qty": -1},
        {"trade_qty": float("nan")},
        {"threshold_bps": -1},
        {"threshold_bps": float("inf")},
    ],
)
def test_constructor_rejects_bad_values(kwargs):
    with pytest.raises(ValueError):
        make(**kwargs)


@pytest.mark.parametrize("kwargs", [{"fast": 2.5}, {"slow": "6"}, {"fast": True}, {"trade_qty": "1"}])
def test_constructor_rejects_bad_types(kwargs):
    with pytest.raises(TypeError):
        make(**kwargs)


# ---- set_params -----------------------------------------------------------
def test_params_roundtrip():
    assert make().params() == {"fast": 3, "slow": 6, "trade_qty": 2.0, "threshold_bps": 1.0}


def test_set_params_returns_old_and_new():
    s = make()
    assert s.set_params(threshold_bps=3.0) == {"threshold_bps": (1.0, 3.0)}
    assert s.threshold_bps == 3.0


def test_set_params_unchanged_value_not_reported():
    assert make().set_params(trade_qty=2.0) == {}


def test_set_params_unknown_rejected_and_atomic():
    s = make()
    before = s.params()
    with pytest.raises(ValueError, match="unknown"):
        s.set_params(bogus=1)
    assert s.params() == before


def test_set_params_invalid_is_atomic():
    s = make()
    before = s.params()
    with pytest.raises(ValueError):
        s.set_params(trade_qty=5.0, fast=10)  # fast >= slow: nothing may apply
    assert s.params() == before
    with pytest.raises(TypeError):
        s.set_params(trade_qty=5.0, slow=7.5)
    assert s.params() == before


def test_changing_window_resets_state_and_warmup():
    s = make()
    drive(s, [100.0] * 8)
    assert s._state  # warmed up
    s.set_params(fast=4)
    assert s._state == {}
    assert s._alpha_fast == pytest.approx(2.0 / 5)
    orders, _ = drive(s, [100.0 + i for i in range(5)])  # fewer than slow=6 ticks
    assert orders == []


def test_changing_non_window_param_keeps_state():
    s = make()
    drive(s, [100.0] * 8)
    s.set_params(trade_qty=3.0)
    assert s._state


# ---- notional sizing --------------------------------------------------------------
UP = [100.0] * 8 + [100 + 2 * i for i in range(1, 11)]
DOWN = [120 - 3 * i for i in range(1, 15)]


def test_notional_sizes_by_price_at_entry():
    orders, pos = drive(make(trade_notional=1_000.0), UP)
    assert len(orders) == 1  # no resizing during the rest of the uptrend
    assert orders[0].side is Side.BUY
    assert orders[0].qty == pytest.approx(1_000.0 / 102.0, abs=1e-6)
    assert pos == pytest.approx(orders[0].qty)


def test_notional_qty_scales_inversely_with_price():
    low, _ = drive(make(trade_notional=1_000.0), UP)
    high, _ = drive(make(trade_notional=1_000.0), [p * 10 for p in UP])
    assert high[0].qty * 10 == pytest.approx(low[0].qty, rel=1e-4)


def test_notional_holds_existing_position_without_resizing():
    s = make(trade_notional=1_000.0)
    assert all(s.on_tick(tick(p), 5.0) is None for p in UP)


def test_notional_exit_sells_whole_position():
    orders, pos = drive(make(trade_notional=1_000.0), UP + DOWN)
    assert [o.side for o in orders] == [Side.BUY, Side.SELL]
    assert orders[1].qty == pytest.approx(orders[0].qty)
    assert pos == pytest.approx(0.0, abs=1e-9)


def test_notional_retries_when_position_not_updated():
    s = make(trade_notional=1_000.0)
    orders = [s.on_tick(tick(p), 0.0) for p in UP]
    assert sum(o is not None for o in orders) > 1


def test_notional_params_report_mode():
    params = make(trade_notional=1_000.0).params()
    assert set(params) == {"fast", "slow", "trade_notional", "threshold_bps"}
    assert "trade_qty" not in make(trade_notional=1_000.0).params()
    assert "trade_notional" not in make().params()


def test_notional_takes_precedence_over_qty():
    s = MomentumStrategy(symbols=["BTC"], fast=3, slow=6, trade_qty=3.0, trade_notional=500.0)
    assert s.params()["trade_notional"] == 500.0


def test_notional_mode_rejects_trade_qty_param():
    s = make(trade_notional=1_000.0)
    before = s.params()
    with pytest.raises(ValueError, match="unknown"):
        s.set_params(trade_qty=1.0)
    assert s.params() == before


def test_qty_mode_rejects_trade_notional_param():
    s = make()
    before = s.params()
    with pytest.raises(ValueError, match="unknown"):
        s.set_params(trade_notional=100.0)
    assert s.params() == before


def test_set_notional_returns_old_and_new():
    s = make(trade_notional=1_000.0)
    assert s.set_params(trade_notional=2_000.0) == {"trade_notional": (1_000.0, 2_000.0)}
    assert s.trade_notional == 2_000.0


def test_set_notional_invalid_is_atomic():
    s = make(trade_notional=1_000.0)
    before = s.params()
    with pytest.raises(ValueError):
        s.set_params(trade_notional=5_000.0, fast=10)  # fast >= slow: nothing may apply
    assert s.params() == before
    with pytest.raises(TypeError):
        s.set_params(trade_notional="lots")
    assert s.params() == before


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
def test_notional_constructor_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        make(trade_notional=bad)


@pytest.mark.parametrize("bad", ["x", True, "100"])
def test_notional_constructor_rejects_bad_types(bad):
    with pytest.raises(TypeError):
        make(trade_notional=bad)
