import pytest

from zetaline.core.models import Fill, Order, OrderType, Position, Side, Tick


def fill(side: Side, qty: float, price: float, fee: float = 0.0) -> Fill:
    return Fill(order_id=1, symbol="BTC", side=side, qty=qty, price=price, fee=fee)


def test_tick_mid_and_spread():
    t = Tick("BTC", bid=100.0, ask=102.0, bid_size=1, ask_size=1)
    assert t.mid == 101.0
    assert t.spread == 2.0


def test_order_rejects_non_positive_qty():
    with pytest.raises(ValueError):
        Order("BTC", Side.BUY, qty=0)


def test_limit_order_requires_price():
    with pytest.raises(ValueError):
        Order("BTC", Side.BUY, qty=1, order_type=OrderType.LIMIT)


def test_order_ids_increase():
    a = Order("BTC", Side.BUY, 1)
    b = Order("BTC", Side.BUY, 1)
    assert b.order_id > a.order_id


def test_open_long_and_average_up():
    p = Position("BTC")
    p.apply_fill(fill(Side.BUY, 1, 100))
    p.apply_fill(fill(Side.BUY, 1, 110))
    assert p.qty == 2
    assert p.avg_price == pytest.approx(105.0)
    assert p.realized_pnl == 0.0


def test_partial_close_realizes_pnl():
    p = Position("BTC")
    p.apply_fill(fill(Side.BUY, 2, 100))
    p.apply_fill(fill(Side.SELL, 1, 110))
    assert p.qty == 1
    assert p.avg_price == pytest.approx(100.0)
    assert p.realized_pnl == pytest.approx(10.0)


def test_full_close_resets_avg_price():
    p = Position("BTC")
    p.apply_fill(fill(Side.BUY, 1, 100))
    p.apply_fill(fill(Side.SELL, 1, 90))
    assert p.qty == 0
    assert p.avg_price == 0.0
    assert p.realized_pnl == pytest.approx(-10.0)


def test_short_position_profit():
    p = Position("BTC")
    p.apply_fill(fill(Side.SELL, 1, 100))
    p.apply_fill(fill(Side.BUY, 1, 90))
    assert p.qty == 0
    assert p.realized_pnl == pytest.approx(10.0)


def test_flip_long_to_short():
    p = Position("BTC")
    p.apply_fill(fill(Side.BUY, 1, 100))
    p.apply_fill(fill(Side.SELL, 3, 110))
    assert p.qty == -2
    assert p.avg_price == pytest.approx(110.0)
    assert p.realized_pnl == pytest.approx(10.0)


def test_fees_reduce_realized_pnl():
    p = Position("BTC")
    p.apply_fill(fill(Side.BUY, 1, 100, fee=0.5))
    assert p.realized_pnl == pytest.approx(-0.5)


def test_unrealized_pnl():
    p = Position("BTC")
    p.apply_fill(fill(Side.BUY, 2, 100))
    assert p.unrealized_pnl(105) == pytest.approx(10.0)
    short = Position("ETH")
    short.apply_fill(fill(Side.SELL, 2, 100))
    assert short.unrealized_pnl(95) == pytest.approx(10.0)
    assert Position("X").unrealized_pnl(100) == 0.0
