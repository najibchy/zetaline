import pytest

from zetaline.core.models import Fill, Side, Tick
from zetaline.execution.portfolio import Portfolio
from zetaline.feed.book import OrderBook


def fill(side: Side, qty: float, price: float, fee: float = 0.0, symbol: str = "BTC") -> Fill:
    return Fill(order_id=1, symbol=symbol, side=side, qty=qty, price=price, fee=fee)


def quote(book: OrderBook, symbol: str, bid: float, ask: float, ts_ns: int = 1) -> None:
    book.update(Tick(symbol, bid=bid, ask=ask, bid_size=1, ask_size=1, ts_ns=ts_ns))


def assert_identity(pf: Portfolio, book: OrderBook) -> None:
    lhs = pf.total_pnl(book)
    rhs = pf.realized_pnl + pf.unrealized_pnl(book)
    assert lhs == pytest.approx(rhs, abs=1e-9)


def test_invalid_starting_cash():
    with pytest.raises(ValueError):
        Portfolio(0)
    with pytest.raises(ValueError):
        Portfolio(-5)


def test_initial_state():
    pf = Portfolio(1_000)
    book = OrderBook()
    assert pf.cash == 1_000
    assert pf.equity(book) == 1_000
    assert pf.positions() == {}
    assert pf.total_pnl(book) == 0.0
    assert pf.exposure(book) == 0.0


def test_buy_moves_cash_and_opens_position():
    pf = Portfolio(1_000)
    pf.apply_fill(fill(Side.BUY, 2, 100, fee=1.0))
    assert pf.cash == pytest.approx(1_000 - 200 - 1.0)
    assert pf.position("BTC").qty == 2
    assert pf.fees_paid == 1.0
    assert pf.fills == 1


def test_sell_short_adds_cash():
    pf = Portfolio(1_000)
    pf.apply_fill(fill(Side.SELL, 2, 100, fee=1.0))
    assert pf.cash == pytest.approx(1_000 + 200 - 1.0)
    assert pf.position("BTC").qty == -2


def test_unrealized_pnl_marks_to_mid():
    pf, book = Portfolio(10_000), OrderBook()
    pf.apply_fill(fill(Side.BUY, 1, 100))
    quote(book, "BTC", bid=109, ask=111)  # mid 110
    assert pf.unrealized_pnl(book) == pytest.approx(10.0)
    assert pf.equity(book) == pytest.approx(10_010.0)
    assert_identity(pf, book)


def test_missing_tick_marks_at_cost():
    pf, book = Portfolio(10_000), OrderBook()
    pf.apply_fill(fill(Side.BUY, 1, 100))
    assert pf.unrealized_pnl(book) == 0.0
    assert pf.equity(book) == pytest.approx(10_000.0)


def test_round_trip_realizes_pnl_net_of_fees():
    pf, book = Portfolio(10_000), OrderBook()
    pf.apply_fill(fill(Side.BUY, 1, 100, fee=0.5))
    pf.apply_fill(fill(Side.SELL, 1, 110, fee=0.55))
    assert pf.positions() == {}
    assert pf.realized_pnl == pytest.approx(10.0 - 0.5 - 0.55)
    assert pf.total_pnl(book) == pytest.approx(pf.realized_pnl)
    assert_identity(pf, book)


def test_short_profit_when_price_falls():
    pf, book = Portfolio(10_000), OrderBook()
    pf.apply_fill(fill(Side.SELL, 2, 100))
    quote(book, "BTC", bid=94, ask=96)  # mid 95
    assert pf.unrealized_pnl(book) == pytest.approx(10.0)
    assert_identity(pf, book)


def test_identity_with_multiple_symbols_and_fees():
    pf, book = Portfolio(100_000), OrderBook()
    pf.apply_fill(fill(Side.BUY, 1, 100, fee=0.1, symbol="BTC"))
    pf.apply_fill(fill(Side.SELL, 3, 50, fee=0.2, symbol="ETH"))
    pf.apply_fill(fill(Side.BUY, 1, 105, fee=0.1, symbol="BTC"))
    pf.apply_fill(fill(Side.BUY, 1, 48, fee=0.1, symbol="ETH"))
    quote(book, "BTC", bid=111, ask=113)
    quote(book, "ETH", bid=44, ask=46)
    assert_identity(pf, book)


def test_exposure_is_gross():
    pf, book = Portfolio(100_000), OrderBook()
    pf.apply_fill(fill(Side.BUY, 1, 100, symbol="BTC"))
    pf.apply_fill(fill(Side.SELL, 2, 50, symbol="ETH"))
    quote(book, "BTC", bid=99, ask=101)  # mid 100
    quote(book, "ETH", bid=49, ask=51)   # mid 50
    assert pf.exposure(book) == pytest.approx(100 + 100)


def test_positions_excludes_closed():
    pf = Portfolio(10_000)
    pf.apply_fill(fill(Side.BUY, 1, 100))
    pf.apply_fill(fill(Side.SELL, 1, 100))
    assert pf.positions() == {}
    assert pf.position("BTC") is not None  # still tracked for realized P&L


def test_snapshot_contents():
    pf, book = Portfolio(10_000), OrderBook()
    pf.apply_fill(fill(Side.BUY, 1, 100, fee=1.0))
    quote(book, "BTC", bid=109, ask=111)
    snap = pf.snapshot(book)
    assert snap["fills"] == 1
    assert snap["fees_paid"] == 1.0
    assert snap["realized_pnl"] == pytest.approx(-1.0)
    assert snap["unrealized_pnl"] == pytest.approx(10.0)
    assert snap["total_pnl"] == pytest.approx(9.0)
    btc = snap["positions"]["BTC"]
    assert btc["qty"] == 1 and btc["mark"] == 110.0
