from zetaline.core.models import Tick
from zetaline.feed.book import OrderBook


def tick(symbol: str = "BTCUSDT", bid: float = 100.0, ask: float = 101.0, ts_ns: int = 1_000) -> Tick:
    return Tick(symbol, bid=bid, ask=ask, bid_size=1.0, ask_size=1.0, ts_ns=ts_ns)


def test_empty_book():
    book = OrderBook()
    assert len(book) == 0
    assert book.get("BTCUSDT") is None
    assert book.mid("BTCUSDT") is None
    assert book.symbols() == []
    assert "BTCUSDT" not in book


def test_update_stores_latest_tick():
    book = OrderBook()
    assert book.update(tick(bid=100, ask=102, ts_ns=1)) is True
    assert book.get("BTCUSDT").mid == 101.0
    assert book.mid("BTCUSDT") == 101.0
    assert "BTCUSDT" in book
    assert len(book) == 1
    assert book.updates == 1


def test_newer_tick_replaces_older():
    book = OrderBook()
    book.update(tick(bid=100, ask=102, ts_ns=1))
    assert book.update(tick(bid=110, ask=112, ts_ns=2)) is True
    assert book.mid("BTCUSDT") == 111.0
    assert book.updates == 2


def test_stale_tick_is_rejected_and_counted():
    book = OrderBook()
    book.update(tick(bid=110, ask=112, ts_ns=5))
    assert book.update(tick(bid=90, ask=92, ts_ns=4)) is False
    assert book.mid("BTCUSDT") == 111.0
    assert book.stale_dropped == 1
    assert book.updates == 1


def test_equal_timestamp_is_accepted():
    book = OrderBook()
    book.update(tick(bid=100, ask=102, ts_ns=5))
    assert book.update(tick(bid=104, ask=106, ts_ns=5)) is True
    assert book.mid("BTCUSDT") == 105.0
    assert book.stale_dropped == 0


def test_symbols_are_independent_and_sorted():
    book = OrderBook()
    book.update(tick("SOLUSDT", ts_ns=1))
    book.update(tick("BTCUSDT", ts_ns=1))
    # A newer tick for one symbol must not affect another's staleness check.
    book.update(tick("ETHUSDT", ts_ns=0))
    assert book.symbols() == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    assert book.stale_dropped == 0


def test_snapshot_is_a_copy():
    book = OrderBook()
    book.update(tick("BTCUSDT", ts_ns=1))
    snap = book.snapshot()
    snap.clear()
    assert len(book) == 1
    assert book.get("BTCUSDT") is not None
