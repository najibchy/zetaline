import json

import pytest

from zetaline.feed.binance import (
    MAINNET_HOST,
    TESTNET_HOST,
    ParseError,
    Shutdown,
    build_url,
    parse_message,
)

# A real message captured from the live feed.
RAW = (
    '{"stream":"btcusdt@bookTicker","data":{"u":103397853760,"s":"BTCUSDT",'
    '"b":"60086.94000000","B":"0.68754000","a":"60086.95000000","A":"1.65794000"}}'
)
INNER = json.dumps(json.loads(RAW)["data"])


def test_parses_captured_combined_message():
    t = parse_message(RAW)
    assert t.symbol == "BTCUSDT"
    assert t.bid == 60086.94 and t.ask == 60086.95
    assert t.bid_size == 0.68754 and t.ask_size == 1.65794
    assert t.spread == pytest.approx(0.01)


def test_raw_form_equals_combined_form():
    assert parse_message(RAW, recv_ns=1, ts_ns=2) == parse_message(INNER, recv_ns=1, ts_ns=2)


def test_accepts_bytes():
    assert parse_message(RAW.encode()).symbol == "BTCUSDT"


def test_injected_timestamps_are_used():
    t = parse_message(RAW, recv_ns=111, ts_ns=222)
    assert t.recv_ns == 111 and t.ts_ns == 222


@pytest.mark.parametrize("msg", ['{"result":null,"id":1}', '{"stream":"x","data":{"foo":1}}'])
def test_non_market_data_is_ignored(msg):
    assert parse_message(msg) is None


def test_server_shutdown_marker():
    assert isinstance(parse_message('{"e":"serverShutdown","E":1}'), Shutdown)


@pytest.mark.parametrize(
    ("msg", "match"),
    [
        ("nope", "invalid JSON"),
        ("[1]", "not a JSON object"),
        ('{"s":"X","b":"1","a":"2"}', "missing field"),
        ('{"s":"X","b":"0","B":"1","a":"2","A":"1"}', "non-positive"),
        ('{"s":"X","b":"nan","B":"1","a":"2","A":"1"}', "non-finite"),
        ('{"s":"X","b":"3","B":"1","a":"2","A":"1"}', "crossed"),
        ('{"s":"X","b":"abc","B":"1","a":"2","A":"1"}', "bad b"),
        ('{"s":"X","b":"1","B":"-1","a":"2","A":"1"}', "negative size"),
        ('{"s":"","b":"1","B":"1","a":"2","A":"1"}', "bad symbol"),
    ],
)
def test_malformed_bookticker_raises(msg, match):
    with pytest.raises(ParseError, match=match):
        parse_message(msg)


def test_non_object_data_raises():
    with pytest.raises(ParseError, match="data is not"):
        parse_message('{"data": 5}')


def test_parse_error_is_a_value_error():
    assert issubclass(ParseError, ValueError)


def test_build_url_mainnet():
    assert build_url(["btcusdt", "ethusdt"]) == (
        f"{MAINNET_HOST}/stream?streams=btcusdt@bookTicker/ethusdt@bookTicker"
    )


def test_build_url_testnet():
    assert build_url(["btcusdt"], testnet=True) == (
        f"{TESTNET_HOST}/stream?streams=btcusdt@bookTicker"
    )


def test_build_url_lowercases_symbols():
    assert "btcusdt@bookTicker" in build_url(["BTCUSDT"])


@pytest.mark.parametrize("symbols", [[], ["btc/usdt"], ["x"], ["BTC USDT"]])
def test_build_url_rejects_bad_symbols(symbols):
    with pytest.raises(ValueError):
        build_url(symbols)


@pytest.mark.parametrize("symbol", ["btcusdt\n", "btcusdt\r\n", " btcusdt", "btcusdt "])
def test_build_url_rejects_whitespace_around_symbol(symbol):
    with pytest.raises(ValueError):
        build_url([symbol])
