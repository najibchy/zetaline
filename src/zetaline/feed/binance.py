"""Binance bookTicker message parsing and URL building (pure, no I/O).

Public market data only; no API key is involved anywhere.

Wire format (verified against a live capture):
  combined: {"stream": "btcusdt@bookTicker",
             "data": {"u": id, "s": "BTCUSDT", "b": "bid", "B": "bid_qty",
                      "a": "ask", "A": "ask_qty"}}
  raw:      the inner "data" object on its own.
Prices and sizes are strings. There is no event timestamp.
"""
from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from zetaline.core.models import Tick

MAINNET_HOST = "wss://data-stream.binance.vision"  # market data only, real prices
TESTNET_HOST = "wss://stream.testnet.binance.vision"  # artificial testnet book

_SYMBOL_RE = re.compile(r"^[a-z0-9]{3,20}$")
_REQUIRED = ("s", "b", "B", "a", "A")


class ParseError(ValueError):
    """A bookTicker message was present but malformed or invalid."""


@dataclass(frozen=True, slots=True)
class Shutdown:
    """The server announced it is about to close the connection."""


def build_url(symbols: Iterable[str], testnet: bool = False) -> str:
    """Combined-stream URL for the bookTicker of every symbol."""
    names = [s.lower() for s in symbols]
    if not names:
        raise ValueError("at least one symbol is required")
    for name in names:
        if not _SYMBOL_RE.match(name):
            raise ValueError(f"invalid symbol: {name!r}")
    host = TESTNET_HOST if testnet else MAINNET_HOST
    streams = "/".join(f"{name}@bookTicker" for name in names)
    return f"{host}/stream?streams={streams}"


def _number(data: dict[str, Any], key: str) -> float:
    try:
        value = float(data[key])
    except (TypeError, ValueError) as exc:
        raise ParseError(f"bad {key}: {data[key]!r}") from exc
    if not math.isfinite(value):
        raise ParseError(f"non-finite {key}: {data[key]!r}")
    return value


def parse_message(
    raw: str | bytes,
    *,
    recv_ns: int | None = None,
    ts_ns: int | None = None,
) -> Tick | Shutdown | None:
    """Parse one WebSocket frame.

    Returns a Tick for a valid bookTicker, a Shutdown marker if the server is
    closing, or None for anything that is not market data (e.g. a subscription
    acknowledgement). Raises ParseError for a malformed bookTicker.
    """
    try:
        msg = json.loads(raw)
    except ValueError as exc:  # JSONDecodeError and UnicodeDecodeError
        raise ParseError("invalid JSON") from exc
    if not isinstance(msg, dict):
        raise ParseError("message is not a JSON object")

    data = msg.get("data", msg)  # combined wrapper, or the raw stream form
    if not isinstance(data, dict):
        raise ParseError("data is not a JSON object")
    if data.get("e") == "serverShutdown":  # assumed payload, see module notes
        return Shutdown()
    if "b" not in data and "a" not in data:
        return None  # not a bookTicker (subscription ack, etc.)

    missing = [k for k in _REQUIRED if k not in data]
    if missing:
        raise ParseError(f"missing field(s): {missing}")
    symbol = data["s"]
    if not isinstance(symbol, str) or not symbol:
        raise ParseError(f"bad symbol: {symbol!r}")

    bid, ask = _number(data, "b"), _number(data, "a")
    bid_size, ask_size = _number(data, "B"), _number(data, "A")
    if bid <= 0 or ask <= 0:
        raise ParseError(f"non-positive price: bid={bid} ask={ask}")
    if bid_size < 0 or ask_size < 0:
        raise ParseError(f"negative size: bid_size={bid_size} ask_size={ask_size}")
    if bid > ask:
        raise ParseError(f"crossed book: bid={bid} > ask={ask}")

    extra: dict[str, int] = {}
    if recv_ns is not None:
        extra["recv_ns"] = recv_ns
    if ts_ns is not None:
        extra["ts_ns"] = ts_ns
    return Tick(symbol, bid=bid, ask=ask, bid_size=bid_size, ask_size=ask_size, **extra)
