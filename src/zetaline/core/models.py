"""Core domain models shared across the whole platform.

Hot-path objects are frozen slotted dataclasses: cheap to create, immutable,
safe to share between asyncio tasks.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from itertools import count

_order_ids = count(1)


def next_order_id() -> int:
    """Process-wide monotonically increasing order id."""
    return next(_order_ids)


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

    @property
    def sign(self) -> int:
        return 1 if self is Side.BUY else -1


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(str, Enum):
    NEW = "NEW"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True, slots=True)
class Tick:
    """A top-of-book market data update."""

    symbol: str
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    # Wall-clock time the exchange/feed produced the tick.
    ts_ns: int = field(default_factory=time.time_ns)
    # Monotonic time the tick entered our process (for latency measurement).
    recv_ns: int = field(default_factory=time.perf_counter_ns)

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass(frozen=True, slots=True)
class Order:
    """An order request produced by a strategy."""

    symbol: str
    side: Side
    qty: float
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    order_id: int = field(default_factory=next_order_id)
    strategy: str = "unknown"
    ts_ns: int = field(default_factory=time.time_ns)
    # Monotonic time of the tick that triggered this order, for tick-to-trade latency.
    origin_ns: int = field(default_factory=time.perf_counter_ns)

    def __post_init__(self) -> None:
        if self.qty <= 0:
            raise ValueError(f"qty must be positive, got {self.qty}")
        if self.order_type is OrderType.LIMIT and self.limit_price is None:
            raise ValueError("LIMIT order requires limit_price")


@dataclass(frozen=True, slots=True)
class Fill:
    """The result of an order being executed (fully, for now)."""

    order_id: int
    symbol: str
    side: Side
    qty: float
    price: float
    fee: float = 0.0
    ts_ns: int = field(default_factory=time.time_ns)
    done_ns: int = field(default_factory=time.perf_counter_ns)

    @property
    def notional(self) -> float:
        return self.qty * self.price


@dataclass(slots=True)
class Position:
    """Mutable per-symbol position with average-cost accounting."""

    symbol: str
    qty: float = 0.0  # signed: >0 long, <0 short
    avg_price: float = 0.0
    realized_pnl: float = 0.0

    def apply_fill(self, fill: Fill) -> None:
        signed = fill.side.sign * fill.qty
        self.realized_pnl -= fill.fee

        # Opening or adding in the same direction: update average cost.
        if self.qty == 0 or (self.qty > 0) == (signed > 0):
            new_qty = self.qty + signed
            total_cost = self.avg_price * abs(self.qty) + fill.price * abs(signed)
            self.avg_price = total_cost / abs(new_qty)
            self.qty = new_qty
            return

        # Reducing / closing / flipping.
        closing = min(abs(signed), abs(self.qty))
        direction = 1 if self.qty > 0 else -1
        self.realized_pnl += (fill.price - self.avg_price) * closing * direction
        new_qty = self.qty + signed
        if new_qty == 0:
            self.avg_price = 0.0
        elif (new_qty > 0) != (self.qty > 0):
            self.avg_price = fill.price  # flipped: remainder opens at fill price
        self.qty = new_qty

    def unrealized_pnl(self, mark_price: float) -> float:
        if self.qty == 0:
            return 0.0
        return (mark_price - self.avg_price) * self.qty


@dataclass(frozen=True, slots=True)
class Rejection:
    """An order that was refused (by the risk manager or the executor)."""

    order: Order
    reason: str
    source: str = "unknown"
    ts_ns: int = field(default_factory=time.time_ns)
