"""Strategy interface.

Strategies are deterministic and synchronous: given a tick and the current
position, they return an Order (an unchecked intent) or None. They never
touch the bus, the portfolio or the executor; the StrategyRunner does that.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

from zetaline.core.models import Order, OrderType, Side, Tick


class Strategy(ABC):
    def __init__(self, name: str, symbols: Iterable[str] | None = None) -> None:
        if not name:
            raise ValueError("strategy name must be non-empty")
        self.name = name
        # None means "trade every symbol the feed provides".
        self.symbols: frozenset[str] | None = (
            frozenset(symbols) if symbols is not None else None
        )
        self.enabled = True  # pause switch, toggled by the CLI / agent

    def wants(self, symbol: str) -> bool:
        return self.symbols is None or symbol in self.symbols

    @abstractmethod
    def on_tick(self, tick: Tick, position_qty: float) -> Order | None:
        """Return an order intent for this tick, or None to do nothing.

        position_qty is the signed current position in tick.symbol.
        """

    def params(self) -> dict[str, Any]:
        """Current tunable parameters (read by the CLI and the agent)."""
        return {}

    def set_params(self, **changes: Any) -> dict[str, tuple[Any, Any]]:
        """Validate and apply changes atomically; return {name: (old, new)}.

        The base class has no tunable parameters, so it refuses everything.
        """
        raise ValueError(f"strategy {self.name!r} has no tunable parameters")

    def _order(
        self,
        tick: Tick,
        side: Side,
        qty: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
    ) -> Order:
        """Build an Order tagged with this strategy and the originating tick time."""
        return Order(
            symbol=tick.symbol,
            side=side,
            qty=qty,
            order_type=order_type,
            limit_price=limit_price,
            strategy=self.name,
            origin_ns=tick.recv_ns,
        )
