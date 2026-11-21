"""Portfolio: cash, positions, and P&L, marked to market from the order book."""
from __future__ import annotations

from zetaline.core.models import Fill, Position
from zetaline.feed.book import OrderBook


class Portfolio:
    """Tracks cash and per-symbol positions.

    Invariant: equity - starting_cash == realized_pnl + unrealized_pnl
    (realized P&L is net of fees).
    """

    def __init__(self, starting_cash: float = 100_000.0) -> None:
        if starting_cash <= 0:
            raise ValueError("starting_cash must be positive")
        self.starting_cash = starting_cash
        self.cash = starting_cash
        self.fees_paid = 0.0
        self.fills = 0
        self._positions: dict[str, Position] = {}

    def apply_fill(self, fill: Fill) -> None:
        pos = self._positions.get(fill.symbol)
        if pos is None:
            pos = self._positions[fill.symbol] = Position(fill.symbol)
        pos.apply_fill(fill)
        self.cash -= fill.side.sign * fill.notional + fill.fee
        self.fees_paid += fill.fee
        self.fills += 1

    def position(self, symbol: str) -> Position | None:
        return self._positions.get(symbol)

    def positions(self) -> dict[str, Position]:
        """Open (non-zero) positions only."""
        return {s: p for s, p in self._positions.items() if p.qty != 0}

    @staticmethod
    def _mark(pos: Position, book: OrderBook) -> float:
        mid = book.mid(pos.symbol)
        return mid if mid is not None else pos.avg_price

    @property
    def realized_pnl(self) -> float:
        return sum(p.realized_pnl for p in self._positions.values())

    def unrealized_pnl(self, book: OrderBook) -> float:
        return sum(p.unrealized_pnl(self._mark(p, book)) for p in self._positions.values())

    def equity(self, book: OrderBook) -> float:
        return self.cash + sum(p.qty * self._mark(p, book) for p in self._positions.values())

    def total_pnl(self, book: OrderBook) -> float:
        return self.equity(book) - self.starting_cash

    def exposure(self, book: OrderBook) -> float:
        """Gross exposure: sum of absolute position values."""
        return sum(abs(p.qty) * self._mark(p, book) for p in self._positions.values())

    def snapshot(self, book: OrderBook) -> dict:
        """Plain-dict view for the CLI and the AI agent."""
        return {
            "cash": self.cash,
            "equity": self.equity(book),
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl(book),
            "total_pnl": self.total_pnl(book),
            "exposure": self.exposure(book),
            "fees_paid": self.fees_paid,
            "fills": self.fills,
            "positions": {
                s: {
                    "qty": p.qty,
                    "avg_price": p.avg_price,
                    "mark": self._mark(p, book),
                    "unrealized_pnl": p.unrealized_pnl(self._mark(p, book)),
                    "realized_pnl": p.realized_pnl,
                }
                for s, p in self.positions().items()
            },
        }
