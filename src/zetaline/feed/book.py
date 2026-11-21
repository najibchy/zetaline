"""In-memory top-of-book store: the latest tick per symbol.

Single source of truth for "current price". Strategy, risk manager and the
paper executor all read from here. Safe without locks because asyncio runs
on a single thread.
"""
from __future__ import annotations

from zetaline.core.models import Tick


class OrderBook:
    """Latest top-of-book snapshot for every symbol."""

    __slots__ = ("_ticks", "stale_dropped", "updates")

    def __init__(self) -> None:
        self._ticks: dict[str, Tick] = {}
        self.updates = 0
        self.stale_dropped = 0

    def update(self, tick: Tick) -> bool:
        """Store the tick if it is not older than the current one.

        Returns True if the book changed, False if the tick was stale.
        """
        current = self._ticks.get(tick.symbol)
        if current is not None and tick.ts_ns < current.ts_ns:
            self.stale_dropped += 1
            return False
        self._ticks[tick.symbol] = tick
        self.updates += 1
        return True

    def get(self, symbol: str) -> Tick | None:
        return self._ticks.get(symbol)

    def mid(self, symbol: str) -> float | None:
        tick = self._ticks.get(symbol)
        return tick.mid if tick is not None else None

    def symbols(self) -> list[str]:
        return sorted(self._ticks)

    def snapshot(self) -> dict[str, Tick]:
        """A shallow copy, safe to hand to the CLI or the agent."""
        return dict(self._ticks)

    def __len__(self) -> int:
        return len(self._ticks)

    def __contains__(self, symbol: object) -> bool:
        return symbol in self._ticks
