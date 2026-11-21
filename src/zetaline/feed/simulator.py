"""Seeded random-walk market data simulator.

Publishes Tick events to the event bus. Deterministic given a seed, which
makes tests and benchmarks reproducible.
"""
from __future__ import annotations

import asyncio
import math
import random
import time

from zetaline.core.events import EventBus, Topic
from zetaline.core.models import Tick

DEFAULT_PRICES: dict[str, float] = {
    "BTCUSDT": 65_000.0,
    "ETHUSDT": 3_200.0,
    "SOLUSDT": 150.0,
}


class MarketSimulator:
    """Generates and publishes simulated top-of-book ticks."""

    def __init__(
        self,
        bus: EventBus,
        prices: dict[str, float] | None = None,
        seed: int = 42,
        rate_hz: float = 100.0,
        volatility: float = 0.0005,
        spread_bps: float = 2.0,
    ) -> None:
        if rate_hz < 0:
            raise ValueError("rate_hz must be >= 0 (0 means unthrottled)")
        if volatility < 0 or spread_bps < 0:
            raise ValueError("volatility and spread_bps must be >= 0")
        self._bus = bus
        self._prices = dict(DEFAULT_PRICES if prices is None else prices)
        if not self._prices:
            raise ValueError("at least one symbol is required")
        self._rng = random.Random(seed)
        self.rate_hz = rate_hz
        self.volatility = volatility
        self.spread_bps = spread_bps
        self.ticks_published = 0
        self._running = False

    @property
    def symbols(self) -> list[str]:
        return list(self._prices)

    def next_tick(self, symbol: str) -> Tick:
        """Advance one symbol's price by one step and build a Tick (no publish)."""
        mid = self._prices[symbol] * math.exp(self.volatility * self._rng.gauss(0.0, 1.0))
        self._prices[symbol] = mid
        half_spread = mid * self.spread_bps / 20_000.0  # bps of price, split in two
        return Tick(
            symbol=symbol,
            bid=mid - half_spread,
            ask=mid + half_spread,
            bid_size=round(self._rng.uniform(0.1, 5.0), 3),
            ask_size=round(self._rng.uniform(0.1, 5.0), 3),
        )

    def step(self) -> int:
        """Generate and publish one tick for every symbol. Returns the count."""
        for symbol in self._prices:
            self._bus.publish(Topic.TICK, self.next_tick(symbol))
        self.ticks_published += len(self._prices)
        return len(self._prices)

    async def run(self, max_ticks: int | None = None) -> None:
        """Publish ticks until cancelled or `max_ticks` rounds are done.

        One "round" is one tick per symbol. rate_hz is rounds per second;
        rate_hz == 0 runs unthrottled (still yielding to other tasks).
        """
        self._running = True
        interval = 1.0 / self.rate_hz if self.rate_hz > 0 else 0.0
        next_at = time.perf_counter()
        rounds = 0
        try:
            while max_ticks is None or rounds < max_ticks:
                self.step()
                rounds += 1
                if interval == 0.0:
                    await asyncio.sleep(0)
                    continue
                next_at += interval
                delay = next_at - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
                else:
                    # We're behind schedule: yield once, don't accumulate debt.
                    next_at = time.perf_counter()
                    await asyncio.sleep(0)
        finally:
            self._running = False

    @property
    def running(self) -> bool:
        return self._running
