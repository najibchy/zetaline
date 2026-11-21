"""StrategyRunner: the runtime wiring between market data and strategies.

For every tick: update the order book, ask each enabled strategy for an
order, and publish any order to Topic.INTENT (the risk manager's input).
"""
from __future__ import annotations

from zetaline.core.events import DropPolicy, EventBus, Subscription, Topic
from zetaline.execution.portfolio import Portfolio
from zetaline.feed.book import OrderBook
from zetaline.strategy.base import Strategy


class StrategyRunner:
    SOURCE = "strategy_runner"

    def __init__(
        self,
        bus: EventBus,
        book: OrderBook,
        portfolio: Portfolio,
        strategies: list[Strategy] | None = None,
    ) -> None:
        self._bus = bus
        self._book = book
        self._portfolio = portfolio
        self._strategies: dict[str, Strategy] = {}
        self.ticks = 0
        self.intents = 0
        self.strategy_errors = 0
        for strategy in strategies or []:
            self.add(strategy)
        # DROP_OLDEST: for market data the freshest tick matters most.
        self._sub: Subscription = bus.subscribe(
            Topic.TICK, maxsize=10_000, policy=DropPolicy.DROP_OLDEST
        )

    # ---- strategy management (used by the CLI and the agent) ----------
    def add(self, strategy: Strategy) -> None:
        if strategy.name in self._strategies:
            raise ValueError(f"strategy {strategy.name!r} already registered")
        self._strategies[strategy.name] = strategy

    def remove(self, name: str) -> Strategy:
        try:
            return self._strategies.pop(name)
        except KeyError:
            raise KeyError(f"unknown strategy {name!r}") from None

    def get(self, name: str) -> Strategy:
        try:
            return self._strategies[name]
        except KeyError:
            raise KeyError(f"unknown strategy {name!r}") from None

    def names(self) -> list[str]:
        return sorted(self._strategies)

    def set_enabled(self, name: str, enabled: bool) -> None:
        self.get(name).enabled = enabled

    # ---- hot path ------------------------------------------------------
    def on_tick(self, tick) -> int:
        """Process one tick. Returns how many intents were published."""
        self._book.update(tick)
        self.ticks += 1
        published = 0
        pos = self._portfolio.position(tick.symbol)
        position_qty = pos.qty if pos is not None else 0.0

        for strategy in self._strategies.values():
            if not strategy.enabled or not strategy.wants(tick.symbol):
                continue
            try:
                order = strategy.on_tick(tick, position_qty)
            except Exception as exc:  # noqa: BLE001 - a faulty strategy must not kill the loop
                self.strategy_errors += 1
                strategy.enabled = False
                self._bus.publish(
                    Topic.CONTROL,
                    {
                        "type": "strategy_error",
                        "strategy": strategy.name,
                        "error": f"{type(exc).__name__}: {exc}",
                        "source": self.SOURCE,
                    },
                )
                continue
            if order is not None:
                self._bus.publish(Topic.INTENT, order)
                self.intents += 1
                published += 1
        return published

    async def run(self) -> None:
        """Consume ticks until stop() is called."""
        async for tick in self._sub:
            self.on_tick(tick)

    def stop(self) -> None:
        self._sub.close()
