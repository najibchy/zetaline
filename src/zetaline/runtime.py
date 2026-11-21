"""Platform runtime: wires every component onto one bus and manages its lifecycle.

A Platform is single-use: after stop(), create a new one. All public query and
control methods return plain JSON-safe dicts, so the daemon's control socket and
the AI agent can use them directly.
"""
from __future__ import annotations

import asyncio
import contextlib
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from zetaline.core.events import EventBus, Subscription, Topic
from zetaline.core.latency import LatencyRecorder
from zetaline.core.models import Fill, Rejection
from zetaline.execution.paper import PaperExecutor
from zetaline.execution.portfolio import Portfolio
from zetaline.feed.book import OrderBook
from zetaline.feed.simulator import DEFAULT_PRICES, MarketSimulator
from zetaline.risk.limits import RiskLimits
from zetaline.risk.manager import RiskManager
from zetaline.strategy.momentum import MomentumStrategy
from zetaline.strategy.runner import StrategyRunner

RECENT_MAXLEN = 200


@dataclass(slots=True)
class PlatformConfig:
    prices: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_PRICES))
    seed: int = 42
    rate_hz: float = 100.0
    volatility: float = 0.0005
    spread_bps: float = 2.0
    starting_cash: float = 100_000.0
    fee_bps: float = 5.0
    max_loss: float = 2_000.0
    fast: int = 10
    slow: int = 30
    trade_notional: float = 2_000.0
    threshold_bps: float = 1.0

    def __post_init__(self) -> None:
        if not self.prices:
            raise ValueError("at least one symbol is required")
        if self.trade_notional <= 0:
            raise ValueError("trade_notional must be positive")


def _fill_dict(f: Fill) -> dict[str, Any]:
    return {
        "order_id": f.order_id,
        "symbol": f.symbol,
        "side": f.side.value,
        "qty": f.qty,
        "price": f.price,
        "fee": f.fee,
        "ts_ns": f.ts_ns,
    }


def _reject_dict(r: Rejection) -> dict[str, Any]:
    return {
        "order_id": r.order.order_id,
        "symbol": r.order.symbol,
        "side": r.order.side.value,
        "qty": r.order.qty,
        "strategy": r.order.strategy,
        "reason": r.reason,
        "source": r.source,
        "ts_ns": r.ts_ns,
    }


class Platform:
    def __init__(self, config: PlatformConfig | None = None) -> None:
        cfg = self.config = config or PlatformConfig()
        self.bus = EventBus()
        self.book = OrderBook()
        self.portfolio = Portfolio(cfg.starting_cash)
        self.latency = LatencyRecorder()
        self.limits = RiskLimits(max_loss=cfg.max_loss)

        strategies = [
            MomentumStrategy(
                symbols=[symbol],
                fast=cfg.fast,
                slow=cfg.slow,
                trade_qty=round(cfg.trade_notional / price, 6),
                threshold_bps=cfg.threshold_bps,
                name=f"momentum-{symbol}",
            )
            for symbol, price in cfg.prices.items()
        ]
        self.runner = StrategyRunner(
            self.bus, self.book, self.portfolio, strategies, latency=self.latency
        )
        self.risk = RiskManager(
            self.bus, self.book, self.portfolio, limits=self.limits, latency=self.latency
        )
        self.executor = PaperExecutor(
            self.bus, self.book, self.portfolio, fee_bps=cfg.fee_bps, latency=self.latency
        )
        self.sim = MarketSimulator(
            self.bus,
            prices=cfg.prices,
            seed=cfg.seed,
            rate_hz=cfg.rate_hz,
            volatility=cfg.volatility,
            spread_bps=cfg.spread_bps,
        )

        self._fills: deque[dict[str, Any]] = deque(maxlen=RECENT_MAXLEN)
        self._rejects: deque[dict[str, Any]] = deque(maxlen=RECENT_MAXLEN)
        self._events: deque[dict[str, Any]] = deque(maxlen=RECENT_MAXLEN)
        self._collectors: list[tuple[Subscription, deque, Any]] = [
            (self.bus.subscribe(Topic.FILL), self._fills, _fill_dict),
            (self.bus.subscribe(Topic.REJECT), self._rejects, _reject_dict),
            (self.bus.subscribe(Topic.CONTROL), self._events, dict),
        ]
        self._tasks: list[asyncio.Task] = []
        self._sim_task: asyncio.Task | None = None
        self._running = False
        self._started_at = 0.0

    # ---- lifecycle -------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._running

    async def start(self) -> None:
        if self._running:
            raise RuntimeError("platform already running")
        self._running = True
        self._started_at = time.monotonic()
        for component in (self.runner, self.risk, self.executor):
            self._tasks.append(asyncio.create_task(component.run()))
        for sub, store, convert in self._collectors:
            self._tasks.append(asyncio.create_task(self._collect(sub, store, convert)))
        # Consumers first, producer last, so no early ticks are lost.
        self._sim_task = asyncio.create_task(self.sim.run())

    @staticmethod
    async def _collect(sub: Subscription, store: deque, convert) -> None:
        async for item in sub:
            store.append(convert(item))

    def _drained(self) -> bool:
        dropped_ticks = self.bus.stats()["dropped"].get("tick", 0)
        return (
            self.runner.ticks + dropped_ticks >= self.sim.ticks_published
            and self.risk.approved + self.risk.rejected == self.runner.intents
            and self.executor.fills + self.executor.rejects == self.risk.approved
        )

    async def stop(self, drain_timeout: float = 2.0) -> None:
        if not self._running:
            return
        if self._sim_task is not None:
            self._sim_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._sim_task
        deadline = time.monotonic() + drain_timeout
        while not self._drained() and time.monotonic() < deadline:
            await asyncio.sleep(0.005)
        for component in (self.runner, self.risk, self.executor):
            component.stop()
        for sub, _, _ in self._collectors:
            sub.close()
        await asyncio.gather(*self._tasks)
        self._running = False

    # ---- queries (JSON-safe) --------------------------------------------------
    def status(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "uptime_s": (time.monotonic() - self._started_at) if self._running else 0.0,
            "symbols": self.sim.symbols,
            "ticks": self.runner.ticks,
            "intents": self.runner.intents,
            "approved": self.risk.approved,
            "rejected": self.risk.rejected,
            "fills": self.executor.fills,
            "executor_rejects": self.executor.rejects,
            "strategy_errors": self.runner.strategy_errors,
            "kill_switch": self.risk.kill_switch,
            "kill_reason": self.risk.kill_reason,
            "portfolio": self.portfolio.snapshot(self.book),
            "limits": self.limits.to_dict(),
            "strategies": {
                name: {
                    "enabled": self.runner.get(name).enabled,
                    "params": self.runner.get(name).params(),
                }
                for name in self.runner.names()
            },
            "bus": self.bus.stats(),
        }

    def positions(self) -> dict[str, Any]:
        return self.portfolio.snapshot(self.book)["positions"]

    def prices(self) -> dict[str, dict[str, float]]:
        return {
            s: {"bid": t.bid, "ask": t.ask, "mid": t.mid}
            for s, t in self.book.snapshot().items()
        }

    def latency_summary(self) -> dict[str, dict[str, float]]:
        return self.latency.summary()

    def recent_fills(self, n: int = 20) -> list[dict[str, Any]]:
        return list(self._fills)[-n:]

    def recent_rejects(self, n: int = 20) -> list[dict[str, Any]]:
        return list(self._rejects)[-n:]

    def recent_events(self, n: int = 20) -> list[dict[str, Any]]:
        return list(self._events)[-n:]

    # ---- controls -------------------------------------------------------------
    def kill(self, reason: str = "manual") -> None:
        self.risk.trip_kill_switch(reason)

    def resume(self) -> bool:
        """Clear the kill switch. Returns True if it is still clear afterwards."""
        self.risk.reset_kill_switch()
        return not self.risk.kill_switch

    def set_limits(self, **changes: Any) -> dict[str, tuple[Any, Any]]:
        return self.limits.update(**changes)

    def set_strategy_params(self, name: str, **changes: Any) -> dict[str, tuple[Any, Any]]:
        return self.runner.get(name).set_params(**changes)

    def set_strategy_enabled(self, name: str, enabled: bool) -> None:
        self.runner.set_enabled(name, enabled)
