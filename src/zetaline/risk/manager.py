"""Risk manager: the only component allowed to publish to Topic.ORDER.

Strategies publish unchecked orders on Topic.INTENT. Every intent passes
through check(); approved orders are forwarded to the executor, rejected
ones are published as Rejection events.
"""
from __future__ import annotations

import time
from collections.abc import Callable

from zetaline.core.events import DropPolicy, EventBus, Subscription, Topic
from zetaline.core.latency import (
    STAGE_RISK_CHECK,
    STAGE_TICK_TO_ORDER,
    LatencyRecorder,
    now_ns,
)
from zetaline.core.models import Order, Rejection, Side
from zetaline.execution.portfolio import Portfolio
from zetaline.feed.book import OrderBook
from zetaline.risk.limits import RiskLimits


class RiskManager:
    SOURCE = "risk_manager"

    def __init__(
        self,
        bus: EventBus,
        book: OrderBook,
        portfolio: Portfolio,
        limits: RiskLimits | None = None,
        fee_buffer_bps: float = 10.0,
        clock_ns: Callable[[], int] = time.time_ns,
        *,
        latency: LatencyRecorder | None = None,
    ) -> None:
        self._latency = latency
        if fee_buffer_bps < 0:
            raise ValueError("fee_buffer_bps must be >= 0")
        self._bus = bus
        self._book = book
        self._portfolio = portfolio
        self._clock_ns = clock_ns
        self.limits = limits if limits is not None else RiskLimits()
        self.fee_buffer_bps = fee_buffer_bps
        self.kill_switch = False
        self.kill_reason = ""
        self.approved = 0
        self.rejected = 0
        # Subscribe immediately so intents published before run() are kept.
        self._sub: Subscription = bus.subscribe(
            Topic.INTENT, maxsize=10_000, policy=DropPolicy.DROP_NEWEST
        )

    # ---- kill switch -------------------------------------------------
    def trip_kill_switch(self, reason: str) -> None:
        if self.kill_switch:
            return
        self.kill_switch = True
        self.kill_reason = reason
        self._bus.publish(
            Topic.CONTROL,
            {"type": "kill_switch", "active": True, "reason": reason, "source": self.SOURCE},
        )

    def reset_kill_switch(self) -> None:
        """Clear the kill switch. If the loss limit is still breached, the
        next check() will trip it again."""
        if not self.kill_switch:
            return
        self.kill_switch = False
        self.kill_reason = ""
        self._bus.publish(
            Topic.CONTROL,
            {"type": "kill_switch", "active": False, "reason": "", "source": self.SOURCE},
        )

    def _maybe_trip_on_loss(self) -> None:
        if self.kill_switch:
            return
        pnl = self._portfolio.total_pnl(self._book)
        if pnl <= -self.limits.max_loss:
            self.trip_kill_switch(
                f"max loss breached: pnl {pnl:,.2f} <= -{self.limits.max_loss:,.2f}"
            )

    # ---- checks ------------------------------------------------------
    @staticmethod
    def _is_reduction(current_qty: float, signed_qty: float) -> bool:
        """True if the order only shrinks an existing position (no flip)."""
        return (
            current_qty != 0
            and (current_qty > 0) != (signed_qty > 0)
            and abs(signed_qty) <= abs(current_qty)
        )

    def check(self, order: Order) -> str | None:
        """Return None to approve, or a human-readable rejection reason."""
        self._maybe_trip_on_loss()
        lim = self.limits

        pos = self._portfolio.position(order.symbol)
        cur = pos.qty if pos is not None else 0.0
        signed = order.side.sign * order.qty
        new = cur + signed
        reduction = self._is_reduction(cur, signed)

        if self.kill_switch and not reduction:
            return (
                f"kill switch active ({self.kill_reason}): "
                "only position-reducing orders allowed"
            )

        tick = self._book.get(order.symbol)
        if tick is None:
            return f"no market data for {order.symbol}"
        age_s = (self._clock_ns() - tick.ts_ns) / 1e9
        if age_s > lim.stale_tick_seconds:
            return (
                f"stale market data for {order.symbol}: {age_s:.1f}s old "
                f"(max {lim.stale_tick_seconds:g}s)"
            )

        if order.qty > lim.max_order_qty:
            return f"order qty {order.qty:g} exceeds max {lim.max_order_qty:g}"

        price = tick.ask if order.side is Side.BUY else tick.bid
        notional = order.qty * price
        if notional > lim.max_order_notional:
            return (
                f"order notional {notional:,.2f} exceeds max {lim.max_order_notional:,.2f}"
            )

        if reduction:
            return None

        mid = tick.mid
        position_after = abs(new) * mid
        if position_after > lim.max_position_notional:
            return (
                f"position notional {position_after:,.2f} would exceed "
                f"max {lim.max_position_notional:,.2f}"
            )

        gross_after = self._portfolio.exposure(self._book) - abs(cur) * mid + position_after
        if gross_after > lim.max_gross_exposure:
            return (
                f"gross exposure {gross_after:,.2f} would exceed "
                f"max {lim.max_gross_exposure:,.2f}"
            )

        if order.side is Side.BUY:
            cost = notional * (1.0 + self.fee_buffer_bps / 10_000.0)
            if cost > self._portfolio.cash:
                return (
                    f"insufficient cash: need {cost:,.2f}, have {self._portfolio.cash:,.2f}"
                )

        return None

    # ---- pipeline ----------------------------------------------------
    def process(self, order: Order) -> Order | Rejection:
        """Check one order and publish the outcome."""
        t0 = now_ns()
        reason = self.check(order)
        if self._latency is not None:
            self._latency.record(STAGE_RISK_CHECK, now_ns() - t0)
        if reason is None:
            self.approved += 1
            self._bus.publish(Topic.ORDER, order)
            if self._latency is not None:
                self._latency.record(STAGE_TICK_TO_ORDER, now_ns() - order.origin_ns)
            return order
        rejection = Rejection(order=order, reason=reason, source=self.SOURCE)
        self.rejected += 1
        self._bus.publish(Topic.REJECT, rejection)
        return rejection

    async def run(self) -> None:
        """Consume order intents until stop() is called."""
        async for order in self._sub:
            self.process(order)

    def stop(self) -> None:
        self._sub.close()
