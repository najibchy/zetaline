"""EMA-crossover momentum strategy (long-only, target-position based).

Position size is either a fixed quantity (trade_qty) or, when trade_notional is
given, a notional amount converted to a quantity at the moment of entry.
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from zetaline.core.models import Order, Side, Tick
from zetaline.strategy.base import Strategy


class _EmaState:
    __slots__ = ("count", "fast", "slow")

    def __init__(self, price: float) -> None:
        self.fast = price
        self.slow = price
        self.count = 1


class MomentumStrategy(Strategy):
    def __init__(
        self,
        symbols: Iterable[str] | None = None,
        fast: int = 10,
        slow: int = 30,
        trade_qty: float = 1.0,
        threshold_bps: float = 1.0,
        name: str = "momentum",
        *,
        trade_notional: float | None = None,
    ) -> None:
        super().__init__(name, symbols)
        self._validate(
            fast=fast,
            slow=slow,
            threshold_bps=threshold_bps,
            trade_qty=None if trade_notional is not None else trade_qty,
            trade_notional=trade_notional,
        )
        self.fast_n = fast
        self.slow_n = slow
        self.trade_notional = float(trade_notional) if trade_notional is not None else None
        self.trade_qty = float(trade_qty)  # unused in notional mode
        self.threshold_bps = float(threshold_bps)
        self._alpha_fast = 2.0 / (fast + 1)
        self._alpha_slow = 2.0 / (slow + 1)
        self._state: dict[str, _EmaState] = {}

    @staticmethod
    def _validate(
        fast: Any,
        slow: Any,
        threshold_bps: Any,
        trade_qty: Any = None,
        trade_notional: Any = None,
    ) -> None:
        for label, value in (("fast", fast), ("slow", slow)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{label} must be an int, got {value!r}")
        if trade_notional is not None:
            size_label, size = "trade_notional", trade_notional
        else:
            size_label, size = "trade_qty", trade_qty
        for label, value in ((size_label, size), ("threshold_bps", threshold_bps)):
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(f"{label} must be a number, got {value!r}")
        if fast < 1 or slow < 2 or fast >= slow:
            raise ValueError(f"need 1 <= fast < slow, got fast={fast}, slow={slow}")
        if not math.isfinite(size) or size <= 0:
            raise ValueError(f"{size_label} must be a finite positive number, got {size!r}")
        if not math.isfinite(threshold_bps) or threshold_bps < 0:
            raise ValueError(f"threshold_bps must be finite and >= 0, got {threshold_bps!r}")

    def _long_target(self, mid: float, position_qty: float) -> float:
        """Target position when the signal is long."""
        if self.trade_notional is None:
            return self.trade_qty
        if position_qty > 1e-9:
            return position_qty  # already long: hold, don't resize as the price drifts
        return round(self.trade_notional / mid, 6)

    def on_tick(self, tick: Tick, position_qty: float) -> Order | None:
        mid = tick.mid
        st = self._state.get(tick.symbol)
        if st is None:
            self._state[tick.symbol] = _EmaState(mid)
            return None

        st.fast += self._alpha_fast * (mid - st.fast)
        st.slow += self._alpha_slow * (mid - st.slow)
        st.count += 1
        if st.count < self.slow_n:
            return None  # warming up

        diff_bps = (st.fast - st.slow) / st.slow * 10_000.0
        if diff_bps > self.threshold_bps:
            target = self._long_target(mid, position_qty)
        elif diff_bps < -self.threshold_bps:
            target = 0.0
        else:
            return None  # inside the deadband: hold

        delta = target - position_qty
        if abs(delta) < 1e-9:
            return None
        return self._order(tick, Side.BUY if delta > 0 else Side.SELL, abs(delta))

    def params(self) -> dict[str, Any]:
        size = (
            {"trade_notional": self.trade_notional}
            if self.trade_notional is not None
            else {"trade_qty": self.trade_qty}
        )
        return {
            "fast": self.fast_n,
            "slow": self.slow_n,
            **size,
            "threshold_bps": self.threshold_bps,
        }

    def set_params(self, **changes: Any) -> dict[str, tuple[Any, Any]]:
        unknown = set(changes) - set(self.params())
        if unknown:
            raise ValueError(f"unknown parameter(s): {sorted(unknown)}")
        merged = {**self.params(), **changes}
        self._validate(**merged)  # raises before anything is applied
        applied = {
            k: (old, merged[k]) for k, old in self.params().items() if merged[k] != old
        }
        self.fast_n, self.slow_n = merged["fast"], merged["slow"]
        if self.trade_notional is not None:
            self.trade_notional = float(merged["trade_notional"])
        else:
            self.trade_qty = float(merged["trade_qty"])
        self.threshold_bps = float(merged["threshold_bps"])
        if "fast" in applied or "slow" in applied:
            self._alpha_fast = 2.0 / (self.fast_n + 1)
            self._alpha_slow = 2.0 / (self.slow_n + 1)
            self._state.clear()  # old EMAs used the old smoothing
        return applied
