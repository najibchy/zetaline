"""Risk limits: a validated, runtime-adjustable set of thresholds.

The CLI and (later) the AI agent change limits through update(), which is
all-or-nothing: a rejected change leaves the current limits untouched.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields, replace
from typing import Any


@dataclass(slots=True)
class RiskLimits:
    max_order_qty: float = 1_000.0
    max_order_notional: float = 10_000.0
    max_position_notional: float = 25_000.0
    max_gross_exposure: float = 50_000.0
    max_loss: float = 2_000.0
    stale_tick_seconds: float = 2.0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(f"{f.name} must be a number, got {value!r}")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{f.name} must be a finite positive number, got {value!r}")
        if not (
            self.max_order_notional
            <= self.max_position_notional
            <= self.max_gross_exposure
        ):
            raise ValueError(
                "need max_order_notional <= max_position_notional <= max_gross_exposure"
            )

    def update(self, **changes: Any) -> dict[str, tuple[Any, Any]]:
        """Validate and apply changes atomically. Returns {field: (old, new)}."""
        names = {f.name for f in fields(self)}
        unknown = set(changes) - names
        if unknown:
            raise ValueError(f"unknown risk limit(s): {sorted(unknown)}")
        candidate = replace(self, **changes)  # runs validation; raises on failure
        applied: dict[str, tuple[Any, Any]] = {}
        for name in changes:
            old, new = getattr(self, name), getattr(candidate, name)
            if old != new:
                applied[name] = (old, new)
                setattr(self, name, new)
        return applied

    def to_dict(self) -> dict[str, float]:
        return asdict(self)
