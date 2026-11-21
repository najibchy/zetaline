"""Low-overhead latency recorder: p50/p95/p99 per pipeline stage.

record() is O(1) and allocation-free (fixed ring buffer). Percentiles are
computed on demand from the rolling window, never on the hot path.
Record nanoseconds; report microseconds.
"""
from __future__ import annotations

import math
from array import array

now_ns = __import__("time").perf_counter_ns

STAGE_STRATEGY = "strategy"
STAGE_RISK_CHECK = "risk_check"
STAGE_EXECUTE = "execute"
STAGE_TICK_TO_INTENT = "tick_to_intent"
STAGE_TICK_TO_ORDER = "tick_to_order"
STAGE_TICK_TO_FILL = "tick_to_fill"


def percentile(sorted_vals: list[int], p: int) -> int:
    """Nearest-rank percentile of an already sorted list (p in 1..100)."""
    if not sorted_vals:
        return 0
    n = len(sorted_vals)
    k = math.ceil(p * n / 100) - 1
    return sorted_vals[max(0, min(n - 1, k))]


class StageStats:
    """Rolling window of samples plus all-time counters for one stage."""

    __slots__ = ("_buf", "_cap", "_idx", "count", "max_ns", "min_ns", "total_ns")

    def __init__(self, capacity: int = 100_000) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._cap = capacity
        self._buf = array("q", [0]) * capacity
        self._idx = 0
        self.count = 0
        self.total_ns = 0
        self.min_ns = 0
        self.max_ns = 0

    def record(self, ns: int) -> None:
        ns = max(ns, 0)
        self._buf[self._idx] = ns
        self._idx += 1
        if self._idx == self._cap:
            self._idx = 0
        if self.count == 0 or ns < self.min_ns:
            self.min_ns = ns
        self.max_ns = max(self.max_ns, ns)
        self.count += 1
        self.total_ns += ns

    def window(self) -> list[int]:
        """Samples currently in the rolling window (unordered)."""
        return self._buf[: min(self.count, self._cap)].tolist()

    def summary(self) -> dict[str, float]:
        vals = sorted(self.window())
        us = 1_000.0
        return {
            "count": self.count,
            "mean_us": (self.total_ns / self.count / us) if self.count else 0.0,
            "min_us": self.min_ns / us,
            "p50_us": percentile(vals, 50) / us,
            "p95_us": percentile(vals, 95) / us,
            "p99_us": percentile(vals, 99) / us,
            "max_us": self.max_ns / us,
        }


class LatencyRecorder:
    """A set of named stages, created on first use."""

    def __init__(self, capacity: int = 100_000) -> None:
        self._capacity = capacity
        self._stages: dict[str, StageStats] = {}

    def record(self, stage: str, ns: int) -> None:
        stats = self._stages.get(stage)
        if stats is None:
            stats = self._stages[stage] = StageStats(self._capacity)
        stats.record(ns)

    def stages(self) -> list[str]:
        return list(self._stages)

    def summary(self) -> dict[str, dict[str, float]]:
        """Plain-dict view for the CLI and the AI agent."""
        return {name: s.summary() for name, s in self._stages.items()}

    def reset(self) -> None:
        self._stages.clear()
