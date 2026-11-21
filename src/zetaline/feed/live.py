"""Live market data feed: Binance public bookTicker over WebSocket.

Same surface as MarketSimulator (symbols, ticks_published, running, run()),
so the runtime can swap feeds. Public market data only; no API key is used.
"""
from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Iterable
from typing import Any

from websockets.asyncio.client import connect
from websockets.exceptions import WebSocketException

from zetaline.core.events import EventBus, Topic
from zetaline.feed.binance import ParseError, Shutdown, build_url, parse_message


class IdleTimeout(TimeoutError):
    """No message arrived within idle_timeout; the connection is presumed dead."""


class LiveFeed:
    def __init__(
        self,
        bus: EventBus,
        symbols: Iterable[str],
        *,
        testnet: bool = False,
        url: str | None = None,
        open_timeout: float = 20.0,
        idle_timeout: float = 30.0,
        max_session_s: float = 23 * 3600.0,
        backoff_initial: float = 1.0,
        backoff_max: float = 30.0,
    ) -> None:
        symbols = list(symbols)
        built = build_url(symbols, testnet=testnet)  # validates the symbols
        for label, value in (
            ("open_timeout", open_timeout),
            ("idle_timeout", idle_timeout),
            ("max_session_s", max_session_s),
            ("backoff_initial", backoff_initial),
            ("backoff_max", backoff_max),
        ):
            if value <= 0:
                raise ValueError(f"{label} must be positive")
        if backoff_max < backoff_initial:
            raise ValueError("backoff_max must be >= backoff_initial")

        self._bus = bus
        self._url = url if url is not None else built
        self._symbols = [s.upper() for s in symbols]
        self._symbol_set = frozenset(self._symbols)
        self._open_timeout = open_timeout
        self._idle_timeout = idle_timeout
        self._max_session_s = max_session_s
        self._backoff_initial = backoff_initial
        self._backoff_max = backoff_max
        self._rng = random.Random()

        self.ticks_published = 0
        self.messages = 0
        self.parse_errors = 0
        self.ignored = 0
        self.connects = 0
        self.failures = 0
        self.rotations = 0
        self.shutdown_events = 0
        self.last_error = ""
        self.connected = False
        self._running = False
        self._last_message = 0.0  # time.monotonic() of the last frame

    @property
    def symbols(self) -> list[str]:
        return list(self._symbols)

    @property
    def running(self) -> bool:
        return self._running

    def stats(self) -> dict[str, Any]:
        age = (time.monotonic() - self._last_message) if self._last_message else None
        return {
            "url": self._url,
            "connected": self.connected,
            "connects": self.connects,
            "reconnects": max(0, self.connects - 1),
            "failures": self.failures,
            "rotations": self.rotations,
            "shutdown_events": self.shutdown_events,
            "messages": self.messages,
            "ticks_published": self.ticks_published,
            "parse_errors": self.parse_errors,
            "ignored": self.ignored,
            "last_error": self.last_error,
            "last_message_age_s": age,
        }

    def _backoff(self, attempt: int) -> float:
        base = min(self._backoff_max, self._backoff_initial * 2 ** max(0, attempt - 1))
        return base * self._rng.uniform(0.5, 1.0)

    def _done(self, max_ticks: int | None) -> bool:
        return max_ticks is not None and self.ticks_published >= max_ticks

    async def run(self, max_ticks: int | None = None) -> None:
        """Stream ticks until cancelled (or until max_ticks have been published)."""
        self._running = True
        attempt = 0
        try:
            while not self._done(max_ticks):
                before = self.messages
                clean = False
                try:
                    await self._session(max_ticks)
                    clean = True
                except (OSError, TimeoutError, WebSocketException) as exc:
                    self.failures += 1
                    self.last_error = f"{type(exc).__name__}: {exc}"
                finally:
                    self.connected = False
                if self._done(max_ticks):
                    break
                progressed = self.messages > before
                attempt = 0 if progressed else attempt + 1
                if clean and progressed:
                    await asyncio.sleep(0)  # rotation or shutdown: reconnect at once
                else:
                    await asyncio.sleep(self._backoff(attempt))
        finally:
            self._running = False
            self.connected = False

    async def _session(self, max_ticks: int | None) -> None:
        async with connect(self._url, open_timeout=self._open_timeout, close_timeout=5) as ws:
            self.connects += 1
            self.connected = True
            deadline = time.monotonic() + self._max_session_s
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.rotations += 1
                    return
                try:
                    raw = await asyncio.wait_for(
                        ws.recv(), timeout=min(self._idle_timeout, remaining)
                    )
                except TimeoutError:
                    if remaining <= self._idle_timeout:  # the deadline set this wait
                        self.rotations += 1
                        return
                    raise IdleTimeout(f"no message for {self._idle_timeout:g}s") from None
                recv_ns = time.perf_counter_ns()  # before parsing: latency includes parse time
                self.messages += 1
                self._last_message = time.monotonic()
                try:
                    item = parse_message(raw, recv_ns=recv_ns)
                except ParseError:
                    self.parse_errors += 1
                    continue
                if isinstance(item, Shutdown):
                    self.shutdown_events += 1
                    return
                if item is None:
                    continue
                if item.symbol not in self._symbol_set:
                    self.ignored += 1
                    continue
                self._bus.publish(Topic.TICK, item)
                self.ticks_published += 1
                if self._done(max_ticks):
                    return
