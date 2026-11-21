"""In-process asyncio event bus.

Design goals:
- publish() is synchronous and non-blocking (safe to call on the hot path).
- Every subscriber owns a bounded queue, so slow consumers can never stall
  producers. Overflow is handled by an explicit, counted drop policy.
"""
from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator
from enum import Enum
from typing import Any


class Topic(str, Enum):
    TICK = "tick"        # market data (Tick)
    INTENT = "intent"    # strategy -> risk manager (Order, unchecked)
    ORDER = "order"      # risk manager -> execution (Order, approved)
    FILL = "fill"        # execution results (Fill)
    REJECT = "reject"    # orders blocked by the risk manager
    CONTROL = "control"  # pause/resume/kill-switch/parameter changes


class DropPolicy(Enum):
    DROP_OLDEST = "drop_oldest"  # keep the freshest data (market data)
    DROP_NEWEST = "drop_newest"  # keep history, discard incoming


class _Closed:
    """Sentinel pushed into a queue when a subscription closes."""


_CLOSED = _Closed()


class Subscription:
    """A single subscriber's view of one topic."""

    __slots__ = ("_bus", "_closed", "_queue", "dropped", "policy", "topic")

    def __init__(
        self,
        bus: EventBus,
        topic: Topic,
        maxsize: int,
        policy: DropPolicy,
    ) -> None:
        self.topic = topic
        self.policy = policy
        self.dropped = 0
        self._queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=maxsize)
        self._bus = bus
        self._closed = False

    def _offer(self, event: Any) -> bool:
        """Non-blocking enqueue. Returns True if the event was accepted."""
        try:
            self._queue.put_nowait(event)
            return True
        except asyncio.QueueFull:
            self.dropped += 1
            if self.policy is DropPolicy.DROP_OLDEST:
                self._queue.get_nowait()  # discard the stalest event
                self._queue.put_nowait(event)
                return True
            return False

    async def get(self) -> Any:
        """Wait for the next event. Raises asyncio.CancelledError-free
        StopAsyncIteration semantics via the iterator interface instead."""
        item = await self._queue.get()
        if item is _CLOSED:
            raise EOFError("subscription closed")
        return item

    def get_nowait(self) -> Any:
        item = self._queue.get_nowait()
        if item is _CLOSED:
            raise EOFError("subscription closed")
        return item

    def qsize(self) -> int:
        return self._queue.qsize()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._bus._remove(self)
        # Wake any waiting consumer. Make room if the queue is full.
        if self._queue.full():
            self._queue.get_nowait()
        self._queue.put_nowait(_CLOSED)

    def __aiter__(self) -> AsyncIterator[Any]:
        return self

    async def __anext__(self) -> Any:
        item = await self._queue.get()
        if item is _CLOSED:
            raise StopAsyncIteration
        return item


class EventBus:
    """Topic-based fan-out bus. One instance per running platform."""

    def __init__(self) -> None:
        self._subs: dict[Topic, list[Subscription]] = defaultdict(list)
        self.published: dict[Topic, int] = defaultdict(int)

    def subscribe(
        self,
        topic: Topic,
        maxsize: int = 10_000,
        policy: DropPolicy = DropPolicy.DROP_OLDEST,
    ) -> Subscription:
        sub = Subscription(self, topic, maxsize, policy)
        self._subs[topic].append(sub)
        return sub

    def publish(self, topic: Topic, event: Any) -> int:
        """Fan out an event. Never blocks. Returns how many subscribers accepted it."""
        self.published[topic] += 1
        accepted = 0
        for sub in self._subs[topic]:
            if sub._offer(event):
                accepted += 1
        return accepted

    def subscriber_count(self, topic: Topic) -> int:
        return len(self._subs[topic])

    def stats(self) -> dict[str, dict[str, int]]:
        """Counters for health/latency reporting (used by the CLI later)."""
        return {
            "published": {t.value: n for t, n in self.published.items()},
            "dropped": {
                t.value: sum(s.dropped for s in subs) for t, subs in self._subs.items()
            },
        }

    def _remove(self, sub: Subscription) -> None:
        subs = self._subs[sub.topic]
        if sub in subs:
            subs.remove(sub)
