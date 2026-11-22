"""Provider-agnostic LLM interface for the agent.

The agent loop and tools only use the types defined here. A provider adapter
(e.g. Gemini) translates to and from them, so swapping providers touches one file.
"""
from __future__ import annotations

import asyncio
import random
import re
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

ROLES = ("user", "assistant", "tool")
_TOOL_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


# ---- errors --------------------------------------------------------------------
class LLMError(Exception):
    """Any LLM failure. Not retried unless it is one of the subclasses below."""


class RateLimitError(LLMError):
    """The provider throttled us (HTTP 429). retry_after is in seconds, if given."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class LLMUnavailable(LLMError):
    """Transient provider or network failure (5xx, timeout). Safe to retry."""


# ---- types -----------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A tool the model may call. `parameters` is a JSON Schema object."""

    name: str
    description: str
    parameters: dict[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )

    def __post_init__(self) -> None:
        if not _TOOL_NAME_RE.fullmatch(self.name):
            raise ValueError(f"invalid tool name: {self.name!r}")
        if not self.description:
            raise ValueError("tool description must be non-empty")


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Message:
    role: str
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str = ""  # tool messages only: which call this answers
    tool_name: str = ""  # tool messages only

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}, got {self.role!r}")
        if self.role == "tool" and not (self.tool_call_id and self.tool_name):
            raise ValueError("tool messages need tool_call_id and tool_name")
        if self.tool_calls and self.role != "assistant":
            raise ValueError("only assistant messages can carry tool_calls")

    @classmethod
    def user(cls, text: str) -> Message:
        return cls("user", text)

    @classmethod
    def assistant(cls, text: str = "", tool_calls: Sequence[ToolCall] = ()) -> Message:
        return cls("assistant", text, tuple(tool_calls))

    @classmethod
    def tool_result(cls, call: ToolCall, content: str) -> Message:
        return cls("tool", content, tool_call_id=call.id, tool_name=call.name)


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


# ---- interface ---------------------------------------------------------------------
class LLMClient(ABC):
    @abstractmethod
    async def generate(
        self,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
    ) -> LLMResponse:
        """One model turn: returns text and/or tool calls."""


class FakeLLM(LLMClient):
    """Replays a script of LLMResponse objects and exceptions. For tests, no network."""

    def __init__(self, script: Sequence[LLMResponse | Exception]) -> None:
        self._script = list(script)
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
    ) -> LLMResponse:
        self.calls.append({"system": system, "messages": list(messages), "tools": list(tools)})
        if not self._script:
            raise RuntimeError("FakeLLM script exhausted")
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class RetryingClient(LLMClient):
    """Wraps a client with retry on RateLimitError and LLMUnavailable.

    Never waits longer than max_delay: a 429 that asks for more than that is a
    quota problem (e.g. a daily limit), and waiting would just hang the agent.
    """

    def __init__(
        self,
        inner: LLMClient,
        max_retries: int = 3,
        base_delay: float = 2.0,
        max_delay: float = 60.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if base_delay <= 0 or max_delay < base_delay:
            raise ValueError("need 0 < base_delay <= max_delay")
        self._inner = inner
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._sleep = sleep
        self._rng = random.Random()
        self.retries = 0

    def _delay(self, exc: LLMError, attempt: int) -> float | None:
        """Seconds to wait before retrying, or None if we should give up now."""
        if isinstance(exc, RateLimitError) and exc.retry_after is not None:
            return exc.retry_after if exc.retry_after <= self._max_delay else None
        base = min(self._max_delay, self._base_delay * 2**attempt)
        return base * self._rng.uniform(0.5, 1.0)

    async def generate(
        self,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
    ) -> LLMResponse:
        attempt = 0
        while True:
            try:
                return await self._inner.generate(system, messages, tools)
            except (RateLimitError, LLMUnavailable) as exc:
                delay = self._delay(exc, attempt)
                if delay is None or attempt >= self._max_retries:
                    raise
                self.retries += 1
                await self._sleep(delay)
                attempt += 1
