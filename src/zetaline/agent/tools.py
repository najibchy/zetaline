"""Read-only tools the agent may call.

Every tool maps to one daemon control command from READ_ONLY_COMMANDS; the
registry refuses to build otherwise, so state-changing commands (kill, resume,
set_*, shutdown) are unreachable from here. Results are bounded JSON strings and
failures come back as {"error": ...} so the model can recover; call() does not
raise on bad input or daemon trouble.
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from zetaline.agent.llm import ToolSpec
from zetaline.control import ControlError, DaemonNotRunning

READ_ONLY_COMMANDS = frozenset(
    {"status", "positions", "prices", "latency", "fills", "rejects", "events"}
)
DEFAULT_N = 20
MAX_N = 50
DEFAULT_MAX_CHARS = 8_000
MIN_MAX_CHARS = 200


class Caller(Protocol):
    async def call(self, cmd: str, **args: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class ToolOutcome:
    content: str  # JSON text handed back to the model
    ok: bool


@dataclass(frozen=True, slots=True)
class _Tool:
    spec: ToolSpec
    command: str
    takes_n: bool = False
    pick: Callable[[Any], Any] | None = None  # post-process the daemon reply


def _limits(st: dict[str, Any]) -> dict[str, Any]:
    return {
        "kill_switch": st.get("kill_switch"),
        "kill_reason": st.get("kill_reason"),
        "limits": st.get("limits"),
    }


def _strategies(st: dict[str, Any]) -> Any:
    return st.get("strategies", {})


def _feed(st: dict[str, Any]) -> Any:
    return st.get("feed", {})


_NO_ARGS: dict[str, Any] = {"type": "object", "properties": {}}
_N_ARG: dict[str, Any] = {
    "type": "object",
    "properties": {
        "n": {
            "type": "integer",
            "description": f"How many of the most recent items, 1 to {MAX_N} (default {DEFAULT_N}).",
        }
    },
}


def _build_tools() -> list[_Tool]:
    return [
        _Tool(
            ToolSpec(
                "get_status",
                "Overall platform status: running state, uptime, tick/intent/approved/rejected/"
                "fill counters, kill switch, portfolio (cash, equity, realized and unrealized "
                "P&L, fees paid, exposure, open positions), risk limits, strategies and feed.",
                _NO_ARGS,
            ),
            "status",
        ),
        _Tool(
            ToolSpec(
                "get_positions",
                "Open positions per symbol: quantity, average price, current mark price, "
                "unrealized and realized P&L. An empty object means the portfolio is flat.",
                _NO_ARGS,
            ),
            "positions",
        ),
        _Tool(
            ToolSpec(
                "get_prices",
                "Latest top-of-book bid, ask and mid price for every symbol.",
                _NO_ARGS,
            ),
            "prices",
        ),
        _Tool(
            ToolSpec(
                "get_latency",
                "Per-stage latency statistics in microseconds (count, mean, p50, p95, p99, max) "
                "for strategy, risk_check, execute, tick_to_intent, tick_to_order and "
                "tick_to_fill. Empty before any samples exist.",
                _NO_ARGS,
            ),
            "latency",
        ),
        _Tool(
            ToolSpec(
                "get_recent_fills",
                "The most recent executed paper fills, oldest first: order id, symbol, side, "
                "quantity, price, fee. Empty if nothing has traded yet.",
                _N_ARG,
            ),
            "fills",
            takes_n=True,
        ),
        _Tool(
            ToolSpec(
                "get_recent_rejects",
                "The most recent orders refused by the risk manager or the executor, oldest "
                "first, each with the reason (limit hit, kill switch, stale data, no data).",
                _N_ARG,
            ),
            "rejects",
            takes_n=True,
        ),
        _Tool(
            ToolSpec(
                "get_recent_events",
                "The most recent control events, oldest first: kill-switch trips and resets, "
                "strategy errors and feed errors.",
                _N_ARG,
            ),
            "events",
            takes_n=True,
        ),
        _Tool(
            ToolSpec(
                "get_risk_limits",
                "Kill-switch state (and reason) plus the current risk limits: max order "
                "quantity and notional, max position notional, max gross exposure, max loss, "
                "stale-tick threshold in seconds.",
                _NO_ARGS,
            ),
            "status",
            pick=_limits,
        ),
        _Tool(
            ToolSpec(
                "get_strategies",
                "Each strategy's name, whether it is enabled, and its current parameters.",
                _NO_ARGS,
            ),
            "status",
            pick=_strategies,
        ),
        _Tool(
            ToolSpec(
                "get_feed_health",
                "Market data feed health: kind (sim or live), whether it is running and "
                "connected, reconnect count, age of the last message, parse errors and any "
                "error text.",
                _NO_ARGS,
            ),
            "status",
            pick=_feed,
        ),
    ]


def _parse_args(tool: _Tool, args: Any) -> dict[str, Any]:
    """Validate model-supplied arguments. Raises TypeError or ValueError."""
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise TypeError("arguments must be an object")
    if not tool.takes_n:
        if args:
            raise ValueError(f"{tool.spec.name} takes no arguments, got {sorted(args)}")
        return {}
    extra = set(args) - {"n"}
    if extra:
        raise ValueError(f"unexpected argument(s) {sorted(extra)}; only 'n' is accepted")
    n = args.get("n", DEFAULT_N)
    if isinstance(n, float) and n.is_integer():
        n = int(n)
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError(f"n must be an integer, got {n!r}")
    if not 1 <= n <= MAX_N:
        raise ValueError(f"n must be between 1 and {MAX_N}, got {n}")
    return {"n": n}


class ToolRegistry:
    def __init__(
        self,
        caller: Caller,
        max_chars: int = DEFAULT_MAX_CHARS,
        timeout: float = 5.0,
    ) -> None:
        if max_chars < MIN_MAX_CHARS:
            raise ValueError(f"max_chars must be >= {MIN_MAX_CHARS}")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        tools = _build_tools()
        forbidden = {t.command for t in tools} - READ_ONLY_COMMANDS
        if forbidden:
            raise RuntimeError(f"tools may only use read-only commands, found {sorted(forbidden)}")
        self._caller = caller
        self._max_chars = max_chars
        self._timeout = timeout
        self._tools = {t.spec.name: t for t in tools}

    def specs(self) -> list[ToolSpec]:
        return [t.spec for t in self._tools.values()]

    def names(self) -> list[str]:
        return sorted(self._tools)

    def commands(self) -> frozenset[str]:
        """Every daemon command these tools can issue."""
        return frozenset(t.command for t in self._tools.values())

    @staticmethod
    def _error(message: str) -> ToolOutcome:
        return ToolOutcome(json.dumps({"error": message}), False)

    def _render(self, result: Any) -> str:
        text = json.dumps(result, separators=(",", ":"), default=str)
        if len(text) <= self._max_chars:
            return text
        return json.dumps(
            {"truncated": True, "original_chars": len(text), "preview": text[: self._max_chars]},
            separators=(",", ":"),
        )

    async def call(self, name: str, args: Any = None) -> ToolOutcome:
        tool = self._tools.get(name)
        if tool is None:
            return self._error(f"unknown tool {name!r}; available tools: {self.names()}")
        try:
            cmd_args = _parse_args(tool, args)
        except (TypeError, ValueError) as exc:
            return self._error(f"invalid arguments for {name}: {exc}")
        try:
            result = await asyncio.wait_for(
                self._caller.call(tool.command, **cmd_args), timeout=self._timeout
            )
            if tool.pick is not None:
                result = tool.pick(result)
        except DaemonNotRunning:
            return self._error("the trading daemon is not running (start it with: zetaline start)")
        except ControlError as exc:
            return self._error(f"daemon error: {exc}")
        except TimeoutError:  # before OSError: TimeoutError is a subclass of it
            return self._error(f"the daemon did not answer within {self._timeout:g}s")
        except OSError as exc:
            return self._error(f"connection problem: {type(exc).__name__}")
        except (ValueError, KeyError, AttributeError, TypeError):
            return self._error("the daemon returned an unexpected reply")
        return ToolOutcome(self._render(result), True)
