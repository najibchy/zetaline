#!/usr/bin/env python3
"""Live check of the Gemini adapter: three stages, a handful of real requests.

Sends only made-up data (a stub replaces the trading daemon). Prints the model, the key's
source and a redacted key, never the key itself. Needs a valid .env (see .env.example).

Usage:
    python scripts/live_check.py
    python scripts/live_check.py --max-calls 6
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from collections.abc import Sequence
from typing import Any

from zetaline.agent.config import ConfigError, load_config
from zetaline.agent.gemini import GeminiClient
from zetaline.agent.llm import LLMClient, LLMError, LLMResponse, Message, ToolSpec
from zetaline.agent.loop import AgentLoop, AgentResult
from zetaline.agent.tools import ToolRegistry

STUB_PRICES = {
    "BTCUSDT": {"bid": 60000.0, "ask": 60001.0, "mid": 60000.5},
    "ETHUSDT": {"bid": 1680.0, "ask": 1680.1, "mid": 1680.05},
}


class BudgetExceeded(Exception):
    """The script's own cap on requests was reached."""


class Counting(LLMClient):
    """Counts requests and refuses to exceed a hard cap."""

    def __init__(self, inner: LLMClient, max_calls: int) -> None:
        self.inner = inner
        self.max_calls = max_calls
        self.calls = 0

    async def generate(
        self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> LLMResponse:
        if self.calls >= self.max_calls:
            raise BudgetExceeded
        self.calls += 1
        return await self.inner.generate(system, messages, tools)


class StubDaemon:
    """Stands in for the control socket: made-up prices, empty everything else."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call(self, cmd: str, **args: Any) -> Any:
        self.calls.append((cmd, args))
        return STUB_PRICES if cmd == "prices" else {}


def show(result: AgentResult) -> None:
    tools = [(t.name, t.ok) for t in result.traces]
    print(f"status: {result.status} | steps: {result.steps} | tools: {tools}")
    print(f"tokens in/out: {result.input_tokens}/{result.output_tokens}")
    if result.error:
        print(f"error: {result.error}")
    print(f"answer: {result.answer[:400]!r}")


async def stage_plain(llm: LLMClient) -> bool:
    start = time.perf_counter()
    async with asyncio.timeout(60):
        resp = await llm.generate(
            "You are a terse assistant.", [Message.user("Reply with exactly one word: pong")], []
        )
    print(f"reply: {resp.text.strip()!r} | {time.perf_counter() - start:.1f}s")
    print(f"tokens in/out: {resp.input_tokens}/{resp.output_tokens}")
    return bool(resp.text.strip())


async def stage_tools(llm: LLMClient) -> bool:
    agent = AgentLoop(llm, ToolRegistry(StubDaemon()), max_steps=4, timeout=60)
    result = await agent.ask("What is the current BTCUSDT mid price? Use your tools to check.")
    show(result)
    called = any(t.name == "get_prices" and t.ok for t in result.traces)
    return result.status == "ok" and called and "60000" in result.answer.replace(",", "")


async def stage_final(llm: LLMClient) -> bool:
    agent = AgentLoop(llm, ToolRegistry(StubDaemon()), max_steps=1, timeout=60)
    result = await agent.ask("Check the prices with your tools, then summarise them.")
    show(result)
    if result.status == "ok":
        print("note: the model answered without a tool call, so this run did not exercise it")
    return result.status == "limit" and bool(result.answer.strip())


async def main() -> int:
    ap = argparse.ArgumentParser(description="Live check of the Gemini adapter")
    ap.add_argument("--max-calls", type=int, default=8, help="hard cap on requests (default 8)")
    args = ap.parse_args()

    try:
        config = load_config()
    except ConfigError as exc:
        print(f"config problem: {exc}")
        return 2
    print(f"model: {config.model} | key from: {config.source} | {config.redacted()}")

    llm = Counting(GeminiClient(config), args.max_calls)
    stages = (
        ("plain call", stage_plain),
        ("tool round trip", stage_tools),
        ("forced final answer", stage_final),
    )
    results: dict[str, bool | None] = {name: None for name, _ in stages}
    for name, stage in stages:
        print(f"\n--- {name} ---")
        try:
            results[name] = await stage(llm)
        except BudgetExceeded:
            print(f"request budget of {args.max_calls} reached; stopping")
            results[name] = False
            break
        except TimeoutError:
            print("FAILED: no response within 60s")
            results[name] = False
            break
        except LLMError as exc:
            print(f"FAILED: {type(exc).__name__}: {exc}")
            results[name] = False
            break
        if not results[name]:
            break

    print("\n=== summary ===")
    for name, outcome in results.items():
        label = "not run" if outcome is None else ("PASS" if outcome else "FAIL")
        print(f"{name}: {label}")
    print(f"requests made: {llm.calls}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
