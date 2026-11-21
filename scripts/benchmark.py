#!/usr/bin/env python3
"""Zetaline pipeline benchmark: simulator -> runner -> risk -> executor.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --ticks 50000 --rate 500 --symbols 3
"""
from __future__ import annotations

import argparse
import asyncio
import time

from zetaline.core.events import EventBus
from zetaline.core.latency import LatencyRecorder
from zetaline.execution.paper import PaperExecutor
from zetaline.execution.portfolio import Portfolio
from zetaline.feed.book import OrderBook
from zetaline.feed.simulator import MarketSimulator
from zetaline.risk.manager import RiskManager
from zetaline.strategy.momentum import MomentumStrategy
from zetaline.strategy.runner import StrategyRunner

BASE_PRICES = {"S0": 100.0, "S1": 50.0, "S2": 200.0, "S3": 25.0, "S4": 400.0}


async def drain(sim, runner, rm, ex, timeout: float = 10.0) -> bool:
    """Wait until every published tick/intent/order has been fully processed."""
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if (
            runner.ticks == sim.ticks_published
            and rm.approved + rm.rejected == runner.intents
            and ex.fills + ex.rejects == rm.approved
        ):
            return True
        await asyncio.sleep(0.005)
    return False


async def run_scenario(ticks: int, rate_hz: float, seed: int, n_symbols: int, instrument: bool):
    prices = dict(list(BASE_PRICES.items())[:n_symbols])
    bus, book, pf = EventBus(), OrderBook(), Portfolio(1_000_000.0)
    lat = LatencyRecorder() if instrument else None
    runner = StrategyRunner(
        bus, book, pf, [MomentumStrategy(fast=3, slow=6, trade_qty=1.0)], latency=lat
    )
    rm = RiskManager(bus, book, pf, latency=lat)
    ex = PaperExecutor(bus, book, pf, latency=lat)
    sim = MarketSimulator(bus, prices=prices, seed=seed, rate_hz=rate_hz, volatility=0.005)

    components = (runner, rm, ex)
    tasks = [asyncio.create_task(c.run()) for c in components]

    # Warmup: let Python warm its caches, then discard those samples.
    warm = min(500, max(50, ticks // 10))
    await sim.run(max_ticks=warm)
    await drain(sim, runner, rm, ex)
    if lat is not None:
        lat.reset()

    start = time.perf_counter()
    await sim.run(max_ticks=ticks)
    drained = await drain(sim, runner, rm, ex)
    elapsed = time.perf_counter() - start

    for c in components:
        c.stop()
    await asyncio.gather(*tasks)

    return {
        "lat": lat,
        "elapsed": elapsed,
        "ticks": ticks * len(prices),
        "drained": drained,
        "runner": runner,
        "rm": rm,
        "ex": ex,
        "bus": bus,
    }


def print_table(lat: LatencyRecorder) -> None:
    print(f"  {'stage':<16}{'count':>9}{'mean':>9}{'p50':>9}{'p95':>9}{'p99':>9}{'max':>10}  (us)")
    for name, s in lat.summary().items():
        print(
            f"  {name:<16}{s['count']:>9}{s['mean_us']:>9.1f}{s['p50_us']:>9.1f}"
            f"{s['p95_us']:>9.1f}{s['p99_us']:>9.1f}{s['max_us']:>10.1f}"
        )


def report(title: str, r: dict) -> None:
    print(f"\n=== {title} ===")
    runner, rm, ex = r["runner"], r["rm"], r["ex"]
    rate = r["ticks"] / r["elapsed"] if r["elapsed"] > 0 else 0.0
    print(f"  ticks: {r['ticks']:,}  wall: {r['elapsed']:.2f}s  throughput: {rate:,.0f} ticks/s")
    print(
        f"  intents: {runner.intents:,}  approved: {rm.approved:,}  rejected: {rm.rejected:,}  "
        f"fills: {ex.fills:,}  drained: {r['drained']}"
    )
    print(f"  bus drops: {r['bus'].stats()['dropped']}")
    if r["lat"] is not None:
        print_table(r["lat"])


async def main() -> None:
    ap = argparse.ArgumentParser(description="Zetaline pipeline benchmark")
    ap.add_argument("--ticks", type=int, default=20_000, help="rounds per symbol (default 20000)")
    ap.add_argument("--rate", type=float, default=1_000.0, help="paced rounds/s (default 1000)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--symbols", type=int, default=1, choices=range(1, 6), help="1-5 symbols")
    args = ap.parse_args()

    stress = await run_scenario(args.ticks, 0.0, args.seed, args.symbols, instrument=True)
    report("STRESS (unthrottled)", stress)

    paced_ticks = min(args.ticks, 5_000)
    paced = await run_scenario(paced_ticks, args.rate, args.seed, args.symbols, instrument=True)
    report(f"PACED ({args.rate:,.0f} rounds/s)", paced)

    bare = await run_scenario(args.ticks, 0.0, args.seed, args.symbols, instrument=False)
    t_on = stress["elapsed"] / stress["ticks"] * 1e6
    t_off = bare["elapsed"] / bare["ticks"] * 1e6
    print("\n=== INSTRUMENTATION OVERHEAD (stress, per tick) ===")
    print(f"  with recorder: {t_on:.2f} us   without: {t_off:.2f} us   delta: {t_on - t_off:+.2f} us")


if __name__ == "__main__":
    asyncio.run(main())
