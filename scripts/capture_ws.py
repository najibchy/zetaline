#!/usr/bin/env python3
"""Connectivity check and raw capture of Binance bookTicker messages.

Public market data only: no API key is needed or used.
Usage:
    python scripts/capture_ws.py
    python scripts/capture_ws.py --symbols btcusdt ethusdt solusdt -n 10
    python scripts/capture_ws.py --testnet
"""
from __future__ import annotations

import argparse
import asyncio
import time

from websockets.asyncio.client import connect
from websockets.exceptions import WebSocketException

MAINNET = "wss://data-stream.binance.vision"
TESTNET = "wss://stream.testnet.binance.vision"


async def main() -> int:
    ap = argparse.ArgumentParser(description="Capture raw Binance bookTicker messages")
    ap.add_argument("--symbols", nargs="+", default=["btcusdt", "ethusdt"])
    ap.add_argument("-n", type=int, default=6, help="messages to capture")
    ap.add_argument("--testnet", action="store_true", help="use the spot testnet host")
    args = ap.parse_args()

    base = TESTNET if args.testnet else MAINNET
    streams = "/".join(f"{s.lower()}@bookTicker" for s in args.symbols)
    url = f"{base}/stream?streams={streams}"
    print("connecting:", url)

    t0 = time.perf_counter()
    try:
        async with connect(url, open_timeout=10) as ws:
            print(f"connected in {(time.perf_counter() - t0) * 1000:.0f} ms")
            stamps = []
            for i in range(args.n):
                raw = await asyncio.wait_for(ws.recv(), timeout=10)
                stamps.append(time.perf_counter())
                print(f"[{i}] {raw}")
            if len(stamps) > 1:
                span = stamps[-1] - stamps[0]
                print(f"{len(stamps) - 1} gaps in {span:.2f}s -> ~{(len(stamps) - 1) / span:.1f} msg/s")
    except (OSError, TimeoutError, WebSocketException) as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
