"""Daemon entry point: runs the Platform plus its control socket until shutdown.

Run with:  python -m zetaline.daemon [--feed sim|live] [--rate 100] [--seed 42]
Stops on the `shutdown` control command, SIGINT, or SIGTERM.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import signal
import sys

from zetaline.control import ControlServer
from zetaline.runtime import Platform, PlatformConfig

DEFAULT_LIVE_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


async def run_daemon(config: PlatformConfig, socket_path: str | None = None) -> None:
    platform = Platform(config)
    server = ControlServer(platform, socket_path)
    await server.start()  # fails fast if another daemon is listening
    try:
        await platform.start()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, server.shutdown_event.set)
        print(f"zetaline daemon up on {server.path} (feed: {config.feed})", flush=True)
        await server.shutdown_event.wait()
        print("zetaline daemon shutting down", flush=True)
    finally:
        await platform.stop()
        await server.stop()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Zetaline daemon")
    ap.add_argument("--feed", choices=["sim", "live"], default="sim",
                    help="market data source (default: sim)")
    ap.add_argument("--testnet", action="store_true", help="live feed: use the spot testnet host")
    ap.add_argument("--symbols", nargs="+", default=None, help="live feed: symbols to stream")
    ap.add_argument("--rate", type=float, default=100.0, help="sim feed: tick rounds per second")
    ap.add_argument("--seed", type=int, default=42, help="sim feed: random seed")
    ap.add_argument("--max-loss", type=float, default=2_000.0)
    ap.add_argument("--cash", type=float, default=100_000.0)
    ap.add_argument("--socket", default=None, help="override the control socket path")
    args = ap.parse_args(argv)

    if args.feed != "live" and (args.testnet or args.symbols):
        ap.error("--testnet and --symbols require --feed live")

    options: dict = {
        "rate_hz": args.rate,
        "seed": args.seed,
        "max_loss": args.max_loss,
        "starting_cash": args.cash,
        "feed": args.feed,
        "testnet": args.testnet,
    }
    if args.feed == "live":
        # Only the symbol names matter for the live feed; the prices are unused.
        symbols = [s.upper() for s in (args.symbols or DEFAULT_LIVE_SYMBOLS)]
        options["prices"] = {s: 1.0 for s in symbols}

    try:
        config = PlatformConfig(**options)
        with contextlib.suppress(KeyboardInterrupt):
            asyncio.run(run_daemon(config, args.socket))
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
