"""Daemon entry point: runs the Platform plus its control socket until shutdown.

Run with:  python -m zetaline.daemon [--rate 100] [--seed 42] [--max-loss 2000]
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


async def run_daemon(config: PlatformConfig, socket_path: str | None = None) -> None:
    platform = Platform(config)
    server = ControlServer(platform, socket_path)
    await server.start()  # fails fast if another daemon is listening
    try:
        await platform.start()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, server.shutdown_event.set)
        print(f"zetaline daemon up on {server.path}", flush=True)
        await server.shutdown_event.wait()
        print("zetaline daemon shutting down", flush=True)
    finally:
        await platform.stop()
        await server.stop()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Zetaline daemon")
    ap.add_argument("--rate", type=float, default=100.0, help="tick rounds per second")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-loss", type=float, default=2_000.0)
    ap.add_argument("--cash", type=float, default=100_000.0)
    ap.add_argument("--socket", default=None, help="override the control socket path")
    args = ap.parse_args(argv)
    config = PlatformConfig(
        rate_hz=args.rate, seed=args.seed, max_loss=args.max_loss, starting_cash=args.cash
    )
    try:
        with contextlib.suppress(KeyboardInterrupt):
            asyncio.run(run_daemon(config, args.socket))
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
