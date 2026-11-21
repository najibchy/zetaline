"""Zetaline CLI: a thin client for the daemon's control socket."""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from zetaline.control import ControlClient, ControlError, DaemonNotRunning, default_socket_path

app = typer.Typer(help="Zetaline: AI-agent driven paper trading platform.", no_args_is_help=True)
console = Console()
err = Console(stderr=True)

STATE_DIR = Path.home() / ".local" / "state" / "zetaline"
LOG_PATH = STATE_DIR / "daemon.log"


# ---- helpers -----------------------------------------------------------------
def call(cmd: str, **args: Any) -> Any:
    """Run one control call; exit with a friendly message on failure."""
    try:
        return asyncio.run(ControlClient().call(cmd, **args))
    except DaemonNotRunning:
        err.print("[red]No daemon running.[/red] Start one with: [bold]zetaline start[/bold]")
        raise typer.Exit(1) from None
    except ControlError as exc:
        err.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from None


def money(x: float) -> str:
    return f"{x:,.2f}"


def pnl(x: float) -> str:
    color = "green" if x > 0 else "red" if x < 0 else "white"
    return f"[{color}]{x:+,.2f}[/{color}]"


def render_status(st: dict[str, Any]) -> Table:
    pf = st["portfolio"]
    table = Table(title="Zetaline status", show_header=False)
    table.add_column(style="bold")
    table.add_column()
    state = "[green]running[/green]" if st["running"] else "[red]stopped[/red]"
    table.add_row("state", f"{state}  (uptime {st['uptime_s']:.0f}s)")
    kill = f"[red]ACTIVE[/red] - {st['kill_reason']}" if st["kill_switch"] else "[green]off[/green]"
    table.add_row("kill switch", kill)
    table.add_row("symbols", ", ".join(st["symbols"]))
    table.add_row("ticks / intents", f"{st['ticks']:,} / {st['intents']:,}")
    table.add_row("approved / rejected", f"{st['approved']:,} / {st['rejected']:,}")
    table.add_row("fills", f"{st['fills']:,}")
    table.add_row("cash", money(pf["cash"]))
    table.add_row("equity", money(pf["equity"]))
    table.add_row("total P&L", pnl(pf["total_pnl"]))
    table.add_row("realized / unrealized", f"{pnl(pf['realized_pnl'])} / {pnl(pf['unrealized_pnl'])}")
    table.add_row("fees paid", money(pf["fees_paid"]))
    table.add_row("exposure", money(pf["exposure"]))
    if st["strategy_errors"]:
        table.add_row("strategy errors", f"[red]{st['strategy_errors']}[/red]")
    return table


def render_positions(positions: dict[str, Any]) -> Table:
    table = Table(title="Open positions")
    for col in ("symbol", "qty", "avg price", "mark", "unrealized", "realized"):
        table.add_column(col, justify="left" if col == "symbol" else "right")
    for symbol, p in sorted(positions.items()):
        table.add_row(
            symbol,
            f"{p['qty']:,.6g}",
            money(p["avg_price"]),
            money(p["mark"]),
            pnl(p["unrealized_pnl"]),
            pnl(p["realized_pnl"]),
        )
    return table


def render_prices(prices: dict[str, Any]) -> Table:
    table = Table(title="Prices")
    for col in ("symbol", "bid", "ask", "mid"):
        table.add_column(col, justify="left" if col == "symbol" else "right")
    for symbol, q in sorted(prices.items()):
        table.add_row(symbol, money(q["bid"]), money(q["ask"]), money(q["mid"]))
    return table


def render_latency(summary: dict[str, Any]) -> Table:
    table = Table(title="Latency (microseconds)")
    for col in ("stage", "count", "mean", "p50", "p95", "p99", "max"):
        table.add_column(col, justify="left" if col == "stage" else "right")
    for stage, s in summary.items():
        table.add_row(
            stage,
            f"{s['count']:,}",
            f"{s['mean_us']:.1f}",
            f"{s['p50_us']:.1f}",
            f"{s['p95_us']:.1f}",
            f"{s['p99_us']:.1f}",
            f"{s['max_us']:.1f}",
        )
    return table


def render_fills(fills: list[dict[str, Any]]) -> Table:
    table = Table(title="Recent fills")
    for col in ("order", "symbol", "side", "qty", "price", "fee"):
        table.add_column(col, justify="left" if col in ("symbol", "side") else "right")
    for f in fills:
        table.add_row(
            str(f["order_id"]), f["symbol"], f["side"], f"{f['qty']:,.6g}",
            money(f["price"]), f"{f['fee']:.4f}",
        )
    return table


def render_rejects(rejects: list[dict[str, Any]]) -> Table:
    table = Table(title="Recent rejections")
    for col in ("order", "symbol", "side", "source", "reason"):
        table.add_column(col)
    for r in rejects:
        table.add_row(str(r["order_id"]), r["symbol"], r["side"], r["source"], r["reason"])
    return table


def render_limits(limits: dict[str, float]) -> Table:
    table = Table(title="Risk limits")
    table.add_column("limit")
    table.add_column("value", justify="right")
    for name, value in limits.items():
        table.add_row(name, f"{value:,.2f}")
    return table


def render_strategies(strategies: dict[str, Any]) -> Table:
    table = Table(title="Strategies")
    table.add_column("name")
    table.add_column("enabled")
    table.add_column("params")
    for name, s in sorted(strategies.items()):
        enabled = "[green]yes[/green]" if s["enabled"] else "[red]no[/red]"
        table.add_row(name, enabled, ", ".join(f"{k}={v}" for k, v in s["params"].items()))
    return table


def _daemon_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = ""  # keep ROS paths out, as with the tests
    return env


def _daemon_args(rate: float, seed: int, max_loss: float, cash: float) -> list[str]:
    return [
        sys.executable, "-m", "zetaline.daemon",
        "--rate", str(rate), "--seed", str(seed),
        "--max-loss", str(max_loss), "--cash", str(cash),
    ]


# ---- lifecycle ---------------------------------------------------------------------
@app.command()
def start(
    rate: float = typer.Option(100.0, help="Tick rounds per second."),
    seed: int = typer.Option(42, help="Simulator seed."),
    max_loss: float = typer.Option(2_000.0, help="Max-loss kill-switch threshold."),
    cash: float = typer.Option(100_000.0, help="Starting cash."),
    foreground: bool = typer.Option(False, "--foreground", "-f", help="Run in this terminal."),
) -> None:
    """Start the trading daemon."""
    try:
        asyncio.run(ControlClient().call("ping"))
        err.print(f"[yellow]A daemon is already running[/yellow] on {default_socket_path()}")
        raise typer.Exit(1)
    except DaemonNotRunning:
        pass

    cmd = _daemon_args(rate, seed, max_loss, cash)
    if foreground:
        raise typer.Exit(subprocess.call(cmd, env=_daemon_env()))

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("ab") as log:
        proc = subprocess.Popen(
            cmd, stdout=log, stderr=log, stdin=subprocess.DEVNULL,
            start_new_session=True, env=_daemon_env(),
        )
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            tail = LOG_PATH.read_text(errors="replace").splitlines()[-5:]
            err.print(f"[red]Daemon exited early (code {proc.returncode}).[/red] Log tail:")
            for line in tail:
                err.print(f"  {line}")
            raise typer.Exit(1)
        try:
            asyncio.run(ControlClient().call("ping"))
            console.print(f"[green]Daemon started[/green] (pid {proc.pid}). Log: {LOG_PATH}")
            return
        except DaemonNotRunning:
            time.sleep(0.1)
    err.print(f"[red]Daemon did not answer within 5s.[/red] Check {LOG_PATH}")
    raise typer.Exit(1)


@app.command()
def stop() -> None:
    """Stop the daemon."""
    call("shutdown")
    path = default_socket_path()
    deadline = time.monotonic() + 5.0
    while path.exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    if path.exists():
        err.print("[yellow]Shutdown requested but the socket is still present.[/yellow]")
        raise typer.Exit(1)
    console.print("[green]Daemon stopped.[/green]")


# ---- queries ---------------------------------------------------------------------------
@app.command()
def status() -> None:
    """Show platform status, P&L and kill-switch state."""
    console.print(render_status(call("status")))


@app.command()
def positions() -> None:
    """Show open positions."""
    data = call("positions")
    if not data:
        console.print("No open positions.")
        return
    console.print(render_positions(data))


@app.command()
def prices() -> None:
    """Show the latest top-of-book prices."""
    data = call("prices")
    if not data:
        console.print("No market data yet.")
        return
    console.print(render_prices(data))


@app.command()
def latency() -> None:
    """Show per-stage latency percentiles."""
    data = call("latency")
    if not data:
        console.print("No latency samples yet.")
        return
    console.print(render_latency(data))


@app.command()
def fills(n: int = typer.Option(20, "-n", help="How many recent fills.")) -> None:
    """Show recent fills."""
    data = call("fills", n=n)
    console.print(render_fills(data) if data else "No fills yet.")


@app.command()
def rejects(n: int = typer.Option(20, "-n", help="How many recent rejections.")) -> None:
    """Show recent order rejections (risk manager and executor)."""
    data = call("rejects", n=n)
    console.print(render_rejects(data) if data else "No rejections.")


# ---- controls ------------------------------------------------------------------------------
@app.command()
def risk(
    max_loss: float | None = typer.Option(None, help="Set max loss."),
    max_order_notional: float | None = typer.Option(None, help="Set max order notional."),
    max_position_notional: float | None = typer.Option(None, help="Set max position notional."),
    max_gross_exposure: float | None = typer.Option(None, help="Set max gross exposure."),
    max_order_qty: float | None = typer.Option(None, help="Set max order quantity."),
    stale_tick_seconds: float | None = typer.Option(None, help="Set stale-tick threshold."),
) -> None:
    """Show risk limits, or change them with the options."""
    changes = {
        k: v
        for k, v in {
            "max_loss": max_loss,
            "max_order_notional": max_order_notional,
            "max_position_notional": max_position_notional,
            "max_gross_exposure": max_gross_exposure,
            "max_order_qty": max_order_qty,
            "stale_tick_seconds": stale_tick_seconds,
        }.items()
        if v is not None
    }
    if changes:
        applied = call("set_limits", **changes)
        if not applied:
            console.print("No change (values already set).")
        for name, (old, new) in applied.items():
            console.print(f"{name}: {old:,.2f} -> [bold]{new:,.2f}[/bold]")
        return
    console.print(render_limits(call("status")["limits"]))


@app.command()
def kill(reason: str = typer.Argument("manual", help="Why the kill switch is being tripped.")) -> None:
    """Trip the kill switch: only position-reducing orders are allowed."""
    call("kill", reason=reason)
    console.print("[red]Kill switch ACTIVE.[/red]")


@app.command()
def resume() -> None:
    """Clear the kill switch."""
    if call("resume"):
        console.print("[green]Kill switch cleared.[/green]")
    else:
        console.print("[yellow]Cleared, but the max-loss limit is still breached.[/yellow]")


@app.command()
def strategy(
    action: str = typer.Argument("list", help="list | enable | disable"),
    name: str | None = typer.Argument(None, help="Strategy name (for enable/disable)."),
) -> None:
    """List strategies, or enable/disable one."""
    if action == "list":
        console.print(render_strategies(call("status")["strategies"]))
        return
    if action not in ("enable", "disable") or name is None:
        err.print("[red]Usage:[/red] zetaline strategy [list | enable NAME | disable NAME]")
        raise typer.Exit(2)
    call("set_strategy_enabled", name=name, enabled=(action == "enable"))
    console.print(f"{name}: {'enabled' if action == 'enable' else 'disabled'}")


if __name__ == "__main__":
    app()
