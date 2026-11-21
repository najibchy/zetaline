import io

import pytest
from rich.console import Console
from typer.testing import CliRunner

from zetaline.cli import main as cli
from zetaline.control import ControlClient

runner = CliRunner()


def text(renderable) -> str:
    buf = io.StringIO()
    Console(file=buf, width=140, color_system=None).print(renderable)
    return buf.getvalue()


STATUS = {
    "running": True,
    "uptime_s": 12.0,
    "symbols": ["BTCUSDT", "ETHUSDT"],
    "ticks": 1_234,
    "intents": 10,
    "approved": 9,
    "rejected": 1,
    "fills": 9,
    "executor_rejects": 0,
    "strategy_errors": 0,
    "kill_switch": False,
    "kill_reason": "",
    "portfolio": {
        "cash": 95_000.0,
        "equity": 99_900.0,
        "realized_pnl": -90.0,
        "unrealized_pnl": -10.0,
        "total_pnl": -100.0,
        "exposure": 4_900.0,
        "fees_paid": 20.0,
        "fills": 9,
        "positions": {},
    },
    "limits": {"max_loss": 2_000.0, "stale_tick_seconds": 2.0},
    "strategies": {"momentum-BTCUSDT": {"enabled": True, "params": {"fast": 10, "slow": 30}}},
    "bus": {},
}


# ---- rendering ---------------------------------------------------------------
def test_render_status_content():
    out = text(cli.render_status(STATUS))
    assert "running" in out and "1,234" in out and "-100.00" in out
    assert "BTCUSDT, ETHUSDT" in out and "off" in out


def test_render_status_shows_active_kill_switch_and_errors():
    st = {**STATUS, "kill_switch": True, "kill_reason": "max loss", "strategy_errors": 2}
    out = text(cli.render_status(st))
    assert "ACTIVE" in out and "max loss" in out and "strategy errors" in out


def test_render_positions_and_prices():
    pos = {"BTC": {"qty": 0.5, "avg_price": 100.0, "mark": 110.0,
                   "unrealized_pnl": 5.0, "realized_pnl": -1.0}}
    out = text(cli.render_positions(pos))
    assert "BTC" in out and "+5.00" in out and "-1.00" in out
    out = text(cli.render_prices({"BTC": {"bid": 99.0, "ask": 101.0, "mid": 100.0}}))
    assert "99.00" in out and "101.00" in out


def test_render_latency_fills_rejects_limits_strategies():
    lat = {"strategy": {"count": 5, "mean_us": 1.5, "p50_us": 1.4, "p95_us": 2.0,
                        "p99_us": 2.5, "max_us": 3.0}}
    assert "strategy" in text(cli.render_latency(lat))
    fills = [{"order_id": 1, "symbol": "BTC", "side": "BUY", "qty": 1.0,
              "price": 100.0, "fee": 0.05}]
    assert "BUY" in text(cli.render_fills(fills))
    rejects = [{"order_id": 2, "symbol": "BTC", "side": "BUY", "source": "risk_manager",
                "reason": "kill switch active"}]
    assert "kill switch active" in text(cli.render_rejects(rejects))
    assert "max_loss" in text(cli.render_limits({"max_loss": 2_000.0}))
    assert "momentum-BTCUSDT" in text(cli.render_strategies(STATUS["strategies"]))


def test_pnl_colors():
    assert "green" in cli.pnl(1.0)
    assert "red" in cli.pnl(-1.0)
    assert "+0.00" in cli.pnl(0.0)


# ---- commands with the control call stubbed -------------------------------------
@pytest.fixture
def stub(monkeypatch):
    calls = []
    responses = {}

    def fake_call(cmd, **args):
        calls.append((cmd, args))
        return responses.get(cmd)

    monkeypatch.setattr(cli, "call", fake_call)
    return calls, responses


def test_status_command(stub):
    _, responses = stub
    responses["status"] = STATUS
    result = runner.invoke(cli.app, ["status"])
    assert result.exit_code == 0 and "Zetaline status" in result.output


def test_positions_empty_and_nonempty(stub):
    _, responses = stub
    responses["positions"] = {}
    assert "No open positions" in runner.invoke(cli.app, ["positions"]).output
    responses["positions"] = {"BTC": {"qty": 1.0, "avg_price": 1.0, "mark": 1.0,
                                      "unrealized_pnl": 0.0, "realized_pnl": 0.0}}
    assert "BTC" in runner.invoke(cli.app, ["positions"]).output


def test_kill_and_resume_commands(stub):
    calls, responses = stub
    result = runner.invoke(cli.app, ["kill", "because"])
    assert "ACTIVE" in result.output and calls[-1] == ("kill", {"reason": "because"})
    responses["resume"] = True
    assert "cleared" in runner.invoke(cli.app, ["resume"]).output
    responses["resume"] = False
    assert "still breached" in runner.invoke(cli.app, ["resume"]).output


def test_risk_set_and_show(stub):
    calls, responses = stub
    responses["set_limits"] = {"max_loss": [2_000.0, 1_500.0]}
    result = runner.invoke(cli.app, ["risk", "--max-loss", "1500"])
    assert "2,000.00 -> 1,500.00" in result.output
    assert calls[-1] == ("set_limits", {"max_loss": 1500.0})
    responses["status"] = STATUS
    assert "Risk limits" in runner.invoke(cli.app, ["risk"]).output


def test_strategy_commands(stub):
    calls, responses = stub
    responses["status"] = STATUS
    assert "momentum-BTCUSDT" in runner.invoke(cli.app, ["strategy"]).output
    result = runner.invoke(cli.app, ["strategy", "disable", "momentum-BTCUSDT"])
    assert "disabled" in result.output
    assert calls[-1] == ("set_strategy_enabled", {"name": "momentum-BTCUSDT", "enabled": False})
    assert runner.invoke(cli.app, ["strategy", "enable"]).exit_code == 2
    assert runner.invoke(cli.app, ["strategy", "bogus", "x"]).exit_code == 2


# ---- no daemon ----------------------------------------------------------------------
def test_no_daemon_gives_friendly_error(tmp_path, monkeypatch):
    missing = tmp_path / "none.sock"
    monkeypatch.setattr(
        cli, "ControlClient", lambda: ControlClient(missing)
    )
    result = runner.invoke(cli.app, ["status"])
    assert result.exit_code == 1
    assert "No daemon running" in result.output
    assert "Traceback" not in result.output
