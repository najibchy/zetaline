import asyncio
import json

import pytest

from zetaline.runtime import Platform, PlatformConfig


def fast_config(**kw) -> PlatformConfig:
    defaults = {"rate_hz": 500.0, "volatility": 0.002, "fast": 3, "slow": 6, "seed": 1}
    return PlatformConfig(**{**defaults, **kw})


def test_config_requires_symbols():
    with pytest.raises(ValueError):
        PlatformConfig(prices={})


def test_config_requires_positive_trade_notional():
    with pytest.raises(ValueError):
        PlatformConfig(trade_notional=0)


def test_one_strategy_per_symbol_sized_by_notional():
    p = Platform(PlatformConfig(prices={"BTC": 50_000.0}, trade_notional=2_000.0))
    assert p.runner.names() == ["momentum-BTC"]
    assert p.runner.get("momentum-BTC").params()["trade_qty"] == pytest.approx(0.04)


async def test_start_runs_pipeline_and_stop_reconciles():
    p = Platform(fast_config())
    await p.start()
    assert p.running
    await asyncio.sleep(0.5)
    await p.stop()
    s = p.status()
    assert not s["running"]
    assert s["ticks"] > 50
    assert s["intents"] == s["approved"] + s["rejected"]
    assert s["fills"] == s["approved"]
    assert all(t.done() for t in p._tasks)


async def test_double_start_refused():
    p = Platform(fast_config())
    await p.start()
    with pytest.raises(RuntimeError):
        await p.start()
    await p.stop()


async def test_stop_is_idempotent_and_safe_before_start():
    p = Platform(fast_config())
    await p.stop()  # never started
    await p.start()
    await p.stop()
    await p.stop()


async def test_queries_are_json_safe():
    p = Platform(fast_config())
    await p.start()
    await asyncio.sleep(0.3)
    json.dumps([
        p.status(), p.positions(), p.prices(), p.latency_summary(),
        p.recent_fills(), p.recent_rejects(), p.recent_events(),
    ])
    assert sorted(p.prices()) == sorted(p.status()["symbols"])
    await p.stop()


async def test_kill_and_resume_publish_events():
    p = Platform(fast_config())
    await p.start()
    p.kill("test")
    await asyncio.sleep(0.01)  # collector lags the bus by one loop iteration
    assert p.status()["kill_switch"] is True
    assert p.recent_events()[-1]["type"] == "kill_switch"
    assert p.resume() is True
    assert p.status()["kill_switch"] is False
    await p.stop()


async def test_set_limits_and_strategy_controls():
    p = Platform(fast_config())
    assert p.set_limits(max_loss=1_500.0) == {"max_loss": (2_000.0, 1_500.0)}
    with pytest.raises(ValueError):
        p.set_limits(max_loss=-1)
    assert p.limits.max_loss == 1_500.0
    assert p.set_strategy_params("momentum-BTCUSDT", threshold_bps=2.0) == {
        "threshold_bps": (1.0, 2.0)
    }
    p.set_strategy_enabled("momentum-ETHUSDT", False)
    assert p.status()["strategies"]["momentum-ETHUSDT"]["enabled"] is False
    with pytest.raises(KeyError):
        p.set_strategy_enabled("nope", True)


async def test_recent_lists_are_bounded():
    p = Platform(fast_config())
    await p.start()
    await asyncio.sleep(0.3)
    assert len(p.recent_fills(5)) <= 5
    await p.stop()
