# Zetaline benchmarks

Pipeline: simulator -> strategy runner -> risk manager -> paper executor, one asyncio
event loop, plain CPython, no uvloop. Reproduce with `python scripts/benchmark.py`.

Environment: 13th Gen Intel(R) Core(TM) i7-1355U | Python 3.12.3 | seed 7, 1 symbol, EMA(3/6) momentum.

## Results (microseconds)

| Stage | Stress p50 | Stress p99 | Paced p50 | Paced p99 |
|---|---|---|---|---|
| strategy | 1.9 | 2.3 | 6.7 | 29.5 |
| risk_check | 2.7 | 3.3 | 23.2 | 39.6 |
| execute | 2.7 | 3.2 | 20.2 | 35.6 |
| tick_to_intent | 17.0 | 20.3 | 123.7 | 168.1 |
| tick_to_order | 33.3 | 38.2 | 192.6 | 268.1 |
| tick_to_fill | 48.7 | 54.9 | 232.6 | 346.5 |

- Stress: unthrottled simulator, 20,000 ticks, about 54,000 ticks/s.
- Paced: 1,000 ticks/s, 5,000 ticks.
- Zero bus drops in both runs, and every approved order was filled.

## Findings

- Component work (strategy + risk + execute) is about 7 us under load. Most of the
  end-to-end latency is asyncio task hops between components (about 15 us each).
- Paced latency is higher than stress latency. Likely cause: the loop and CPU idle
  between ticks (wake-up, cold caches, lower clock). Not yet verified.
- The latency recorder costs about 3 us per tick (18.5 us vs 15.6 us).
- The stress run had 726 risk rejections; the paced run had none. Cause (verified): the
  max-loss kill switch tripped (pnl -2,000.89 <= -2,000.00) after many small round trips
  paid spread and fees. Later buys were blocked, so the stress run is partly a
  kill-switch run. Raising `max_loss` in the benchmark would give a cleaner comparison.

## Caveats

- Counters (intents, fills) include 500 warmup ticks; latency samples do not.
- Percentiles use a rolling window of the last 100,000 samples per stage.
- tick_to_* includes queueing between components, by design.
- Simulated data only; real network and exchange latency are not measured.

## Next: Step 15

uvloop, fewer asyncio hops (fusing risk + execute), cheaper recording, and possibly
a compiled hot path.
