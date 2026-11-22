# Live market data

Zetaline can run on real prices while all trading stays simulated: orders go to the
paper executor and never reach an exchange. No API key is used anywhere in the feed.

## Source

Binance public `bookTicker` over WebSocket (best bid/ask, real time), combined-stream URL:

    wss://data-stream.binance.vision/stream?streams=btcusdt@bookTicker/ethusdt@bookTicker

- `data-stream.binance.vision` serves market data only, with real mainnet prices.
- `--testnet` switches to `wss://stream.testnet.binance.vision`, which has its own
  artificial order book (prices do not match the real market).

## Message format (verified against a live capture)

    {"stream":"btcusdt@bookTicker","data":{"u":103397853760,"s":"BTCUSDT",
     "b":"60086.94000000","B":"0.68754000","a":"60086.95000000","A":"1.65794000"}}

Prices and sizes are strings. There is no exchange timestamp, so `Tick.ts_ns` is the local
receive time.

## Measured on the development machine

- Connect (DNS + TLS + handshake): about 6 s on the first capture, hence a 20 s open timeout.
- 150 real ticks across 3 symbols: 7.57 s including the connect, 0 parse errors, 0 ignored.
- Rate: 536 ticks in about 10 s for 2 symbols, roughly 27 ticks/s per symbol.
- Real prices matched the market at capture time (BTC about 60,100, ETH about 1,680, SOL about 70).

## Operating rules handled by `LiveFeed`

- Connections are closed by the server after 24 hours: the feed rotates after 23 hours.
- The server may announce a shutdown: the feed reconnects immediately if it was receiving data.
- Reconnect uses exponential backoff with jitter (1 s up to 30 s); a session that received
  data resets the counter.
- Idle watchdog: no message for 30 s forces a reconnect.
- Malformed messages (bad JSON, crossed book, non-positive prices) are counted and skipped.

## Known limitations

- Staleness is measured against local receive time. A dead or stalled feed is caught by the
  risk manager (ticks older than 2 s are rejected), but delayed delivery on the exchange side
  is not detectable without an exchange timestamp.
- The `serverShutdown` payload shape is assumed (`"e": "serverShutdown"`). A wrong guess is
  harmless because any connection close also triggers a reconnect.
- Top-of-book only: fills ignore depth and queue position.
