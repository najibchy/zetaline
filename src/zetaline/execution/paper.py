"""Paper execution engine: fills orders against the current top of book.

Buys fill at the ask, sells at the bid. Orders that cannot be filled are
published as Rejection events rather than dropped silently.
"""
from __future__ import annotations

from zetaline.core.events import DropPolicy, EventBus, Subscription, Topic
from zetaline.core.models import Fill, Order, OrderType, Rejection, Side
from zetaline.execution.portfolio import Portfolio
from zetaline.feed.book import OrderBook


class PaperExecutor:
    SOURCE = "paper_executor"

    def __init__(
        self,
        bus: EventBus,
        book: OrderBook,
        portfolio: Portfolio,
        fee_bps: float = 5.0,
    ) -> None:
        if fee_bps < 0:
            raise ValueError("fee_bps must be >= 0")
        self._bus = bus
        self._book = book
        self._portfolio = portfolio
        self.fee_bps = fee_bps
        self.fills = 0
        self.rejects = 0
        # Subscribe immediately so orders published before run() starts are kept.
        # DROP_NEWEST: never silently discard older (already accepted) orders.
        self._sub: Subscription = bus.subscribe(
            Topic.ORDER, maxsize=10_000, policy=DropPolicy.DROP_NEWEST
        )

    def execute(self, order: Order) -> Fill | Rejection:
        """Fill or reject one order, publish the outcome, and return it."""
        tick = self._book.get(order.symbol)
        if tick is None:
            return self._reject(order, f"no market data for {order.symbol}")

        price = tick.ask if order.side is Side.BUY else tick.bid

        if order.order_type is OrderType.LIMIT and order.limit_price is not None:
            if order.side is Side.BUY and order.limit_price < price:
                return self._reject(
                    order, f"limit {order.limit_price} not marketable at ask {price}"
                )
            if order.side is Side.SELL and order.limit_price > price:
                return self._reject(
                    order, f"limit {order.limit_price} not marketable at bid {price}"
                )

        fee = order.qty * price * self.fee_bps / 10_000.0
        fill = Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            qty=order.qty,
            price=price,
            fee=fee,
        )
        self._portfolio.apply_fill(fill)
        self.fills += 1
        self._bus.publish(Topic.FILL, fill)
        return fill

    def _reject(self, order: Order, reason: str) -> Rejection:
        rejection = Rejection(order=order, reason=reason, source=self.SOURCE)
        self.rejects += 1
        self._bus.publish(Topic.REJECT, rejection)
        return rejection

    async def run(self) -> None:
        """Consume orders from the bus until stop() is called."""
        async for order in self._sub:
            self.execute(order)

    def stop(self) -> None:
        self._sub.close()
