import asyncio

import pytest

from zetaline.core.events import DropPolicy, EventBus, Topic


async def test_publish_fans_out_to_all_subscribers():
    bus = EventBus()
    a = bus.subscribe(Topic.TICK)
    b = bus.subscribe(Topic.TICK)
    assert bus.publish(Topic.TICK, "x") == 2
    assert await a.get() == "x"
    assert await b.get() == "x"


async def test_topics_are_isolated():
    bus = EventBus()
    tick_sub = bus.subscribe(Topic.TICK)
    fill_sub = bus.subscribe(Topic.FILL)
    bus.publish(Topic.TICK, 1)
    assert tick_sub.qsize() == 1
    assert fill_sub.qsize() == 0


async def test_publish_with_no_subscribers_is_safe():
    bus = EventBus()
    assert bus.publish(Topic.TICK, 1) == 0
    assert bus.stats()["published"]["tick"] == 1


async def test_drop_oldest_keeps_freshest():
    bus = EventBus()
    sub = bus.subscribe(Topic.TICK, maxsize=2, policy=DropPolicy.DROP_OLDEST)
    for i in range(5):
        bus.publish(Topic.TICK, i)
    assert sub.dropped == 3
    assert [await sub.get(), await sub.get()] == [3, 4]


async def test_drop_newest_keeps_history():
    bus = EventBus()
    sub = bus.subscribe(Topic.TICK, maxsize=2, policy=DropPolicy.DROP_NEWEST)
    accepted = [bus.publish(Topic.TICK, i) for i in range(5)]
    assert accepted == [1, 1, 0, 0, 0]
    assert sub.dropped == 3
    assert [await sub.get(), await sub.get()] == [0, 1]


async def test_slow_subscriber_does_not_block_fast_one():
    bus = EventBus()
    slow = bus.subscribe(Topic.TICK, maxsize=1)
    fast = bus.subscribe(Topic.TICK, maxsize=100)
    for i in range(50):
        bus.publish(Topic.TICK, i)
    assert fast.qsize() == 50 and fast.dropped == 0
    assert slow.qsize() == 1 and slow.dropped == 49


async def test_async_iteration_ends_when_closed():
    bus = EventBus()
    sub = bus.subscribe(Topic.TICK)
    bus.publish(Topic.TICK, 1)
    bus.publish(Topic.TICK, 2)

    async def consume():
        return [e async for e in sub]

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)
    sub.close()
    assert await asyncio.wait_for(task, timeout=1) == [1, 2]
    assert bus.subscriber_count(Topic.TICK) == 0


async def test_get_raises_eof_after_close():
    bus = EventBus()
    sub = bus.subscribe(Topic.TICK)
    sub.close()
    with pytest.raises(EOFError):
        await sub.get()


async def test_close_is_idempotent_and_works_on_full_queue():
    bus = EventBus()
    sub = bus.subscribe(Topic.TICK, maxsize=1)
    bus.publish(Topic.TICK, 1)
    sub.close()
    sub.close()
    assert bus.subscriber_count(Topic.TICK) == 0
