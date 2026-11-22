import dataclasses

import pytest

from zetaline.agent.llm import (
    FakeLLM,
    LLMClient,
    LLMError,
    LLMResponse,
    LLMUnavailable,
    Message,
    RateLimitError,
    RetryingClient,
    ToolCall,
    ToolSpec,
)

OK = LLMResponse(text="ok")


class Sleeps:
    """An injectable sleep that records the requested delays instead of waiting."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


def call(name: str = "get_status", id_: str = "c1", **args) -> ToolCall:
    return ToolCall(id_, name, args)


# ---- ToolSpec -----------------------------------------------------------------
@pytest.mark.parametrize("name", ["a", "_x", "get_status", "A1_b2", "x" * 64])
def test_toolspec_accepts_valid_names(name):
    assert ToolSpec(name, "desc").name == name


@pytest.mark.parametrize("name", ["", "1abc", "bad name", "a-b", "x" * 65, "abc\n", "é"])
def test_toolspec_rejects_invalid_names(name):
    with pytest.raises(ValueError, match="invalid tool name"):
        ToolSpec(name, "desc")


def test_toolspec_requires_description():
    with pytest.raises(ValueError, match="description"):
        ToolSpec("ok_name", "")


def test_toolspec_default_parameters_are_not_shared():
    a, b = ToolSpec("a", "d"), ToolSpec("b", "d")
    assert a.parameters == {"type": "object", "properties": {}}
    assert a.parameters is not b.parameters


def test_toolspec_is_frozen():
    spec = ToolSpec("a", "d")
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.name = "b"


# ---- Message ------------------------------------------------------------------
def test_message_helpers():
    assert Message.user("hi") == Message("user", "hi")
    assistant = Message.assistant("x", [call()])
    assert assistant.role == "assistant" and assistant.tool_calls == (call(),)
    result = Message.tool_result(call("get_prices", "c9"), "42")
    assert (result.role, result.content) == ("tool", "42")
    assert (result.tool_call_id, result.tool_name) == ("c9", "get_prices")


def test_message_rejects_unknown_role():
    with pytest.raises(ValueError, match="role"):
        Message("system", "x")


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"tool_call_id": "c"}, {"tool_name": "n"}],
)
def test_tool_message_needs_both_ids(kwargs):
    with pytest.raises(ValueError, match="tool_call_id"):
        Message("tool", "x", **kwargs)


@pytest.mark.parametrize("role", ["user", "tool"])
def test_only_assistant_messages_carry_tool_calls(role):
    extra = {"tool_call_id": "c", "tool_name": "n"} if role == "tool" else {}
    with pytest.raises(ValueError, match="only assistant"):
        Message(role, "x", tool_calls=(call(),), **extra)


def test_message_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        Message.user("x").content = "y"


# ---- LLMResponse and errors -------------------------------------------------------
def test_response_wants_tools_and_defaults():
    assert LLMResponse(tool_calls=(call(),)).wants_tools is True
    plain = LLMResponse(text="x")
    assert plain.wants_tools is False
    assert plain.input_tokens == 0 and plain.output_tokens == 0


def test_error_hierarchy():
    assert issubclass(RateLimitError, LLMError)
    assert issubclass(LLMUnavailable, LLMError)
    assert RateLimitError("x").retry_after is None
    assert RateLimitError("x", retry_after=2.5).retry_after == 2.5


# ---- FakeLLM ------------------------------------------------------------------------
async def test_fake_replays_script_in_order_and_records_calls():
    llm = FakeLLM([LLMResponse(text="one"), LLMResponse(text="two")])
    spec = ToolSpec("t", "d")
    first = await llm.generate("sys", [Message.user("q")], [spec])
    second = await llm.generate("sys2", [], [])
    assert (first.text, second.text) == ("one", "two")
    assert llm.calls[0]["system"] == "sys" and llm.calls[0]["tools"] == [spec]
    assert llm.calls[1]["system"] == "sys2"


async def test_fake_records_a_snapshot_of_messages():
    msgs = [Message.user("a")]
    llm = FakeLLM([OK])
    await llm.generate("s", msgs, [])
    msgs.append(Message.user("b"))
    assert len(llm.calls[0]["messages"]) == 1


async def test_fake_raises_exception_items_and_reports_exhaustion():
    llm = FakeLLM([LLMUnavailable("down")])
    with pytest.raises(LLMUnavailable):
        await llm.generate("s", [], [])
    with pytest.raises(RuntimeError, match="exhausted"):
        await llm.generate("s", [], [])


# ---- RetryingClient -------------------------------------------------------------------
def test_retrying_client_is_an_llm_client():
    assert isinstance(RetryingClient(FakeLLM([])), LLMClient)


async def test_success_needs_no_retry():
    inner, sleeps = FakeLLM([OK]), Sleeps()
    client = RetryingClient(inner, sleep=sleeps)
    assert await client.generate("s", [], []) is OK
    assert client.retries == 0 and sleeps.delays == [] and len(inner.calls) == 1


async def test_retry_after_is_honoured_exactly():
    inner, sleeps = FakeLLM([RateLimitError("429", retry_after=5.0), OK]), Sleeps()
    client = RetryingClient(inner, sleep=sleeps)
    assert await client.generate("s", [], []) is OK
    assert sleeps.delays == [5.0] and client.retries == 1


async def test_retry_after_equal_to_max_delay_is_allowed():
    inner = FakeLLM([RateLimitError("429", retry_after=60.0), OK])
    sleeps = Sleeps()
    await RetryingClient(inner, max_delay=60.0, sleep=sleeps).generate("s", [], [])
    assert sleeps.delays == [60.0]


async def test_exponential_backoff_with_jitter_bounds():
    inner = FakeLLM([LLMUnavailable("a")] * 3 + [OK])
    sleeps = Sleeps()
    client = RetryingClient(inner, max_retries=3, base_delay=2.0, max_delay=60.0, sleep=sleeps)
    await client.generate("s", [], [])
    d0, d1, d2 = sleeps.delays
    assert 1.0 <= d0 <= 2.0 and 2.0 <= d1 <= 4.0 and 4.0 <= d2 <= 8.0


async def test_rate_limit_without_retry_after_uses_backoff():
    inner = FakeLLM([RateLimitError("429"), OK])
    sleeps = Sleeps()
    await RetryingClient(inner, base_delay=2.0, sleep=sleeps).generate("s", [], [])
    assert 1.0 <= sleeps.delays[0] <= 2.0


def test_backoff_is_capped_at_max_delay():
    client = RetryingClient(FakeLLM([]), base_delay=10.0, max_delay=12.0)
    for attempt in (3, 10, 50):
        for _ in range(30):
            delay = client._delay(LLMUnavailable("x"), attempt)
            assert 6.0 <= delay <= 12.0


async def test_gives_up_after_max_retries():
    inner, sleeps = FakeLLM([LLMUnavailable("a")] * 3), Sleeps()
    client = RetryingClient(inner, max_retries=2, sleep=sleeps)
    with pytest.raises(LLMUnavailable):
        await client.generate("s", [], [])
    assert len(inner.calls) == 3 and client.retries == 2 and len(sleeps.delays) == 2


async def test_zero_retries_means_a_single_attempt():
    inner, sleeps = FakeLLM([LLMUnavailable("a"), OK]), Sleeps()
    with pytest.raises(LLMUnavailable):
        await RetryingClient(inner, max_retries=0, sleep=sleeps).generate("s", [], [])
    assert len(inner.calls) == 1 and sleeps.delays == []


async def test_long_retry_after_raises_at_once_without_sleeping():
    inner, sleeps = FakeLLM([RateLimitError("daily quota", retry_after=3600.0), OK]), Sleeps()
    with pytest.raises(RateLimitError):
        await RetryingClient(inner, max_delay=60.0, sleep=sleeps).generate("s", [], [])
    assert len(inner.calls) == 1 and sleeps.delays == []


async def test_other_llm_errors_are_not_retried():
    inner, sleeps = FakeLLM([LLMError("bad request"), OK]), Sleeps()
    with pytest.raises(LLMError, match="bad request"):
        await RetryingClient(inner, sleep=sleeps).generate("s", [], [])
    assert len(inner.calls) == 1 and sleeps.delays == []


async def test_non_llm_exceptions_propagate_untouched():
    inner = FakeLLM([ValueError("bug")])
    with pytest.raises(ValueError, match="bug"):
        await RetryingClient(inner, sleep=Sleeps()).generate("s", [], [])


async def test_retries_accumulate_across_calls():
    inner = FakeLLM([LLMUnavailable("a"), OK, LLMUnavailable("b"), OK])
    client = RetryingClient(inner, sleep=Sleeps())
    await client.generate("s", [], [])
    await client.generate("s", [], [])
    assert client.retries == 2


@pytest.mark.parametrize(
    "kwargs",
    [{"max_retries": -1}, {"base_delay": 0}, {"base_delay": 5.0, "max_delay": 1.0}],
)
def test_retrying_client_rejects_bad_settings(kwargs):
    with pytest.raises(ValueError):
        RetryingClient(FakeLLM([]), **kwargs)
