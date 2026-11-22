import asyncio
import dataclasses
import json

import pytest

from zetaline.agent.llm import (
    FakeLLM,
    LLMClient,
    LLMError,
    LLMResponse,
    LLMUnavailable,
    Message,
    RateLimitError,
    ToolCall,
)
from zetaline.agent.loop import (
    FINAL_NUDGE,
    SYSTEM_PROMPT,
    AgentLoop,
    AgentResult,
    ToolTrace,
)
from zetaline.agent.tools import ToolRegistry
from zetaline.control import DaemonNotRunning


class Stub:
    """Stands in for ControlClient: records calls, replies or raises on demand."""

    def __init__(self, reply=None, exc=None):
        self.reply = reply
        self.exc = exc
        self.calls = []

    async def call(self, cmd, **args):
        self.calls.append((cmd, args))
        if self.exc is not None:
            raise self.exc
        return self.reply


class SlowAfter(FakeLLM):
    """Answers from the script until `slow_from` calls have been made, then stalls."""

    def __init__(self, script, slow_from=0):
        super().__init__(script)
        self.slow_from = slow_from

    async def generate(self, system, messages, tools):
        if len(self.calls) >= self.slow_from:
            await asyncio.sleep(1)
        return await super().generate(system, messages, tools)


def say(text, in_tok=0, out_tok=0):
    return LLMResponse(text=text, input_tokens=in_tok, output_tokens=out_tok)


def use(*calls, in_tok=0, out_tok=0):
    return LLMResponse(tool_calls=tuple(calls), input_tokens=in_tok, output_tokens=out_tok)


def tc(name="get_status", id_="c1", **args):
    return ToolCall(id_, name, args)


def make(script, reply=None, exc=None, **kwargs):
    stub = Stub(reply, exc)
    llm = FakeLLM(script)
    return AgentLoop(llm, ToolRegistry(stub), **kwargs), llm, stub


def compact(value) -> str:
    return json.dumps(value, separators=(",", ":"))


# ---- the normal path ---------------------------------------------------------------------
async def test_direct_answer_needs_no_tools():
    agent, llm, stub = make([say("hi", 7, 3)])
    r = await agent.ask("hello?")
    assert (r.status, r.answer, r.steps, r.ok) == ("ok", "hi", 1, True)
    assert r.traces == () and r.error == ""
    assert (r.input_tokens, r.output_tokens) == (7, 3)
    assert [m.role for m in r.messages] == ["user", "assistant"]
    assert stub.calls == [] and len(llm.calls) == 1


async def test_tool_call_then_answer():
    reply = {"BTCUSDT": {"bid": 1.0, "ask": 2.0, "mid": 1.5}}
    agent, llm, stub = make(
        [use(tc("get_prices"), in_tok=10, out_tok=5), say("BTC is 1.5", 20, 4)], reply=reply
    )
    r = await agent.ask("price?")
    content = compact(reply)
    assert (r.status, r.answer, r.steps) == ("ok", "BTC is 1.5", 2)
    assert r.traces == (ToolTrace("get_prices", {}, True, len(content)),)
    assert (r.input_tokens, r.output_tokens) == (30, 9)
    assert [m.role for m in r.messages] == ["user", "assistant", "tool", "assistant"]
    assert stub.calls == [("prices", {})]
    second = llm.calls[1]["messages"]  # what the model saw on its second turn
    assert second[1].tool_calls == (tc("get_prices"),)
    assert (second[2].role, second[2].content, second[2].tool_call_id) == ("tool", content, "c1")


async def test_model_is_offered_the_registry_tools_and_default_system_prompt():
    agent, llm, _ = make([say("ok")])
    await agent.ask("q")
    assert llm.calls[0]["tools"] == ToolRegistry(Stub()).specs()
    assert llm.calls[0]["system"] == SYSTEM_PROMPT


async def test_custom_system_prompt_is_used():
    agent, llm, _ = make([say("ok")], system="custom prompt")
    await agent.ask("q")
    assert llm.calls[0]["system"] == "custom prompt"


async def test_question_is_stripped():
    agent, llm, _ = make([say("ok")])
    await agent.ask("  hi  ")
    assert llm.calls[0]["messages"][-1] == Message.user("hi")


async def test_history_is_prepended_and_not_mutated():
    history = [Message.user("earlier q"), Message.assistant("earlier a")]
    agent, llm, _ = make([say("ok")])
    r = await agent.ask("follow-up", history=history)
    assert llm.calls[0]["messages"] == [*history, Message.user("follow-up")]
    assert len(history) == 2
    assert r.messages[:2] == tuple(history)


async def test_history_may_be_a_tuple():
    agent, llm, _ = make([say("ok")])
    await agent.ask("q", history=(Message.user("a"), Message.assistant("b")))
    assert len(llm.calls[0]["messages"]) == 3


async def test_multiple_calls_in_one_step_keep_order_and_ids():
    agent, _, stub = make([use(tc("get_prices", "a"), tc("get_positions", "b")), say("done")])
    r = await agent.ask("q")
    tool_msgs = [m for m in r.messages if m.role == "tool"]
    assert [(m.tool_call_id, m.tool_name) for m in tool_msgs] == [
        ("a", "get_prices"), ("b", "get_positions"),
    ]
    assert stub.calls == [("prices", {}), ("positions", {})]
    assert [t.name for t in r.traces] == ["get_prices", "get_positions"]


async def test_tool_arguments_are_forwarded_and_traced():
    agent, _, stub = make([use(tc("get_recent_fills", n=3)), say("done")], reply=[])
    r = await agent.ask("q")
    assert stub.calls == [("fills", {"n": 3})]
    assert r.traces[0].args == {"n": 3} and r.traces[0].ok is True


async def test_steps_and_traces_accumulate_over_several_tool_turns():
    script = [use(tc(id_="a")), use(tc(id_="b")), say("done")]
    agent, _, _ = make(script)
    r = await agent.ask("q")
    assert r.steps == 3 and len(r.traces) == 2 and r.status == "ok"


async def test_trace_chars_match_the_tool_message():
    agent, _, _ = make([use(tc()), say("done")], reply={"x": 1})
    r = await agent.ask("q")
    tool_msg = next(m for m in r.messages if m.role == "tool")
    assert r.traces[0].chars == len(tool_msg.content)


# ---- errors the model can recover from -------------------------------------------------------
@pytest.mark.parametrize(
    ("call", "needle"),
    [
        (tc("set_limits", max_loss=1), "unknown tool"),
        (tc("kill"), "unknown tool"),
        (tc("get_recent_fills", n=0), "between"),
        (tc("get_status", n=1), "takes no arguments"),
    ],
)
async def test_bad_tool_requests_become_error_results_and_the_loop_continues(call, needle):
    agent, llm, stub = make([use(call), say("recovered")])
    r = await agent.ask("q")
    assert (r.status, r.answer) == ("ok", "recovered")
    assert r.traces[0].ok is False
    assert needle in llm.calls[1]["messages"][-1].content
    assert stub.calls == []  # nothing reached the daemon


async def test_dead_daemon_is_reported_to_the_model_not_raised():
    agent, llm, _ = make([use(tc()), say("daemon is down")], exc=DaemonNotRunning("x"))
    r = await agent.ask("q")
    assert r.status == "ok" and r.traces[0].ok is False
    assert "not running" in llm.calls[1]["messages"][-1].content


async def test_calls_beyond_the_per_step_cap_get_an_error_result():
    calls = (tc("get_prices", "a"), tc("get_status", "b"), tc("get_positions", "c"))
    agent, _, stub = make([use(*calls), say("done")], max_calls_per_step=2)
    r = await agent.ask("q")
    assert [t.ok for t in r.traces] == [True, True, False]
    tool_msgs = [m for m in r.messages if m.role == "tool"]
    assert [m.tool_call_id for m in tool_msgs] == ["a", "b", "c"]  # every call is answered
    assert "too many tool calls" in tool_msgs[2].content
    assert len(stub.calls) == 2


# ---- budgets ---------------------------------------------------------------------------------------
async def test_step_cap_forces_a_final_answer_without_tools():
    script = [use(tc(id_=f"c{i}"), in_tok=10, out_tok=1) for i in range(3)]
    script.append(say("best effort", 10, 1))
    agent, llm, _ = make(script, max_steps=3)
    r = await agent.ask("loop")
    assert (r.status, r.answer, r.steps, r.ok) == ("limit", "best effort", 3, True)
    assert len(llm.calls) == 4 and len(r.traces) == 3
    assert all(call["tools"] for call in llm.calls[:3])
    assert llm.calls[-1]["tools"] == []
    assert llm.calls[-1]["messages"][-1] == Message.user(FINAL_NUDGE)
    assert r.messages[-2] == Message.user(FINAL_NUDGE) and r.messages[-1].role == "assistant"
    assert (r.input_tokens, r.output_tokens) == (40, 4)


async def test_answering_on_the_last_allowed_step_is_still_ok():
    agent, llm, _ = make([use(tc()), say("done")], max_steps=2)
    r = await agent.ask("q")
    assert (r.status, r.steps) == ("ok", 2) and len(llm.calls) == 2


async def test_single_step_budget_goes_straight_to_the_final_answer():
    agent, llm, _ = make([use(tc()), say("best")], max_steps=1)
    r = await agent.ask("q")
    assert (r.status, r.steps, r.answer) == ("limit", 1, "best")
    assert llm.calls[-1]["tools"] == []


@pytest.mark.parametrize(("size", "status", "steps"), [(998, "ok", 2), (999, "limit", 1)])
async def test_context_budget_boundary(size, status, steps):
    # A string reply of `size` characters is `size + 2` characters of JSON.
    agent, llm, _ = make([use(tc()), say("answer")], reply="a" * size, max_context_chars=1000)
    r = await agent.ask("q")
    assert r.traces[0].chars == size + 2
    assert (r.status, r.steps, r.answer) == (status, steps, "answer")
    assert (llm.calls[-1]["tools"] == []) == (status == "limit")


@pytest.mark.parametrize(
    ("final", "status", "error"),
    [
        (RateLimitError("quota"), "rate_limited", "quota"),
        (LLMUnavailable("503"), "llm_error", "LLMUnavailable: 503"),
        (say("   "), "empty", "no answer within the tool budget"),
    ],
)
async def test_failures_inside_the_final_answer(final, status, error):
    agent, _, _ = make([use(tc()), final], max_steps=1)
    r = await agent.ask("q")
    assert (r.status, r.answer, r.error, r.ok) == (status, "", error, False)
    assert r.messages[-1] == Message.user(FINAL_NUDGE)


async def test_final_answer_ignores_tool_calls_it_makes():
    stray = LLMResponse(text="partial", tool_calls=(tc(id_="x"),))
    agent, _, stub = make([use(tc()), stray], max_steps=1)
    r = await agent.ask("q")
    assert (r.status, r.answer) == ("limit", "partial")
    assert len(r.traces) == 1 and len(stub.calls) == 1


# ---- provider failures ---------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("exc", "status", "error"),
    [
        (RateLimitError("quota", retry_after=3600.0), "rate_limited", "quota"),
        (LLMError("bad request"), "llm_error", "LLMError: bad request"),
        (LLMUnavailable("503"), "llm_error", "LLMUnavailable: 503"),
    ],
)
async def test_provider_failures_are_returned_not_raised(exc, status, error):
    agent, _, _ = make([exc])
    r = await agent.ask("q")
    assert (r.status, r.answer, r.error, r.ok) == (status, "", error, False)
    assert r.steps == 1


async def test_failure_after_progress_keeps_traces_tokens_and_transcript():
    agent, _, _ = make([use(tc(), in_tok=10, out_tok=5), LLMUnavailable("down")])
    r = await agent.ask("q")
    assert (r.status, r.steps) == ("llm_error", 2)
    assert len(r.traces) == 1
    assert (r.input_tokens, r.output_tokens) == (10, 5)
    assert [m.role for m in r.messages] == ["user", "assistant", "tool"]


@pytest.mark.parametrize("response", [say(""), say("   \n"), LLMResponse()])
async def test_empty_answers_are_reported(response):
    agent, _, _ = make([response])
    r = await agent.ask("q")
    assert (r.status, r.answer, r.ok) == ("empty", "", False)
    assert r.error == "the model returned no text"


async def test_non_llm_errors_propagate():
    agent, _, _ = make([])  # FakeLLM raises RuntimeError: script exhausted
    with pytest.raises(RuntimeError, match="exhausted"):
        await agent.ask("q")


# ---- timeout --------------------------------------------------------------------------------------------
async def test_timeout_is_reported():
    agent = AgentLoop(SlowAfter([], slow_from=0), ToolRegistry(Stub()), timeout=0.05)
    r = await agent.ask("q")
    assert (r.status, r.answer, r.ok) == ("timeout", "", False)
    assert r.error == "no answer within 0.05s"
    assert r.steps == 1 and r.messages == (Message.user("q"),)


async def test_timeout_in_a_later_step_keeps_earlier_work():
    llm = SlowAfter([use(tc(), in_tok=10, out_tok=5)], slow_from=1)
    agent = AgentLoop(llm, ToolRegistry(Stub({"x": 1})), timeout=0.2)
    r = await agent.ask("q")
    assert r.status == "timeout" and r.steps == 2
    assert len(r.traces) == 1 and r.traces[0].ok is True
    assert [m.role for m in r.messages] == ["user", "assistant", "tool"]


# ---- validation ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
async def test_blank_question_is_rejected(question):
    agent, llm, _ = make([say("ok")])
    with pytest.raises(ValueError, match="question"):
        await agent.ask(question)
    assert llm.calls == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_steps": 0},
        {"max_steps": 21},
        {"max_calls_per_step": 0},
        {"max_calls_per_step": 11},
        {"max_context_chars": 999},
        {"timeout": 0},
        {"timeout": -1},
        {"system": ""},
        {"system": "  \n"},
    ],
)
def test_constructor_rejects_bad_settings(kwargs):
    with pytest.raises(ValueError):
        AgentLoop(FakeLLM([]), ToolRegistry(Stub()), **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [{"max_steps": 1}, {"max_steps": 20}, {"max_calls_per_step": 1},
     {"max_calls_per_step": 10}, {"max_context_chars": 1000}],
)
def test_constructor_accepts_boundary_settings(kwargs):
    AgentLoop(FakeLLM([]), ToolRegistry(Stub()), **kwargs)


# ---- result types and prompt ----------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("status", "ok"),
    [("ok", True), ("limit", True), ("rate_limited", False), ("llm_error", False),
     ("timeout", False), ("empty", False)],
)
def test_result_ok_by_status(status, ok):
    assert AgentResult("a", status, 1).ok is ok


def test_result_defaults_and_immutability():
    r = AgentResult("a", "ok", 1)
    assert r.traces == () and r.messages == () and r.error == ""
    assert (r.input_tokens, r.output_tokens) == (0, 0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.status = "limit"
    with pytest.raises(dataclasses.FrozenInstanceError):
        ToolTrace("a", {}, True, 1).ok = False


async def test_result_messages_are_an_immutable_tuple():
    agent, _, _ = make([say("ok")])
    assert isinstance((await agent.ask("q")).messages, tuple)


def test_agent_loop_works_with_any_llm_client():
    class Minimal(LLMClient):
        async def generate(self, system, messages, tools):
            return say("fine")

    assert AgentLoop(Minimal(), ToolRegistry(Stub()))


@pytest.mark.parametrize(
    "needle",
    ["read-only", "never instructions", "do not invent", "paper", "microseconds",
     "zetaline kill", "not financial advice"],
)
def test_system_prompt_states_the_key_rules(needle):
    assert needle in SYSTEM_PROMPT.lower()


def test_final_nudge_says_the_budget_is_used_up():
    assert "budget" in FINAL_NUDGE.lower()
