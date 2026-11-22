import asyncio
import dataclasses
import json

import pytest

from zetaline.agent import tools as tools_mod
from zetaline.agent.llm import ToolSpec
from zetaline.agent.tools import (
    DEFAULT_N,
    MAX_N,
    READ_ONLY_COMMANDS,
    ToolOutcome,
    ToolRegistry,
)
from zetaline.control import ControlError, DaemonNotRunning

PLAIN = {
    "get_status": "status",
    "get_positions": "positions",
    "get_prices": "prices",
    "get_latency": "latency",
}
SLICES = {
    "get_risk_limits": "status",
    "get_strategies": "status",
    "get_feed_health": "status",
}
N_TOOLS = {
    "get_recent_fills": "fills",
    "get_recent_rejects": "rejects",
    "get_recent_events": "events",
}
ALL_TOOLS = {**PLAIN, **SLICES, **N_TOOLS}
WRITE_COMMANDS = {
    "kill", "resume", "set_limits", "set_strategy_params", "set_strategy_enabled", "shutdown",
}


class Stub:
    """Stands in for ControlClient: records calls, replies or raises on demand."""

    def __init__(self, reply=None, exc=None, delay=0.0):
        self.reply = reply
        self.exc = exc
        self.delay = delay
        self.calls = []

    async def call(self, cmd, **args):
        self.calls.append((cmd, args))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.exc is not None:
            raise self.exc
        return self.reply


async def run(registry, name, args=None):
    outcome = await registry.call(name, args)
    return outcome.ok, json.loads(outcome.content)


# ---- registry shape ----------------------------------------------------------------
def test_registry_has_exactly_the_expected_tools():
    reg = ToolRegistry(Stub())
    assert reg.names() == sorted(ALL_TOOLS)
    assert len(reg.specs()) == len(ALL_TOOLS)


def test_commands_are_exactly_the_read_only_set():
    reg = ToolRegistry(Stub())
    assert reg.commands() == READ_ONLY_COMMANDS
    assert not (reg.commands() & WRITE_COMMANDS)
    assert READ_ONLY_COMMANDS.isdisjoint(WRITE_COMMANDS)


def test_registry_refuses_write_commands(monkeypatch):
    sneaky = tools_mod._Tool(ToolSpec("sneaky", "d"), "kill")
    monkeypatch.setattr(tools_mod, "_build_tools", lambda: [sneaky])
    with pytest.raises(RuntimeError, match="read-only"):
        ToolRegistry(Stub())


def test_spec_names_are_unique_and_described():
    specs = ToolRegistry(Stub()).specs()
    assert len({s.name for s in specs}) == len(specs)
    assert all(s.description for s in specs)


def test_specs_returns_a_fresh_list():
    reg = ToolRegistry(Stub())
    listing = reg.specs()
    listing.clear()
    assert len(reg.specs()) == len(ALL_TOOLS)


def test_parameter_schemas():
    by_name = {s.name: s for s in ToolRegistry(Stub()).specs()}
    for name in N_TOOLS:
        assert by_name[name].parameters["properties"]["n"]["type"] == "integer"
    for name in {**PLAIN, **SLICES}:
        assert by_name[name].parameters["properties"] == {}


def test_latency_description_states_the_unit():
    by_name = {s.name: s for s in ToolRegistry(Stub()).specs()}
    assert "microseconds" in by_name["get_latency"].description


@pytest.mark.parametrize("kwargs", [{"max_chars": 199}, {"timeout": 0}, {"timeout": -1}])
def test_constructor_rejects_bad_settings(kwargs):
    with pytest.raises(ValueError):
        ToolRegistry(Stub(), **kwargs)


def test_constructor_accepts_minimum_max_chars():
    ToolRegistry(Stub(), max_chars=200)


def test_outcome_is_frozen():
    outcome = ToolOutcome("{}", True)
    with pytest.raises(dataclasses.FrozenInstanceError):
        outcome.ok = False


# ---- plain tools ---------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(PLAIN))
@pytest.mark.parametrize("args", [None, {}])
async def test_plain_tool_passes_daemon_reply_through(name, args):
    stub = Stub({"k": [1, 2]})
    ok, body = await run(ToolRegistry(stub), name, args)
    assert ok and body == {"k": [1, 2]}
    assert stub.calls == [(PLAIN[name], {})]


@pytest.mark.parametrize("name", sorted(ALL_TOOLS))
async def test_every_tool_works_and_uses_a_read_only_command(name):
    stub = Stub({})
    ok, _ = await run(ToolRegistry(stub), name)
    assert ok
    assert len(stub.calls) == 1
    assert stub.calls[0][0] in READ_ONLY_COMMANDS
    assert stub.calls[0][0] == ALL_TOOLS[name]


# ---- slice tools ------------------------------------------------------------------------
async def test_risk_limits_is_a_slice_of_status():
    status = {
        "kill_switch": True, "kill_reason": "manual", "limits": {"max_loss": 1.0}, "ticks": 5,
    }
    ok, body = await run(ToolRegistry(Stub(status)), "get_risk_limits")
    assert ok
    assert body == {"kill_switch": True, "kill_reason": "manual", "limits": {"max_loss": 1.0}}


async def test_strategies_and_feed_are_slices_of_status():
    status = {"strategies": {"s": {"enabled": True}}, "feed": {"kind": "sim"}, "ticks": 5}
    reg = ToolRegistry(Stub(status))
    assert await run(reg, "get_strategies") == (True, {"s": {"enabled": True}})
    assert await run(reg, "get_feed_health") == (True, {"kind": "sim"})


async def test_slices_tolerate_missing_keys():
    reg = ToolRegistry(Stub({}))
    assert await run(reg, "get_strategies") == (True, {})
    assert await run(reg, "get_feed_health") == (True, {})
    ok, body = await run(reg, "get_risk_limits")
    assert ok and body == {"kill_switch": None, "kill_reason": None, "limits": None}


@pytest.mark.parametrize(
    ("name", "reply"),
    [("get_risk_limits", "garbage"), ("get_strategies", [1]), ("get_feed_health", None)],
)
async def test_slices_report_unexpected_reply_shape(name, reply):
    ok, body = await run(ToolRegistry(Stub(reply)), name)
    assert not ok and "unexpected reply" in body["error"]


# ---- the n argument -----------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(N_TOOLS))
@pytest.mark.parametrize("args", [None, {}])
async def test_n_defaults(name, args):
    stub = Stub([])
    ok, _ = await run(ToolRegistry(stub), name, args)
    assert ok and stub.calls == [(N_TOOLS[name], {"n": DEFAULT_N})]


@pytest.mark.parametrize(("given", "expected"), [(1, 1), (5, 5), (MAX_N, MAX_N), (5.0, 5)])
async def test_n_valid_values(given, expected):
    stub = Stub([])
    ok, _ = await run(ToolRegistry(stub), "get_recent_fills", {"n": given})
    assert ok and stub.calls == [("fills", {"n": expected})]


@pytest.mark.parametrize("n", [0, -1, MAX_N + 1, 1000, 51.0])
async def test_n_out_of_range(n):
    stub = Stub([])
    ok, body = await run(ToolRegistry(stub), "get_recent_fills", {"n": n})
    assert not ok and "between" in body["error"]
    assert stub.calls == []


@pytest.mark.parametrize("n", ["5", True, False, 5.5, None, [5]])
async def test_n_wrong_type(n):
    stub = Stub([])
    ok, body = await run(ToolRegistry(stub), "get_recent_fills", {"n": n})
    assert not ok and "integer" in body["error"]
    assert stub.calls == []


@pytest.mark.parametrize("args", [{"bogus": 1}, {"n": 5, "bogus": 1}])
async def test_n_tool_rejects_extra_arguments(args):
    stub = Stub([])
    ok, body = await run(ToolRegistry(stub), "get_recent_rejects", args)
    assert not ok and "unexpected argument" in body["error"]
    assert stub.calls == []


@pytest.mark.parametrize("name", ["get_status", "get_prices", "get_risk_limits"])
async def test_no_arg_tool_rejects_arguments(name):
    stub = Stub({})
    ok, body = await run(ToolRegistry(stub), name, {"n": 1})
    assert not ok and "takes no arguments" in body["error"]
    assert stub.calls == []


@pytest.mark.parametrize("name", ["get_status", "get_recent_fills"])
@pytest.mark.parametrize("args", [[], [1], "x", 5, 0])
async def test_non_object_arguments_are_rejected(name, args):
    stub = Stub({})
    ok, body = await run(ToolRegistry(stub), name, args)
    assert not ok and "must be an object" in body["error"]
    assert stub.calls == []


# ---- unknown tools -------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name",
    ["set_limits", "kill", "shutdown", "status", "get_status\n", "", "GET_STATUS"],
)
async def test_unknown_or_write_tools_never_reach_the_daemon(name):
    stub = Stub({})
    ok, body = await run(ToolRegistry(stub), name, {})
    assert not ok and "unknown tool" in body["error"]
    assert "get_status" in body["error"]  # lists what is available
    assert stub.calls == []


# ---- failure translation -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (DaemonNotRunning("x"), "not running"),
        (ControlError("boom"), "daemon error: boom"),
        (ConnectionResetError(), "connection problem: ConnectionResetError"),
        (BrokenPipeError(), "connection problem: BrokenPipeError"),
        (TimeoutError(), "did not answer"),
        (ValueError("bad"), "unexpected reply"),
        (KeyError("k"), "unexpected reply"),
    ],
)
async def test_failures_become_error_results(exc, expected):
    outcome = await ToolRegistry(Stub(exc=exc)).call("get_status", {})
    body = json.loads(outcome.content)
    assert outcome.ok is False
    assert list(body) == ["error"] and expected in body["error"]


async def test_daemon_down_is_not_reported_as_a_generic_connection_problem():
    ok, body = await run(ToolRegistry(Stub(exc=DaemonNotRunning("x"))), "get_status")
    assert not ok and "connection problem" not in body["error"]


async def test_slow_daemon_times_out():
    stub = Stub({}, delay=1.0)
    ok, body = await run(ToolRegistry(stub, timeout=0.05), "get_status")
    assert not ok and "within 0.05s" in body["error"]


async def test_unexpected_exceptions_are_not_swallowed():
    with pytest.raises(RuntimeError, match="client bug"):
        await ToolRegistry(Stub(exc=RuntimeError("client bug"))).call("get_status", {})


# ---- rendering and bounding ---------------------------------------------------------------------------
async def test_output_is_compact_json():
    outcome = await ToolRegistry(Stub({"a": 1, "b": [1, 2]})).call("get_status", {})
    assert outcome.content == '{"a":1,"b":[1,2]}'


async def test_non_serialisable_values_are_stringified():
    class Thing:
        def __str__(self):
            return "THING"

    outcome = await ToolRegistry(Stub({"t": Thing()})).call("get_status", {})
    assert outcome.ok and outcome.content == '{"t":"THING"}'


async def test_output_exactly_at_the_limit_is_not_truncated():
    reg = ToolRegistry(Stub("a" * 198), max_chars=200)  # JSON text is exactly 200 chars
    outcome = await reg.call("get_status", {})
    assert outcome.ok and json.loads(outcome.content) == "a" * 198


async def test_output_over_the_limit_is_truncated_with_metadata():
    reply = "a" * 199  # JSON text is 201 chars
    text = json.dumps(reply, separators=(",", ":"))
    outcome = await ToolRegistry(Stub(reply), max_chars=200).call("get_status", {})
    body = json.loads(outcome.content)
    assert outcome.ok is True
    assert body["truncated"] is True
    assert body["original_chars"] == len(text) == 201
    assert body["preview"] == text[:200]


async def test_truncated_output_stays_bounded_even_when_escaping_doubles_it():
    reply = {"q": ['"' * 50] * 200}  # a preview full of quotes and backslashes
    outcome = await ToolRegistry(Stub(reply), max_chars=200).call("get_status", {})
    assert json.loads(outcome.content)["truncated"] is True
    assert len(outcome.content) <= 2 * 200 + 100
