import json
from types import SimpleNamespace as NS

import pytest
from google.genai import errors, types

from zetaline.agent import gemini as gem_mod
from zetaline.agent.config import AgentConfig
from zetaline.agent.gemini import GeminiClient, parse_retry_delay, translate_error
from zetaline.agent.llm import (
    LLMError,
    LLMUnavailable,
    Message,
    RateLimitError,
    RetryingClient,
    ToolCall,
    ToolSpec,
)
from zetaline.agent.loop import AgentLoop
from zetaline.agent.tools import ToolRegistry

SECRET = "AIzaFAKE-key_123"
CFG = AgentConfig(SECRET, "gemini-test")
NO_PARAMS = ToolSpec("get_prices", "Latest prices.", {"type": "object", "properties": {}})
WITH_PARAMS = ToolSpec(
    "get_recent_fills", "Fills.", {"type": "object", "properties": {"n": {"type": "integer"}}}
)


class FakeModels:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    async def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def fake(outcomes):
    models = FakeModels(outcomes)
    return models, NS(aio=NS(models=models))


def reply(parts, usage=None):
    return NS(candidates=[NS(content=NS(parts=parts))], usage_metadata=usage)


def text_part(text, thought=False):
    return types.Part(text=text, thought=thought) if thought else types.Part(text=text)


def call_part(name="get_prices", args=None, id_="srv-1", signature=b"sig"):
    return types.Part(
        function_call=types.FunctionCall(id=id_, name=name, args=args or {}),
        thought_signature=signature,
    )


def api_error(cls, code, message="msg", details=None):
    body = {"error": {"code": code, "message": message, "status": "STATUS"}}
    if details:
        body["error"]["details"] = details
    return cls(code, body)


def client_with(outcomes):
    models, client = fake(outcomes)
    return models, GeminiClient(CFG, client=client)


async def generate(g, messages=None, tools=()):
    return await g.generate("sys", messages or [Message.user("q")], list(tools))


class Stub:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    async def call(self, cmd, **args):
        self.calls.append((cmd, args))
        return self.reply


# ---- constructor -------------------------------------------------------------------------
@pytest.mark.parametrize("tokens", [1, 4096, 65_536])
def test_valid_max_output_tokens(tokens):
    GeminiClient(CFG, client=NS(), max_output_tokens=tokens)


@pytest.mark.parametrize("tokens", [0, -1, 65_537])
def test_invalid_max_output_tokens(tokens):
    with pytest.raises(ValueError, match="max_output_tokens"):
        GeminiClient(CFG, client=NS(), max_output_tokens=tokens)


def test_real_client_is_built_lazily_without_network():
    assert GeminiClient(CFG)._client is not None


# ---- neutral messages -> Gemini contents ---------------------------------------------------
def make_client():
    return GeminiClient(CFG, client=NS())


def test_user_message_becomes_a_user_content():
    contents = make_client().to_contents([Message.user("hi")])
    assert [(c.role, c.parts[0].text) for c in contents] == [("user", "hi")]


def test_assistant_text_uses_the_model_role():
    contents = make_client().to_contents([Message.user("q"), Message.assistant("a")])
    assert [c.role for c in contents] == ["user", "model"]


def test_empty_messages_are_skipped():
    g = make_client()
    assert g.to_contents([Message.user("")]) == []
    assert [c.role for c in g.to_contents([Message.user("q"), Message.assistant("")])] == ["user"]


def test_adjacent_user_messages_merge_into_one_turn():
    contents = make_client().to_contents([Message.user("a"), Message.user("b")])
    assert len(contents) == 1 and [p.text for p in contents[0].parts] == ["a", "b"]


def test_assistant_call_without_cache_builds_a_function_call():
    call = ToolCall("zl-x", "get_recent_fills", {"n": 3})
    contents = make_client().to_contents([Message.user("q"), Message.assistant("", [call])])
    fc = contents[1].parts[0].function_call
    assert (fc.name, fc.args, fc.id) == ("get_recent_fills", {"n": 3}, None)


def test_assistant_text_and_call_share_one_turn():
    call = ToolCall("zl-x", "get_prices", {})
    contents = make_client().to_contents([Message.assistant("looking", [call])])
    assert len(contents) == 1 and len(contents[0].parts) == 2
    assert contents[0].parts[0].text == "looking"


async def test_cached_part_is_replayed_with_its_signature():
    g = make_client()
    g._client = fake([reply([call_part()])])[1]
    response = await generate(g, tools=[NO_PARAMS])
    call = response.tool_calls[0]
    contents = g.to_contents([Message.user("q"), Message.assistant("", [call])])
    part = contents[1].parts[0]
    assert part.thought_signature == b"sig" and part.function_call.id == "srv-1"


def test_tool_result_dict_is_wrapped_as_result():
    call = ToolCall("zl-x", "get_prices", {})
    msgs = [Message.tool_result(call, json.dumps({"BTC": 1}))]
    fr = make_client().to_contents(msgs)[0].parts[0].function_response
    assert (fr.name, fr.response, fr.id) == ("get_prices", {"result": {"BTC": 1}}, None)


def test_error_result_keeps_the_error_convention():
    call = ToolCall("zl-x", "get_status", {})
    msgs = [Message.tool_result(call, json.dumps({"error": "daemon down"}))]
    fr = make_client().to_contents(msgs)[0].parts[0].function_response
    assert fr.response == {"error": "daemon down"}


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ('{"error":"a","other":1}', {"result": {"error": "a", "other": 1}}),
        ("[1,2]", {"result": [1, 2]}),
        ('"text"', {"result": "text"}),
        ("not json {", {"result": "not json {"}),
        ("42", {"result": 42}),
    ],
)
def test_tool_result_shapes(content, expected):
    msgs = [Message.tool_result(ToolCall("zl-x", "t", {}), content)]
    assert make_client().to_contents(msgs)[0].parts[0].function_response.response == expected


def test_tool_results_and_a_following_nudge_merge_into_one_turn():
    a, b = ToolCall("zl-a", "get_prices", {}), ToolCall("zl-b", "get_status", {})
    msgs = [
        Message.user("q"),
        Message.assistant("", [a, b]),
        Message.tool_result(a, "1"),
        Message.tool_result(b, "2"),
        Message.user("answer now"),
    ]
    contents = make_client().to_contents(msgs)
    assert [c.role for c in contents] == ["user", "model", "user"]
    last = contents[2].parts
    assert [p.function_response.name for p in last[:2]] == ["get_prices", "get_status"]
    assert last[2].text == "answer now"


# ---- config ------------------------------------------------------------------------------------
def test_config_disables_automatic_function_calling_and_sets_limits():
    cfg = make_client()._config("be careful", [NO_PARAMS])
    assert cfg.automatic_function_calling.disable is True
    assert cfg.system_instruction == "be careful"
    assert cfg.max_output_tokens == 4096


def test_no_tools_means_no_tool_block():
    assert make_client()._config("s", []).tools is None


def test_tool_declarations_and_schema_handling():
    cfg = make_client()._config("s", [NO_PARAMS, WITH_PARAMS])
    decls = cfg.tools[0].function_declarations
    assert [d.name for d in decls] == ["get_prices", "get_recent_fills"]
    assert decls[0].parameters_json_schema is None  # empty-object schema is omitted
    assert decls[1].parameters_json_schema == WITH_PARAMS.parameters
    assert decls[0].description == "Latest prices."


def test_schema_without_properties_key_is_omitted():
    spec = ToolSpec("t", "d", {"type": "object"})
    decl = make_client()._config("s", [spec]).tools[0].function_declarations[0]
    assert decl.parameters_json_schema is None


# ---- Gemini response -> neutral -----------------------------------------------------------------
async def test_text_parts_are_joined_and_thoughts_are_excluded():
    parts = [text_part("Hel"), text_part("secret reasoning", thought=True), text_part("lo")]
    _, g = client_with([reply(parts)])
    response = await generate(g)
    assert response.text == "Hello" and response.tool_calls == ()


async def test_token_counts_include_thinking_tokens_as_output():
    usage = NS(prompt_token_count=11, response_token_count=4, thoughts_token_count=6)
    _, g = client_with([reply([text_part("x")], usage)])
    response = await generate(g)
    assert (response.input_tokens, response.output_tokens) == (11, 10)


async def test_missing_usage_means_zero_tokens():
    _, g = client_with([reply([text_part("x")], None)])
    response = await generate(g)
    assert (response.input_tokens, response.output_tokens) == (0, 0)


async def test_none_token_fields_count_as_zero():
    usage = NS(prompt_token_count=None, response_token_count=5, thoughts_token_count=None)
    _, g = client_with([reply([text_part("x")], usage)])
    assert (await generate(g)).output_tokens == 5


async def test_function_calls_get_unique_ids_even_if_the_server_repeats_one():
    parts = [call_part(id_="same"), call_part("get_status", id_="same")]
    _, g = client_with([reply(parts)])
    calls = (await generate(g, tools=[NO_PARAMS])).tool_calls
    assert [c.name for c in calls] == ["get_prices", "get_status"]
    assert calls[0].id != calls[1].id and all(c.id.startswith("zl-") for c in calls)
    assert all(c.id in g._parts for c in calls)


async def test_function_call_with_missing_args_and_name():
    part = types.Part(function_call=types.FunctionCall(id="s"))
    _, g = client_with([reply([part])])
    call = (await generate(g)).tool_calls[0]
    assert call.name == "" and call.args == {}


async def test_text_and_call_in_one_response():
    _, g = client_with([reply([text_part("checking"), call_part()])])
    response = await generate(g)
    assert response.text == "checking" and response.wants_tools


@pytest.mark.parametrize(
    "response",
    [
        NS(candidates=[NS(content=None)], usage_metadata=None),
        NS(candidates=[NS(content=NS(parts=None))], usage_metadata=None),
        reply([text_part("hmm", thought=True)]),
    ],
)
async def test_responses_without_answer_text_are_empty_not_errors(response):
    _, g = client_with([response])
    out = await generate(g)
    assert out.text == "" and out.tool_calls == ()


async def test_no_candidates_raises_with_the_block_reason():
    resp = NS(candidates=[], prompt_feedback=NS(block_reason="SAFETY"))
    _, g = client_with([resp])
    with pytest.raises(LLMError, match="SAFETY"):
        await generate(g)


async def test_no_candidates_without_feedback_still_raises():
    _, g = client_with([NS(candidates=None)])
    with pytest.raises(LLMError, match="no candidates") as exc:
        await generate(g)
    assert "blocked" not in str(exc.value)


def test_part_cache_is_bounded_and_evicts_oldest(monkeypatch):
    monkeypatch.setattr(gem_mod, "_PART_CACHE_SIZE", 3)
    g = make_client()
    for i in range(5):
        g._remember(f"id{i}", i)
    assert list(g._parts) == ["id2", "id3", "id4"]


# ---- request building through generate() ---------------------------------------------------------
async def test_request_carries_model_contents_and_config():
    models, g = client_with([reply([text_part("ok")])])
    await g.generate("sys", [Message.user("hi")], [NO_PARAMS])
    request = models.calls[0]
    assert request["model"] == "gemini-test"
    assert [c.role for c in request["contents"]] == ["user"]
    assert request["config"].system_instruction == "sys"


# ---- error translation ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ({"error": {"details": [{"retryDelay": "33s"}]}}, 33.0),
        ({"retryDelay": "1.5s"}, 1.5),
        ({"retryDelay": " 5s "}, 5.0),
        ([{"x": [{"retryDelay": "2s"}]}], 2.0),
        ({"retryDelay": "soon"}, None),
        ({"retryDelay": "33"}, None),
        ({"retryDelay": "-5s"}, None),
        ({"retryDelay": 33}, None),
        ({}, None),
        ([], None),
        ("garbage", None),
        (None, None),
    ],
)
def test_parse_retry_delay(value, expected):
    assert parse_retry_delay(value) == expected


def test_429_becomes_rate_limit_error_with_retry_after():
    details = [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "33s"}]
    err = translate_error(api_error(errors.ClientError, 429, details=details))
    assert isinstance(err, RateLimitError) and err.retry_after == 33.0


def test_429_without_a_delay_has_no_retry_after():
    err = translate_error(api_error(errors.ClientError, 429))
    assert isinstance(err, RateLimitError) and err.retry_after is None


@pytest.mark.parametrize(
    ("cls", "code"),
    [
        (errors.ServerError, 500),
        (errors.ServerError, 503),
        (errors.ServerError, 599),
        (errors.ClientError, 408),
    ],
)
def test_transient_failures_are_unavailable(cls, code):
    err = translate_error(api_error(cls, code))
    assert isinstance(err, LLMUnavailable)


@pytest.mark.parametrize("code", [401, 403])
def test_credential_failures_are_plain_errors_with_a_hint(code):
    err = translate_error(api_error(errors.ClientError, code))
    assert type(err) is LLMError and "check the key" in str(err)


@pytest.mark.parametrize("code", [400, 404, 418])
def test_other_client_errors_are_plain_errors(code):
    err = translate_error(api_error(errors.ClientError, code))
    assert type(err) is LLMError and f"({code})" in str(err)


def test_messages_are_redacted_and_truncated():
    message = f"bad key {SECRET} " + "x" * 2000
    err = translate_error(api_error(errors.ClientError, 400, message=message), SECRET)
    assert SECRET not in str(err) and "<redacted>" in str(err)
    assert len(str(err)) < 600


def test_redaction_also_covers_the_fallback_text():
    exc = errors.ClientError(400, {"error": {"status": f"KEY {SECRET}"}})
    err = translate_error(exc, SECRET)
    assert SECRET not in str(err)


@pytest.mark.parametrize("exc", [TimeoutError(), ConnectionResetError(), OSError("x")])
def test_network_errors_are_unavailable(exc):
    assert isinstance(translate_error(exc), LLMUnavailable)


@pytest.mark.parametrize("module", ["httpx", "httpx2", "httpcore", "httpcore2"])
def test_transport_library_errors_are_unavailable(module):
    exc = type("ReadTimeout", (Exception,), {"__module__": module})()
    assert isinstance(translate_error(exc), LLMUnavailable)


@pytest.mark.parametrize("exc", [ValueError("bug"), KeyError("k"), RuntimeError("x")])
def test_unknown_exceptions_are_not_ours(exc):
    assert translate_error(exc) is None


async def test_generate_translates_and_hides_the_original_exception():
    _, g = client_with([api_error(errors.ClientError, 429)])
    with pytest.raises(RateLimitError) as exc:
        await generate(g)
    assert exc.value.__cause__ is None and exc.value.__suppress_context__ is True


async def test_generate_redacts_the_key_from_errors():
    err = api_error(errors.ClientError, 400, message=f"bad key {SECRET}")
    _, g = client_with([err])
    with pytest.raises(LLMError) as exc:
        await generate(g)
    assert SECRET not in str(exc.value)


async def test_generate_propagates_unknown_exceptions():
    _, g = client_with([ValueError("bug")])
    with pytest.raises(ValueError, match="bug"):
        await generate(g)


# ---- integration ---------------------------------------------------------------------------------------
async def test_full_agent_loop_round_trip():
    models, g = client_with([reply([call_part()]), reply([text_part("BTC is 1")])])
    stub = Stub({"BTCUSDT": {"mid": 1.0}})
    result = await AgentLoop(g, ToolRegistry(stub)).ask("price?")
    assert (result.status, result.answer, result.steps) == ("ok", "BTC is 1", 2)
    assert stub.calls == [("prices", {})]
    first, second = models.calls
    assert first["config"].tools is not None
    assert "read-only" in first["config"].system_instruction
    contents = second["contents"]
    assert [c.role for c in contents] == ["user", "model", "user"]
    assert contents[1].parts[0].thought_signature == b"sig"
    fr = contents[2].parts[0].function_response
    assert (fr.id, fr.name) == ("srv-1", "get_prices")
    assert fr.response == {"result": {"BTCUSDT": {"mid": 1.0}}}


async def test_retrying_client_honours_a_gemini_retry_delay():
    details = [{"retryDelay": "5s"}]
    rate_limited = api_error(errors.ClientError, 429, details=details)
    _, g = client_with([rate_limited, reply([text_part("ok")])])
    delays = []

    async def sleep(seconds):
        delays.append(seconds)

    retrying = RetryingClient(g, sleep=sleep)
    assert (await retrying.generate("s", [Message.user("q")], [])).text == "ok"
    assert delays == [5.0] and retrying.retries == 1


async def test_agent_reports_exhausted_quota_instead_of_hanging():
    details = [{"retryDelay": "3600s"}]
    _, g = client_with([api_error(errors.ClientError, 429, details=details)])
    result = await AgentLoop(RetryingClient(g), ToolRegistry(Stub({}))).ask("q")
    assert result.status == "rate_limited" and not result.ok
