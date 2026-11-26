"""Gemini adapter: translates between the agent's neutral types and google-genai.

Written against google-genai 3.6.1 (inspected, not guessed). Provider data that the neutral
types cannot carry (thought signatures on function-call parts) is kept in a small bounded
cache keyed by our tool-call id, and replayed on later turns.
"""
from __future__ import annotations

import json
import re
import uuid
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

from google import genai
from google.genai import errors, types

from zetaline.agent.config import AgentConfig
from zetaline.agent.llm import (
    LLMClient,
    LLMError,
    LLMResponse,
    LLMUnavailable,
    Message,
    RateLimitError,
    ToolCall,
    ToolSpec,
)

USER_ROLE = "user"
MODEL_ROLE = "model"
TOOL_ROLE = "user"  # function responses travel in a user-side turn (verified by the live check)

_PART_CACHE_SIZE = 512
_MAX_DETAIL_CHARS = 300
_RETRY_DELAY_RE = re.compile(r"\s*(\d+(?:\.\d+)?)s\s*")
_TRANSPORT_MODULES = frozenset({"httpx", "httpx2", "httpcore", "httpcore2", "ssl", "socket"})
_TRANSIENT_CODES = frozenset({408, 500, 502, 503, 504})


def parse_retry_delay(details: Any) -> float | None:
    """Find a Google `retryDelay` such as "33s" anywhere in an error body (best effort)."""
    stack = [details]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            delay = item.get("retryDelay")
            if isinstance(delay, str):
                match = _RETRY_DELAY_RE.fullmatch(delay)
                if match:
                    return float(match.group(1))
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return None


def _redact(text: str, secret: str) -> str:
    return text.replace(secret, "<redacted>") if secret else text


def translate_error(exc: Exception, secret: str = "") -> LLMError | None:
    """Map an SDK or transport exception to our error types; None means 'not ours'."""
    if isinstance(exc, errors.APIError):
        code = getattr(exc, "code", None)
        raw = getattr(exc, "message", None)
        detail = _redact(str(raw if raw else exc), secret)[:_MAX_DETAIL_CHARS]
        if code == 429:
            return RateLimitError(
                f"Gemini rate limit or quota (429): {detail}",
                retry_after=parse_retry_delay(getattr(exc, "details", None)),
            )
        if isinstance(exc, errors.ServerError) or code in _TRANSIENT_CODES:
            return LLMUnavailable(f"Gemini unavailable ({code}): {detail}")
        if code in (401, 403):
            return LLMError(
                f"Gemini rejected the credentials or model access ({code}); "
                f"check the key and the model name. {detail}"
            )
        return LLMError(f"Gemini API error ({code}): {detail}")
    transport = type(exc).__module__.split(".")[0] in _TRANSPORT_MODULES
    if isinstance(exc, TimeoutError | OSError) or transport:
        return LLMUnavailable(f"network problem talking to Gemini: {type(exc).__name__}")
    return None


class GeminiClient(LLMClient):
    def __init__(
        self,
        config: AgentConfig,
        *,
        client: Any = None,
        max_output_tokens: int = 4096,
        api_version: str = "v1beta",
    ) -> None:
        if not 1 <= max_output_tokens <= 65_536:
            raise ValueError("max_output_tokens must be between 1 and 65536")
        if not re.fullmatch(r"v\d+(?:alpha|beta)?\d*", api_version):
            raise ValueError(f"invalid api_version: {api_version!r}")
        self.api_version = api_version
        self._model = config.model
        self._secret = config.api_key
        self._max_output_tokens = max_output_tokens
        self._client = (
            client
            if client is not None
            else genai.Client(
                api_key=config.api_key,
                http_options=types.HttpOptions(api_version=api_version),
            )
        )
        self._parts: OrderedDict[str, Any] = OrderedDict()

    # ---- neutral -> Gemini ---------------------------------------------------------
    def _remember(self, call_id: str, part: Any) -> None:
        self._parts[call_id] = part
        while len(self._parts) > _PART_CACHE_SIZE:
            self._parts.popitem(last=False)

    def _assistant_parts(self, msg: Message) -> list[Any]:
        parts: list[Any] = []
        if msg.content:
            parts.append(types.Part(text=msg.content))
        for call in msg.tool_calls:
            cached = self._parts.get(call.id)
            if cached is not None:
                parts.append(cached)  # the original part, thought signature included
            else:
                function_call = types.FunctionCall(name=call.name, args=dict(call.args))
                parts.append(types.Part(function_call=function_call))
        return parts

    def _tool_part(self, msg: Message) -> Any:
        try:
            parsed = json.loads(msg.content)
        except ValueError:
            parsed = msg.content
        if isinstance(parsed, dict) and set(parsed) == {"error"}:
            response = parsed
        else:
            response = {"result": parsed}
        cached = self._parts.get(msg.tool_call_id)
        server_id = getattr(getattr(cached, "function_call", None), "id", None)
        return types.Part(
            function_response=types.FunctionResponse(
                id=server_id or None, name=msg.tool_name, response=response
            )
        )

    def to_contents(self, messages: Sequence[Message]) -> list[Any]:
        turns: list[tuple[str, list[Any]]] = []
        for msg in messages:
            if msg.role == "assistant":
                role, parts = MODEL_ROLE, self._assistant_parts(msg)
            elif msg.role == "tool":
                role, parts = TOOL_ROLE, [self._tool_part(msg)]
            else:
                role = USER_ROLE
                parts = [types.Part(text=msg.content)] if msg.content else []
            if not parts:
                continue
            if turns and turns[-1][0] == role:
                turns[-1][1].extend(parts)  # adjacent same-side turns become one
            else:
                turns.append((role, parts))
        return [types.Content(role=role, parts=parts) for role, parts in turns]

    def _config(self, system: str, tools: Sequence[ToolSpec]) -> Any:
        declared = None
        if tools:
            declarations = [
                types.FunctionDeclaration(
                    name=spec.name,
                    description=spec.description,
                    # A tool without parameters omits the schema entirely.
                    parameters_json_schema=spec.parameters
                    if spec.parameters.get("properties")
                    else None,
                )
                for spec in tools
            ]
            declared = [types.Tool(function_declarations=declarations)]
        return types.GenerateContentConfig(
            system_instruction=system,
            tools=declared,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            max_output_tokens=self._max_output_tokens,
        )

    # ---- Gemini -> neutral ---------------------------------------------------------
    def _parse(self, response: Any) -> LLMResponse:
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            feedback = getattr(response, "prompt_feedback", None)
            reason = getattr(feedback, "block_reason", None)
            suffix = f" (blocked: {reason})" if reason else ""
            raise LLMError(f"Gemini returned no candidates{suffix}")
        content = getattr(candidates[0], "content", None)
        texts: list[str] = []
        calls: list[ToolCall] = []
        for part in getattr(content, "parts", None) or []:
            function_call = getattr(part, "function_call", None)
            if function_call is not None:
                call_id = f"zl-{uuid.uuid4().hex[:12]}"  # unique even if the server reuses ids
                calls.append(
                    ToolCall(call_id, function_call.name or "", dict(function_call.args or {}))
                )
                self._remember(call_id, part)
            elif getattr(part, "text", None) and not getattr(part, "thought", False):
                texts.append(part.text)
        usage = getattr(response, "usage_metadata", None)
        input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
        output_tokens = int(getattr(usage, "response_token_count", 0) or 0) + int(
            getattr(usage, "thoughts_token_count", 0) or 0
        )
        return LLMResponse(
            text="".join(texts),
            tool_calls=tuple(calls),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    async def generate(
        self,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
    ) -> LLMResponse:
        contents = self.to_contents(messages)
        config = self._config(system, tools)
        try:
            response = await self._client.aio.models.generate_content(
                model=self._model, contents=contents, config=config
            )
        except Exception as exc:
            translated = translate_error(exc, self._secret)
            if translated is None:
                raise
            raise translated from None  # drop the cause: it could echo request details
        return self._parse(response)
