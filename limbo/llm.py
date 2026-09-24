"""Provider-agnostic LLM client used by the LIMBO agent scaffold.

The same normalized conversation is rendered either to the OpenAI-style
``/chat/completions`` protocol or to the ``/responses`` protocol, because the
models reachable through one gateway disagree on which endpoint they accept.

Credentials are resolved at runtime and never written to logs or results:
``LIMBO_BASE_URL`` (default: the OpenAI API) and ``LIMBO_API_KEY`` select any
OpenAI-compatible endpoint, and ``LIMBO_EXTRA_HEADERS`` (a JSON object) adds any
request headers that endpoint requires.
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

DEFAULT_BASE_URL = "https://api.openai.com/v1"
MAX_RETRY_AFTER_S = 180.0

# Models that only accept (or behave best on) the Responses protocol.
_RESPONSES_PREFIXES = ("gpt-5", "gpt-6", "o3", "o4", "grok", "codex")
_CHAT_ONLY = {"gpt-5-mini"}


def default_protocol(model: str) -> str:
    if model in _CHAT_ONLY:
        return "chat"
    return "responses" if model.startswith(_RESPONSES_PREFIXES) else "chat"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON text exactly as the model produced it

    def parsed_arguments(self) -> tuple[dict[str, Any] | None, str | None]:
        try:
            value = json.loads(self.arguments or "{}")
        except json.JSONDecodeError as exc:
            return None, f"arguments are not valid JSON: {exc.msg}"
        if not isinstance(value, dict):
            return None, "arguments must be a JSON object"
        return value, None


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.reasoning_tokens += other.reasoning_tokens
        self.cached_tokens += other.cached_tokens

    def as_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cached_tokens": self.cached_tokens,
        }


@dataclass
class AssistantTurn:
    text: str
    tool_calls: list[ToolCall]
    usage: Usage
    latency_s: float
    # Opaque protocol items (e.g. encrypted reasoning) replayed on the next turn.
    replay_items: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str | None = None


class LLMError(RuntimeError):
    def __init__(self, message: str, status: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


def resolve_credentials() -> tuple[str, str]:
    """Return (base_url, api_key). Never logs the key."""
    key = os.getenv("LIMBO_API_KEY")
    if not key:
        raise LLMError("no credentials: set LIMBO_API_KEY (and LIMBO_BASE_URL for endpoints other than OpenAI)")
    return os.getenv("LIMBO_BASE_URL", DEFAULT_BASE_URL).rstrip("/"), key


def extra_headers() -> dict[str, str]:
    """Additional request headers from ``LIMBO_EXTRA_HEADERS`` (a JSON object), for gateways that need them."""
    raw = os.getenv("LIMBO_EXTRA_HEADERS")
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMError(f"LIMBO_EXTRA_HEADERS is not valid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise LLMError("LIMBO_EXTRA_HEADERS must be a JSON object")
    return {str(k): str(v) for k, v in value.items()}


class RateGate:
    """Per-model concurrency cap shared by all worker threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sems: dict[str, threading.BoundedSemaphore] = {}

    def get(self, model: str, limit: int) -> threading.BoundedSemaphore:
        with self._lock:
            if model not in self._sems:
                self._sems[model] = threading.BoundedSemaphore(limit)
            return self._sems[model]


_GATE = RateGate()


class LLMClient:
    def __init__(
        self,
        model: str,
        *,
        protocol: str | None = None,
        max_output_tokens: int = 8192,
        reasoning_effort: str | None = None,
        temperature: float | None = None,
        timeout_s: float = 240.0,
        max_attempts: int = 8,
        concurrency: int = 6,
    ) -> None:
        self.model = model
        self.protocol = protocol or default_protocol(model)
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.max_attempts = max_attempts
        self.concurrency = concurrency
        self.base_url, self._token = resolve_credentials()
        self._extra_headers = extra_headers()
        self._replay_supported = True

    # ------------------------------------------------------------------ http
    def _headers(self) -> dict[str, str]:
        auth = "Bearer " + self._token
        return {"Content-Type": "application/json", "Authorization": auth, **self._extra_headers}

    def _post(self, path: str, body: dict[str, Any]) -> tuple[dict[str, Any], float]:
        data = json.dumps(body).encode("utf-8")
        last: Exception | None = None
        for attempt in range(self.max_attempts):
            t0 = time.time()
            req = urllib.request.Request(
                self.base_url + path, data=data, headers=self._headers(), method="POST"
            )
            delay: float | None = None
            try:
                with _GATE.get(self.model, self.concurrency):
                    with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                        payload = json.loads(resp.read().decode("utf-8"))
                return payload, time.time() - t0
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:500]
                retryable = exc.code in (408, 409, 429, 500, 502, 503, 504)
                last = LLMError(f"HTTP {exc.code}: {detail}", status=exc.code, retryable=retryable)
                if not retryable:
                    raise last
                retry_after = exc.headers.get("retry-after") if exc.headers else None
                if retry_after and retry_after.strip().isdigit():
                    delay = float(retry_after)
                    if delay > MAX_RETRY_AFTER_S:
                        # A quota window, not a burst limit: fail fast so the episode is retried later
                        # instead of parking a worker thread for an hour.
                        raise LLMError(f"HTTP {exc.code}: quota exhausted, retry after {delay:.0f}s: {detail[:120]}",
                                       status=exc.code, retryable=False)
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                last = LLMError(f"transport error: {exc}", retryable=True)
            except json.JSONDecodeError as exc:
                last = LLMError(f"invalid JSON from gateway: {exc}", retryable=True)
            sleep = delay if delay is not None else min(60.0, 2.0 * (2**attempt)) * (0.5 + random.random())
            time.sleep(sleep)
        raise LLMError(f"gave up after {self.max_attempts} attempts: {last}", retryable=False)

    # ------------------------------------------------------------ protocols
    def complete(
        self,
        system: str,
        conversation: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AssistantTurn:
        if self.protocol == "responses":
            return self._complete_responses(system, conversation, tools)
        return self._complete_chat(system, conversation, tools)

    def _complete_chat(self, system, conversation, tools) -> AssistantTurn:
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for turn in conversation:
            role = turn["role"]
            if role == "user":
                messages.append({"role": "user", "content": turn["content"]})
            elif role == "assistant":
                msg: dict[str, Any] = {"role": "assistant", "content": turn.get("content") or ""}
                if turn.get("tool_calls"):
                    msg["tool_calls"] = [
                        {
                            "id": c["id"],
                            "type": "function",
                            "function": {"name": c["name"], "arguments": c["arguments"] or "{}"},
                        }
                        for c in turn["tool_calls"]
                    ]
                messages.append(msg)
            elif role == "tool":
                messages.append(
                    {"role": "tool", "tool_call_id": turn["tool_call_id"], "content": turn["content"]}
                )
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": [{"type": "function", "function": t} for t in tools],
            "max_tokens": self.max_output_tokens,
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature
        payload, latency = self._post("/chat/completions", body)
        choices = payload.get("choices") or []
        if not choices:
            raise LLMError(f"no choices in response: {json.dumps(payload)[:300]}", retryable=True)
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        finish = None
        # Some gateways split one assistant turn across several choices.
        for ch in choices:
            m = ch.get("message") or {}
            if m.get("content"):
                text_parts.append(m["content"])
            for c in m.get("tool_calls") or []:
                fn = c.get("function") or {}
                calls.append(
                    ToolCall(c.get("id") or f"call_{len(calls)}", fn.get("name", ""), fn.get("arguments") or "{}")
                )
            finish = ch.get("finish_reason") or finish
        u = payload.get("usage") or {}
        usage = Usage(
            input_tokens=int(u.get("prompt_tokens") or 0),
            output_tokens=int(u.get("completion_tokens") or 0),
            reasoning_tokens=int(
                u.get("reasoning_tokens") or (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
            ),
            cached_tokens=int((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0),
        )
        return AssistantTurn("\n".join(text_parts), calls, usage, latency, [], finish)

    def _responses_input(self, conversation: list[dict[str, Any]]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for turn in conversation:
            role = turn["role"]
            if role == "user":
                items.append({"role": "user", "content": turn["content"]})
            elif role == "assistant":
                if self._replay_supported:
                    items.extend(turn.get("replay_items") or [])
                if turn.get("content"):
                    items.append(
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": turn["content"]}],
                        }
                    )
                for c in turn.get("tool_calls") or []:
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": c["id"],
                            "name": c["name"],
                            "arguments": c["arguments"] or "{}",
                        }
                    )
            elif role == "tool":
                items.append(
                    {"type": "function_call_output", "call_id": turn["tool_call_id"], "output": turn["content"]}
                )
        return items

    def _complete_responses(self, system, conversation, tools) -> AssistantTurn:
        body: dict[str, Any] = {
            "model": self.model,
            "instructions": system,
            "input": self._responses_input(conversation),
            "tools": [{"type": "function", **t} for t in tools],
            "max_output_tokens": self.max_output_tokens,
            "store": False,
        }
        if self._replay_supported:
            body["include"] = ["reasoning.encrypted_content"]
        if self.reasoning_effort:
            body["reasoning"] = {"effort": self.reasoning_effort}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        try:
            payload, latency = self._post("/responses", body)
        except LLMError as exc:
            if self._replay_supported and exc.status == 400:
                # Gateway refused encrypted reasoning replay; fall back to stateless turns.
                self._replay_supported = False
                body.pop("include", None)
                body["input"] = self._responses_input(conversation)
                payload, latency = self._post("/responses", body)
            else:
                raise
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        replay: list[dict[str, Any]] = []
        for item in payload.get("output") or []:
            kind = item.get("type")
            if kind == "message":
                for part in item.get("content") or []:
                    if part.get("type") in ("output_text", "text") and part.get("text"):
                        text_parts.append(part["text"])
            elif kind == "function_call":
                calls.append(
                    ToolCall(
                        item.get("call_id") or item.get("id") or f"call_{len(calls)}",
                        item.get("name", ""),
                        item.get("arguments") or "{}",
                    )
                )
            elif kind == "reasoning" and item.get("encrypted_content"):
                replay.append({k: v for k, v in item.items() if k in ("type", "id", "summary", "encrypted_content")})
        u = payload.get("usage") or {}
        usage = Usage(
            input_tokens=int(u.get("input_tokens") or 0),
            output_tokens=int(u.get("output_tokens") or 0),
            reasoning_tokens=int((u.get("output_tokens_details") or {}).get("reasoning_tokens") or 0),
            cached_tokens=int((u.get("input_tokens_details") or {}).get("cached_tokens") or 0),
        )
        status = payload.get("status")
        if status == "incomplete" and not calls and not text_parts:
            raise LLMError(f"incomplete response: {payload.get('incomplete_details')}", retryable=True)
        return AssistantTurn("\n".join(text_parts), calls, usage, latency, replay, status)
