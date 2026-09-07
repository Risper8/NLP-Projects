
from __future__ import annotations
import json
import os
from typing import Any, AsyncIterator

import litellm
_PROVIDER_PREFIXES: dict[str, str] = {
    "ollama": "ollama_chat",
    "openai": "openai",
    "gemini": "gemini",
}

DEFAULT_PROVIDER = "ollama"
DEFAULT_MODEL = "granite4.1:3b"
DEFAULT_OLLAMA_API_BASE = "http://localhost:11434"
DEFAULT_OLLAMA_NUM_CTX = 16384


MODEL_EXTRA_PARAMS: dict[str, dict[str, Any]] = {
    "granite4.2:3b": {"think": True, "reasoning_effort": "medium"},
}


def resolve_provider(provider: str | None = None) -> str:
    provider = (provider or os.environ.get("LLM_PROVIDER") or DEFAULT_PROVIDER).lower()

    if provider not in _PROVIDER_PREFIXES:
        raise ValueError(
            f"Unknown LLM_PROVIDER={provider!r}; "
            f"expected one of {sorted(_PROVIDER_PREFIXES)}"
        )

    return provider


def resolve_model_string(provider: str | None = None, model: str | None = None) -> str:

    resolved_provider = resolve_provider(provider)
    model = model or os.environ.get("LLM_MODEL") or DEFAULT_MODEL

    return f"{_PROVIDER_PREFIXES[resolved_provider]}/{model}"


def resolve_api_base(provider: str | None = None) -> str | None:
    if resolve_provider(provider) != "ollama":
        return None

    return os.environ.get("LLM_API_BASE", DEFAULT_OLLAMA_API_BASE)


class LLMClient:
    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        *,
        api_base: str | None = None,
        extra_params: dict[str, Any] | None = None,
    ):
        self.provider = resolve_provider(provider)
        self.model = resolve_model_string(self.provider, model)
        self.api_base = (
            api_base if api_base is not None else resolve_api_base(self.provider)
        )
        bare_model = model or os.environ.get("LLM_MODEL") or DEFAULT_MODEL
        # Explicit constructor override wins over the known-model table.
        self.extra_params: dict[str, Any] = {
            **MODEL_EXTRA_PARAMS.get(bare_model, {}),
            **(extra_params or {}),
        }

    def _call_kwargs(self, *, think: bool | None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"model": self.model}

        if self.api_base:
            kwargs["api_base"] = self.api_base

        if self.provider == "ollama":
            effective_think = think if think is not None else self.extra_params.get("think")

            if effective_think is not None:
                kwargs["think"] = effective_think
            if effective_think:
                kwargs["num_ctx"] = int(
                    os.environ.get("LLM_OLLAMA_NUM_CTX", DEFAULT_OLLAMA_NUM_CTX)
                )

        if "reasoning_effort" in self.extra_params:
            kwargs["reasoning_effort"] = self.extra_params["reasoning_effort"]

        return kwargs

    async def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        think: bool = False,
    ) -> str:
        response = await litellm.acompletion(
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
            **self._call_kwargs(think=think),
        )

        return response.choices[0].message.content

    async def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        temperature: float = 0.0,
        think: bool | None = None,
    ) -> dict[str, Any]:
        extra: dict[str, Any] = {"tools": tools} if tools else {}
        response = await litellm.acompletion(
            messages=messages,
            temperature=temperature,
            **extra,
            **self._call_kwargs(think=think),
        )

        msg = response.choices[0].message
        tool_calls: list[dict[str, Any]] = []
        for tc in (msg.tool_calls or []):
            try:
                arguments = json.loads(tc.function.arguments)
            except (TypeError, ValueError):
                arguments = {}
            tool_calls.append({"id": tc.id, "name": tc.function.name, "arguments": arguments})

        return {"content": msg.content, "tool_calls": tool_calls}

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        think: bool = True,
    ) -> AsyncIterator[dict[str, str]]:
        response = await litellm.acompletion(
            messages=messages,
            temperature=temperature,
            stream=True,
            **self._call_kwargs(think=think),
        )

        async for chunk in response:
            delta = chunk.choices[0].delta
            reasoning = getattr(delta, "reasoning_content", None)
            content = getattr(delta, "content", None)

            if reasoning:
                yield {"type": "thinking", "text": reasoning}

            if content:
                yield {"type": "content", "text": content}
