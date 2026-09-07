
from __future__ import annotations

from typing import Any, AsyncIterator, Callable

import pytest_asyncio

from src.modules.memory.config import settings
from src.modules.memory.memory_graph import MemoryGraph

KNOWN_DOMAIN = "candelalabs.io"
KNOWN_CLAIM_REF = "PRCL17012020000005"
KNOWN_CLAIM_STATUS = "Completed"
KNOWN_CLAIM_PORTAL_STATUS = "Terminated"
SECOND_KNOWN_CLAIM_REF = "PRCL17012020000004"
OTHER_TENANT_CLAIM_REF = "MNCL0502200007"  
KNOWN_TOTAL_CLAIMS_FOR_DOMAIN = 31


class FakeLLMClient:
    def __init__(
        self,
        json_response: str | Callable[[list[dict]], str] | None = None,
        content_response: str | Callable[[list[dict]], str] = "This is a test response.",
        tool_responses: list[dict[str, Any]] | None = None,
    ):
        self.json_response = json_response
        self.content_response = content_response
        self.tool_responses = list(tool_responses) if tool_responses is not None else None
        self.chat_json_calls: list[list[dict]] = []
        self.stream_chat_calls: list[list[dict]] = []
        self.chat_with_tools_calls: list[list[dict]] = []

    async def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        temperature: float = 0.0,
        think: bool | None = None,
    ) -> dict[str, Any]:
        self.chat_with_tools_calls.append(messages)
        if self.tool_responses is None:
            raise AssertionError(
                "FakeLLMClient.chat_with_tools() called but no tool_responses configured"
            )
        if not self.tool_responses:
            raise AssertionError(
                "FakeLLMClient.chat_with_tools() called more times than "
                "tool_responses provided -- likely an unbounded loop in the "
                "code under test"
            )
        return self.tool_responses.pop(0)

    async def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        think: bool = False,
    ) -> str:
        self.chat_json_calls.append(messages)
        if self.json_response is None:
            raise AssertionError(
                "FakeLLMClient.chat_json() called but no json_response configured"
            )
        if callable(self.json_response):
            return self.json_response(messages)
        return self.json_response

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        think: bool = True,
    ) -> AsyncIterator[dict[str, str]]:
        self.stream_chat_calls.append(messages)
        text = (
            self.content_response(messages)
            if callable(self.content_response)
            else self.content_response
        )
        yield {"type": "thinking", "text": "(fake reasoning)"}
        yield {"type": "content", "text": text}


class FakeLookupClaimTool:
    def __init__(self, result: dict[str, Any] | None = None):
        self.result = result if result is not None else {"rows": []}
        self.calls: list[dict[str, Any]] = []

    async def __call__(
        self,
        query: str,
        *,
        domain: str | None,
        memory_client: Any,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append({"query": query, "domain": domain})
        return self.result


@pytest_asyncio.fixture
async def memory():
    """Real MemoryGraph against the project's dev Neo4j instance."""
    async with MemoryGraph(settings) as mem:
        yield mem
