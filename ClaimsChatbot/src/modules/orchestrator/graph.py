
from __future__ import annotations
import asyncio
import json
import logging
import re
from typing import Any
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from src.modules.guardrails.guardrail import guardrail
from src.modules.llm.client import LLMClient
from src.modules.llm.prompts import AGENT_SYSTEM_PROMPT
from src.modules.orchestrator.state import AgentState
from src.modules.orchestrator.translator import extract_claim_references
from src.modules.tools import portfolio_summary
from src.modules.tools.lookup_claim import get_lookup_claim_tool
from src.modules.memory.memory_graph import MemoryGraph
from src.modules.utils.logging import StageTimer, log_event

logger = logging.getLogger(__name__)
TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_portfolio_summary",
            "description": (
                "Get an exact count and status breakdown of the user's claims "
                "as a group. Use for questions about counts, totals, breakdowns "
                "by status, or listing multiple claims -- not one specific claim."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": (
                            "How many individual claims to list, most recent "
                            "first. Default 10; raise it (up to 100) only "
                            "when the user explicitly asks to see all claims, "
                            "export the full list, or similar."
                        ),
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_claim",
            "description": (
                "Look up one specific named claim, compare named claims, or "
                "find relationships between a claim and another entity like a "
                "broker or cedant. Use for questions naming a specific claim "
                "reference or asking about one claim's details -- including a "
                "follow-up that doesn't repeat the reference but is clearly "
                "still about the claim just discussed (e.g. the previous "
                "turn named or offered one claim, and this message asks for "
                "more about it, like a specific field such as reserves or "
                "settlement date). get_portfolio_summary's results never "
                "include per-claim detail fields like reserves or settlement "
                "date, only status/portal_status/modified_date -- if the "
                "question needs a field beyond that for one particular "
                "claim, this is the right tool even without the reference "
                "being restated. When calling it for such a follow-up, put "
                "the claim reference (resolved from the conversation) into "
                "the query yourself."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The natural-language question about the claim(s).",
                    }
                },
                "required": ["query"],
            },
        },
    },
]


MAX_TOOL_ITERATIONS = 4

REDIRECT_CONFIRMATION_PATTERN = re.compile(r"Let's start with \*\*([A-Za-z0-9]+)\*\*")
SHORT_AFFIRMATIVE_PATTERN = re.compile(
    r"^(yes|yeah|yep|sure|ok(?:ay)?|please|go ahead|that one|please do|start there|do that)[.!]?$",
    re.IGNORECASE,
)


def _emit_status(text: str) -> None:
    get_stream_writer()({"type": "status", "text": text})


def _build_single_claim_query(original_message: str, confirmed_ref: str) -> str:
    # Strips every OTHER claim reference out of the original compound
    # question, leaving the same field-specific intent scoped to just the
    # one claim the user confirmed -- e.g. "reserve, class and type of
    # business for X and Y?" -> "reserve, class and type of business for X?"
    other_refs = [r for r in extract_claim_references(original_message) if r != confirmed_ref]
    text = original_message
    for ref in other_refs:
        text = text.replace(ref, "")
    text = re.sub(r"\s+and\s+(?=[?.,]|$)", " ", text)
    text = re.sub(r"\s*,\s*(?=[?.,]|$)", "", text)
    text = re.sub(r"\s+([?.!,])", r"\1", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def build_graph(
    memory: MemoryGraph,
    llm_client: LLMClient | None = None,
    lookup_claim_tool: Any = None,
):
    llm_client = llm_client or LLMClient()
    lookup_claim = lookup_claim_tool or get_lookup_claim_tool()
    background_tasks: list[asyncio.Task] = []

    def _store_in_background(**kwargs: Any) -> None:
        background_tasks.append(
            asyncio.create_task(memory.store_short_term(**kwargs))
        )

    async def load_memory(state: AgentState) -> dict:
        with StageTimer() as timer:
            context = await memory.get_session_context(
                session_id=state["session_id"],
                max_items=10,
            )

        log_event(
            logger, "stage_complete",
            request_id=state.get("request_id"), session_id=state["session_id"],
            stage="load_memory", duration_ms=timer.duration_ms,
        )

        return {"memory_context": context}

    async def _run_tool(name: str, arguments: dict[str, Any], state: AgentState) -> Any:
        domain = state.get("client_domain")
        request_id = state.get("request_id")

        if name == "get_portfolio_summary":
            _emit_status("Checking your claims...")
            try:
                limit = max(1, min(int(arguments.get("limit") or 10), 100))
            except (TypeError, ValueError):
                limit = 10
            with StageTimer() as timer:
                result = await portfolio_summary.get_portfolio_summary(
                    memory.client, domain=domain, claims_limit=limit
                )
            log_event(
                logger, "stage_complete",
                request_id=request_id, session_id=state["session_id"],
                stage="portfolio", duration_ms=timer.duration_ms,
                total_claims=result.get("total_claims"),
            )
            return result

        if name == "lookup_claim":
            _emit_status("Looking that up...")
            query = arguments.get("query") or state["user_message"]
            result = await lookup_claim(
                query, domain=domain, memory_client=memory.client, request_id=request_id
            )
            return result

        return {"error": f"unknown_tool:{name}"}

    async def run_agent_loop(state: AgentState) -> dict:
        request_id = state.get("request_id")
        session_id = state["session_id"]

        messages: list[dict[str, Any]] = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
        memory_context = state.get("memory_context")
        if memory_context:
            messages.append({"role": "system", "content": f"Recent conversation:\n{memory_context}"})
        messages.append({"role": "user", "content": state["user_message"]})

        final_text: str | None = None
        route_taken = "chitchat"

        for iteration in range(MAX_TOOL_ITERATIONS):
            with StageTimer() as timer:
                result = await llm_client.chat_with_tools(messages, TOOLS, think=None)

            log_event(
                logger, "agent_call",
                request_id=request_id, session_id=session_id,
                iteration=iteration, duration_ms=timer.duration_ms,
                tool_calls=[tc["name"] for tc in result["tool_calls"]],
            )

            if not result["tool_calls"]:
                final_text = result["content"] or ""
                break

            tool_calls = result["tool_calls"]
            messages.append({
                "role": "assistant",
                "content": result["content"],
                "tool_calls": [{
                    "id": tc["id"], "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["arguments"]),
                    },
                } for tc in tool_calls],
            })

            route_taken = tool_calls[-1]["name"]
            tool_results = await asyncio.gather(
                *(_run_tool(tc["name"], tc["arguments"], state) for tc in tool_calls)
            )

            for tc, tool_result in zip(tool_calls, tool_results):
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(tool_result, default=str),
                })

        if final_text is None:
            log_event(
                logger, "agent_loop_exhausted", level=logging.WARNING,
                request_id=request_id, session_id=session_id,
            )

            messages.append({
                "role": "user",
                "content": (
                    "Give the best answer you can now, using only what's "
                    "already been retrieved above -- don't try calling a "
                    "tool again. You likely ran out of attempts before "
                    "covering every part of the original question -- name "
                    "specifically which parts you have real data for and "
                    "which you don't, rather than inventing a plausible "
                    "value for the parts you never actually looked up."
                ),
            })
            with StageTimer() as timer:
                result = await llm_client.chat_with_tools(messages, [], think=None)
            final_text = result["content"] or (
                "I'm having trouble pinning that down -- could you "
                "rephrase, or give me a specific claim reference?"
            )

        _store_in_background(
            session_id=session_id,
            role="assistant",
            content=final_text,
        )

        log_event(
            logger, "stage_complete",
            request_id=request_id, session_id=session_id,
            stage="agent_loop", route=route_taken,
        )

        return {"final_response": final_text, "route": route_taken}

    async def _resolve_redirect_confirmation(state: AgentState) -> dict | None:
        # Detects a short "yes"-style reply to our own deterministic
        # compound-claims redirect, recovers the original multi-claim
        # question from history, and resolves it straight to the
        # confirmed claim -- bypassing the agent's own tool selection,
        # which (measured directly) unreliably falls back to
        # get_portfolio_summary on a bare "yes" and then fabricates an
        # answer instead of admitting it never looked up the claim.
        if extract_claim_references(state["user_message"]):
            return None
        if not SHORT_AFFIRMATIVE_PATTERN.match(state["user_message"].strip()):
            return None

        conversation = await memory.memory.short_term.get_conversation(
            state["session_id"], limit=10
        )
        messages = conversation.messages

        last_assistant_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role.value == "assistant":
                last_assistant_idx = i
                break
        if last_assistant_idx is None or last_assistant_idx == 0:
            return None

        match = REDIRECT_CONFIRMATION_PATTERN.search(messages[last_assistant_idx].content)
        if not match:
            return None
        confirmed_ref = match.group(1)

        original_user_msg = None
        for i in range(last_assistant_idx - 1, -1, -1):
            if messages[i].role.value == "user":
                original_user_msg = messages[i].content
                break
        if original_user_msg is None or confirmed_ref not in original_user_msg:
            return None

        query = _build_single_claim_query(original_user_msg, confirmed_ref)
        request_id = state.get("request_id")
        result = await lookup_claim(
            query, domain=state.get("client_domain"),
            memory_client=memory.client, request_id=request_id,
        )

        synthesis_messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"The user confirmed they want details on claim {confirmed_ref}. "
                    f"Their original request was: {query!r}\n\n"
                    f"Retrieved data:\n{json.dumps(result, default=str)}\n\n"
                    "Answer using only this data."
                ),
            },
        ]
        synth = await llm_client.chat_with_tools(synthesis_messages, [], think=None)
        final_text = synth["content"] or (
            "I wasn't able to pin that down -- could you rephrase your question?"
        )

        _store_in_background(session_id=state["session_id"], role="assistant", content=final_text)

        log_event(
            logger, "stage_complete",
            request_id=request_id, session_id=state.get("session_id"),
            stage="redirect_confirmation", confirmed_ref=confirmed_ref,
        )

        return {"final_response": final_text, "route": "lookup_claim", "resolved_directly": True}

    async def check_compound_claims(state: AgentState) -> dict:

        # Awaited, not fire-and-forget: this node may also fire a second
        # store right after (the redirect text) -- doing both concurrently
        # via create_task has been observed to race on message-id
        # uniqueness in the memory store and silently drop one write.
        await memory.store_short_term(
            session_id=state["session_id"],
            role="user",
            content=state["user_message"],
        )

        refs = extract_claim_references(state["user_message"])

        if len(refs) < 2:
            resolved = await _resolve_redirect_confirmation(state)
            if resolved is not None:
                return resolved
            return {"clarification_required": False}

        log_event(
            logger, "stage_complete",
            request_id=state.get("request_id"), session_id=state.get("session_id"),
            stage="compound_check", clarification_required=True, claim_count=len(refs),
        )

        first = refs[0]
        redirect_text = (
            "That's a few things at once -- let's go through them one "
            f"claim at a time so I get each one right. Let's start with "
            f"**{first}** -- what would you like to know about it?"
        )

        _store_in_background(
            session_id=state["session_id"],
            role="assistant",
            content=redirect_text,
        )

        return {
            "clarification_required": True,
            "claim_choices": [{"ref_num": ref} for ref in refs],
            "final_response": redirect_text,
        }

    def after_guardrail(state: AgentState) -> str:
        return "rejected" if state.get("rejected") else "continue"

    def after_compound_check(state: AgentState) -> str:
        done = state.get("clarification_required") or state.get("resolved_directly")
        return "redirect" if done else "continue"

    graph = StateGraph(AgentState)

    graph.add_node("guardrail", guardrail)
    graph.add_node("check_compound_claims", check_compound_claims)
    graph.add_node("load_memory", load_memory)
    graph.add_node("agent_loop", run_agent_loop)

    graph.add_edge(START, "guardrail")

    graph.add_conditional_edges(
        "guardrail",
        after_guardrail,
        {"rejected": END, "continue": "check_compound_claims"},
    )

    graph.add_conditional_edges(
        "check_compound_claims",
        after_compound_check,
        {"redirect": END, "continue": "load_memory"},
    )

    graph.add_edge("load_memory", "agent_loop")
    graph.add_edge("agent_loop", END)

    return graph.compile(), background_tasks
