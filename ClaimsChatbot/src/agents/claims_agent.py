
from __future__ import annotations
import asyncio
import logging
import uuid
from typing import Any, AsyncIterator
from src.modules.memory.config import settings
from src.modules.memory.memory_graph import MemoryGraph
from src.modules.orchestrator.graph import build_graph
from src.modules.orchestrator.state import AgentState
from src.modules.utils.logging import StageTimer, configure_logging, log_event

configure_logging()

logger = logging.getLogger(__name__)

_shared_embedder = None
_shared_extractor = None
_shared_init_lock = asyncio.Lock()


async def _get_shared_embedder_extractor():
    global _shared_embedder, _shared_extractor

    if _shared_embedder is not None:
        return _shared_embedder, _shared_extractor

    async with _shared_init_lock:
        if _shared_embedder is not None:
            return _shared_embedder, _shared_extractor

        async with MemoryGraph(settings) as warmup:
            _shared_embedder = warmup.memory._embedder
            _shared_extractor = warmup.memory._extractor

    return _shared_embedder, _shared_extractor


async def run_turn(user_message: str, session_id: str, user_email: str) -> AgentState:

    request_id = str(uuid.uuid4())
    embedder, extractor = await _get_shared_embedder_extractor()

    async with MemoryGraph(settings, embedder=embedder, extractor=extractor) as memory:
        graph, background_tasks = build_graph(memory)

        initial_state: AgentState = {
            "user_message": user_message,
            "session_id": session_id,
            "user_email": user_email,
            "request_id": request_id,
        }

        with StageTimer() as turn_timer:
            result = await graph.ainvoke(initial_state)

            if background_tasks:
                await asyncio.gather(*background_tasks, return_exceptions=True)

        log_event(
            logger,
            "turn_complete",
            request_id=request_id,
            session_id=session_id,
            route=result.get("route"),
            rejected=result.get("rejected", False),
            total_duration_ms=turn_timer.duration_ms,
        )

        return result


async def run_turn_streaming(
    user_message: str,
    session_id: str,
    user_email: str,
) -> AsyncIterator[dict[str, Any]]:

    request_id = str(uuid.uuid4())
    embedder, extractor = await _get_shared_embedder_extractor()

    async with MemoryGraph(settings, embedder=embedder, extractor=extractor) as memory:
        graph, background_tasks = build_graph(memory)

        initial_state: AgentState = {
            "user_message": user_message,
            "session_id": session_id,
            "user_email": user_email,
            "request_id": request_id,
        }

        final_state: AgentState = {}

        with StageTimer() as turn_timer:
            async for mode, payload in graph.astream(
                initial_state,
                stream_mode=["custom", "values"],
            ):
                if mode == "custom":
                    yield payload
                else:
                    final_state = payload

        yield {"type": "final", "state": final_state}

        log_event(
            logger,
            "turn_complete",
            request_id=request_id,
            session_id=session_id,
            route=final_state.get("route"),
            rejected=final_state.get("rejected", False),
            total_duration_ms=turn_timer.duration_ms,
        )

        if background_tasks:
            await asyncio.gather(*background_tasks, return_exceptions=True)


async def _interactive_main() -> None:
    user_email = input("Your email: ").strip() # will be adjusted to pick the email address from the portal
    session_id = f"cli-{user_email}"

    print("Type a question (Ctrl+C to quit).")

    while True:
        try:
            user_message = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_message:
            continue

        status_shown = False
        content_started = False

        async for event in run_turn_streaming(user_message, session_id, user_email):

            event_type = event.get("type")

            if event_type == "status":
                print(f"\rbot [{event['text']}]" + " " * 10, end="", flush=True)
                status_shown = True

            elif event_type == "content":
                if not content_started:
                    if status_shown:
                        print()
                    print("bot> ", end="", flush=True)
                    content_started = True
                print(event["text"], end="", flush=True)

            elif event_type == "final":
                if not content_started:
                    print(f"bot> {event['state'].get('final_response')}")
                else:
                    print()


if __name__ == "__main__":
    asyncio.run(_interactive_main())
