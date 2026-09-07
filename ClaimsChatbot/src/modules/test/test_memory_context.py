
from __future__ import annotations
import uuid
import pytest
pytestmark = pytest.mark.integration

def _unique_session_id() -> str:
    return f"test-session-{uuid.uuid4()}"


async def test_session_context_includes_a_claim_discussed_after_the_window_would_have_closed(memory):
    session_id = _unique_session_id()
    early_turns = [
        ("user", "hi"),
        ("assistant", "Hi there! How can I help?"),
        ("user", "which claims are under our company"),
        ("assistant", "You have claims OLD0000001 and NEW0000002."),
        ("user", "tell me about OLD0000001"),
        ("assistant", "OLD0000001 is completed."),
        ("user", "what about the terminated claim?"),
        ("assistant", "OLD0000001 is not terminated."),
    ]
    later_turns = [
        ("user", "ok thanks"),
        ("assistant", "You are welcome."),
        ("user", "one more question"),
        ("assistant", "Sure, go ahead."),
        ("user", "what about this claim NEW0000002"),
        ("assistant", "NEW0000002 is recorded as Terminated, portal status Rejected."),
        ("user", "why was it rejected?"),
    ]

    for role, content in early_turns + later_turns:
        await memory.store_short_term(
            session_id=session_id, role=role, content=content, extract_entities=False
        )

    context = await memory.get_session_context(session_id=session_id, max_items=10)

    assert "NEW0000002" in context, (
        "The most recently discussed claim must be visible in context -- "
        "if this fails, get_session_context has regressed to returning "
        "the oldest N messages instead of the most recent N."
    )


async def test_session_context_is_a_true_recency_window(memory):
    session_id = _unique_session_id()
    for i in range(15):
        await memory.store_short_term(
            session_id=session_id,
            role="user" if i % 2 == 0 else "assistant",
            content=f"marker-{i:03d}-end",
            extract_entities=False,
        )

    context = await memory.get_session_context(session_id=session_id, max_items=10)

    for i in range(5):
        assert f"marker-{i:03d}-end" not in context, (
            f"message {i} is outside the last-10 window and should not appear"
        )
    for i in range(5, 15):
        assert f"marker-{i:03d}-end" in context, (
            f"message {i} is inside the last-10 window and should appear"
        )


async def test_session_context_is_empty_for_a_session_with_no_messages(memory):
    context = await memory.get_session_context(session_id=_unique_session_id(), max_items=10)
    assert context == ""
