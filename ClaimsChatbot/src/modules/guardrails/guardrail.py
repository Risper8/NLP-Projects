from __future__ import annotations
import logging
import re
from src.modules.orchestrator.state import AgentState
from src.modules.guardrails.access import resolve_client_scope
from src.modules.utils.logging import log_event

logger = logging.getLogger(__name__)

_INJECTION_PATTERN = re.compile(
    r"(ignore|disregard|forget)\s+"
    r"(all\s+|your\s+|the\s+|previous\s+|prior\s+|above\s+|earlier\s+){1,3}"
    r"(instructions|prompt|rules)"
    r"|system\s*prompt"
    r"|reveal\s+(your\s+)?(instructions|prompt)"
    r"|print\s+your\s+(instructions|prompt)"
    r"|you\s+are\s+now\s+"
    r"|developer\s+mode"
    r"|jailbreak",
    re.IGNORECASE,
)

_INJECTION_REFUSAL = (
    "I'm only able to help with questions about your claims on this "
    "portal -- things like a claim's status, settlement, reserve, or "
    "an overview of your claims. What would you like to know?"
)


async def guardrail(state: AgentState) -> dict:

    request_id = state.get("request_id")

    try:
        scope = resolve_client_scope(state["user_email"])
    except ValueError as exc:
        log_event(
            logger, "stage_complete",
            request_id=request_id, session_id=state.get("session_id"),
            stage="guardrail", rejected=True, rejection_reason="invalid_email",
        )
        return {
            "rejected": True,
            "rejection_reason": str(exc),
            "final_response": (
                "We couldn't verify your account for this portal. "
                "Please contact support."
            ),
        }

    if _INJECTION_PATTERN.search(state["user_message"]):
        log_event(
            logger, "stage_complete",
            level=logging.WARNING,
            request_id=request_id, session_id=state.get("session_id"),
            stage="guardrail", rejected=True,
            rejection_reason="possible_prompt_injection",
        )
        return {
            "client_domain": scope.filter_domain,
            "is_admin": scope.is_admin,
            "rejected": True,
            "rejection_reason": "possible_prompt_injection",
            "final_response": _INJECTION_REFUSAL,
        }

    log_event(
        logger, "stage_complete",
        request_id=request_id, session_id=state.get("session_id"),
        stage="guardrail", rejected=False, domain=scope.domain, is_admin=scope.is_admin,
    )

    return {
        "client_domain": scope.filter_domain,
        "is_admin": scope.is_admin,
        "rejected": False,
        "rejection_reason": None,
    }
