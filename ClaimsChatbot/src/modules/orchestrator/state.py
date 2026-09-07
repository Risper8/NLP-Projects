from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    # Input
    user_message: str
    session_id: str
    user_email: str
    request_id: str
    client_domain: str | None
    is_admin: bool
    rejected: bool
    rejection_reason: str | None
    route: str | None
    intent: str | None
    claim_reference: str | None
    additional_claim_references: list[str]
    aspect: str | None
    search_query: str | None
    missing_information: list[str]
    clarification_required: bool
    clarification_question: str | None
    claim_choices: list[dict[str, Any]]
    resolved_directly: bool
    cypher_results: dict[str, Any] | None
    additional_cypher_results: list[dict[str, Any]]
    graphrag_results: dict[str, Any] | None
    portfolio_results: dict[str, Any] | None

    memory_context: str | None

    # Output
    final_response: str | None
