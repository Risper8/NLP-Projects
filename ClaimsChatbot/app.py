
from __future__ import annotations
import asyncio
import json
import uuid
from typing import Any, AsyncIterator, Iterator
from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from src.agents.claims_agent import run_turn, run_turn_streaming
from src.modules.guardrails.access import resolve_client_scope
from src.modules.orchestrator.state import AgentState

app = Flask(__name__)
_sessions: dict[str, str] = {}


class UnknownSession(Exception):
    pass


def _require_session(session_id: str | None) -> str:
    """Returns the email_address for session_id, or raises UnknownSession."""

    if not session_id or session_id not in _sessions:
        raise UnknownSession(session_id)

    return _sessions[session_id]


def _public_state(state: AgentState) -> dict[str, Any]:
    return {
        "final_response": state.get("final_response"),
        "route": state.get("route"),
        "clarification_required": bool(state.get("clarification_required")),
        "claim_choices": state.get("claim_choices") or [],
        "rejected": bool(state.get("rejected")),
    }


def _drain_async_generator(async_gen: AsyncIterator[Any]) -> Iterator[Any]:
    loop = asyncio.new_event_loop()

    try:
        while True:
            try:
                yield loop.run_until_complete(async_gen.__anext__())
            except StopAsyncIteration:
                break
    finally:
        loop.close()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/sessions")
def start_session():
    payload = request.get_json(force=True, silent=True) or {}
    email_address = payload.get("email_address", "")

    try:
        scope = resolve_client_scope(email_address)
    except ValueError as exc:
        return jsonify({"detail": str(exc)}), 400

    session_id = str(uuid.uuid4())
    _sessions[session_id] = email_address

    return jsonify(
        {
            "session_id": session_id,
            "domain": scope.domain,
            "is_admin": scope.is_admin,
        }
    )


@app.post("/chat")
def chat():
    payload = request.get_json(force=True, silent=True) or {}
    session_id = payload.get("session_id")

    try:
        email_address = _require_session(session_id)
    except UnknownSession:
        return jsonify({"detail": "Unknown session_id. Call POST /sessions first."}), 404

    message = payload.get("message", "")

    state = asyncio.run(run_turn(message, session_id, email_address))

    return jsonify(_public_state(state))


@app.post("/chat/stream")
def chat_stream():
    payload = request.get_json(force=True, silent=True) or {}
    session_id = payload.get("session_id")

    try:
        email_address = _require_session(session_id)
    except UnknownSession:
        return jsonify({"detail": "Unknown session_id. Call POST /sessions first."}), 404

    message = payload.get("message", "")

    def generate() -> Iterator[str]:
        async_gen = run_turn_streaming(message, session_id, email_address)

        for event in _drain_async_generator(async_gen):

            if event.get("type") == "final":
                data = _public_state(event["state"])
                data["type"] = "final"
            else:
                data = event

            yield f"data: {json.dumps(data)}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000)
