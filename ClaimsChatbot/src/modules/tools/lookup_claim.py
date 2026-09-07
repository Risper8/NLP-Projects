from __future__ import annotations
import asyncio
import logging
import os
import re
from typing import Any
import neo4j
import numpy as np
from dotenv import load_dotenv
from neo4j_graphrag.llm import OllamaLLM
from neo4j_graphrag.retrievers import Text2CypherRetriever
from src.modules.llm.glossary import GLOSSARY_TEXT
from src.modules.orchestrator.translator import (
    CLAIM_REFERENCE_PATTERN,
    extract_claim_references,
)
from src.modules.utils.embeddings import SentenceTransformerEmbeddingProvider
from src.modules.utils.logging import StageTimer, log_event

load_dotenv()

logger = logging.getLogger(__name__)


CLAIMS_SCHEMA = """
Node properties:
Claim {ref_num: STRING, status: STRING, portal_status: STRING, proceed_decision: STRING,
       created_date: STRING, modified_date: STRING, date_of_loss: STRING, request_type: STRING,
       class_of_business: STRING, cause_of_loss: STRING, claim_reason: STRING, reserves: FLOAT,
       base_amount: FLOAT, payment_status: STRING, client_domains: LIST}
Broker {broker_code: STRING, name: STRING}
Cedant {name: STRING, cedant_code: STRING}
Insured {name: STRING}
Payment {payment_id: STRING, status: STRING, amount: FLOAT, currency: STRING, original_currency: STRING, type: STRING, client_domains: LIST}
Query {query_id: STRING, status: STRING, reason: STRING, created_date: STRING, client_domains: LIST}
Location {name: STRING}
ProcessInstance {process_instance_id: STRING, case_id: STRING, portal_status: STRING, proceed_decision: STRING, client_domains: LIST}

Relationship properties:
(none)

The relationships:
(:Broker)-[:SUBMITTED]->(:Claim)
(:Cedant)-[:HAS_CLAIM]->(:Claim)
(:Insured)-[:HAS_CLAIM]->(:Claim)
(:Claim)-[:HAS_PROCESS]->(:ProcessInstance)
(:Claim)-[:HAS_PAYMENT]->(:Payment)
(:Claim)-[:HAS_QUERY]->(:Query)
(:Claim)-[:OCCURRED_IN]->(:Location)
""" + GLOSSARY_TEXT

CLAIMS_EXAMPLES = [
    (
        "USER INPUT: \"For the organization with domain 'candelalabs.io': "
        "what is the status of claim PRCL17012020000005?\" "
        "QUERY: MATCH (c:Claim {ref_num: 'PRCL17012020000005'}) "
        "WHERE 'candelalabs.io' IN c.client_domains "
        "RETURN c.ref_num, c.status, c.portal_status, c.proceed_decision"
    ),
    (
        "USER INPUT: \"For the organization with domain 'candelalabs.io': "
        "compare claims PRCL17012020000005 and PRCL17012020000004\" "
        "QUERY: MATCH (c:Claim) WHERE c.ref_num IN ['PRCL17012020000005', 'PRCL17012020000004'] "
        "AND 'candelalabs.io' IN c.client_domains "
        "RETURN c.ref_num, c.status, c.portal_status, c.class_of_business, c.base_amount"
    ),
    (
        "USER INPUT: \"For the organization with domain 'candelalabs.io': "
        "which claims did broker First Reinsurance Brokers (Kenya) submit?\" "
        "QUERY: MATCH (b:Broker {name: 'First Reinsurance Brokers (Kenya)'})-[:SUBMITTED]->(c:Claim) "
        "WHERE 'candelalabs.io' IN c.client_domains "
        "RETURN c.ref_num, c.status, c.portal_status"
    ),
    (
        "USER INPUT: \"For the organization with domain 'candelalabs.io': "
        "what is the settlement amount and reserve for claim PRCL17012020000005?\" "
        "QUERY: MATCH (c:Claim {ref_num: 'PRCL17012020000005'}) "
        "WHERE 'candelalabs.io' IN c.client_domains "
        "OPTIONAL MATCH (c)-[:HAS_PAYMENT]->(p:Payment) "
        "RETURN c.ref_num, c.reserves, p.amount AS settlement_amount, "
        "p.currency AS settlement_currency, p.original_currency AS reserve_currency"
    ),
    (
        "USER INPUT: \"For the organization with domain 'candelalabs.io': "
        "who is the insured and cedant on claim PRCL17012020000005?\" "
        "QUERY: MATCH (c:Claim {ref_num: 'PRCL17012020000005'}) "
        "WHERE 'candelalabs.io' IN c.client_domains "
        "OPTIONAL MATCH (i:Insured)-[:HAS_CLAIM]->(c) "
        "OPTIONAL MATCH (cd:Cedant)-[:HAS_CLAIM]->(c) "
        "RETURN c.ref_num, i.name AS insured_name, cd.name AS cedant_name"
    ),
    (
        "USER INPUT: \"For the organization with domain 'candelalabs.io': "
        "who is the insurer on claim PRCL17012020000005?\" "
        "QUERY: MATCH (c:Claim {ref_num: 'PRCL17012020000005'}) "
        "WHERE 'candelalabs.io' IN c.client_domains "
        "OPTIONAL MATCH (cd:Cedant)-[:HAS_CLAIM]->(c) "
        "RETURN c.ref_num, cd.name AS insurer_name"

    ),
    (
        "USER INPUT: \"For the organization with domain 'candelalabs.io': "
        "give me the full details for claim PRCL17012020000005\" "
        "QUERY: MATCH (c:Claim {ref_num: 'PRCL17012020000005'}) "
        "WHERE 'candelalabs.io' IN c.client_domains "
        "OPTIONAL MATCH (c)-[:HAS_PAYMENT]->(p:Payment) "
        "RETURN c.ref_num, c.status, c.portal_status, c.proceed_decision, "
        "c.date_of_loss, c.request_type, c.class_of_business, c.cause_of_loss, "
        "c.claim_reason, c.reserves, c.base_amount, c.payment_status, "
        "p.original_currency AS amount_currency"

    ),
    (
        "USER INPUT: \"For the organization with domain 'candelalabs.io': "
        "why was claim PRCL17012020000005 terminated?\" "
        "QUERY: MATCH (c:Claim {ref_num: 'PRCL17012020000005'}) "
        "WHERE 'candelalabs.io' IN c.client_domains "
        "OPTIONAL MATCH (c)-[:HAS_QUERY]->(q:Query) "
        "RETURN c.ref_num, c.portal_status, c.proceed_decision, c.claim_reason, "
        "c.cause_of_loss, collect(q.reason) AS query_reasons"
    ),
    (
        "USER INPUT: \"For the organization with domain 'candelalabs.io': "
        "what queries have been raised on claim PRCL17012020000005?\" "
        "QUERY: MATCH (c:Claim {ref_num: 'PRCL17012020000005'}) "
        "WHERE 'candelalabs.io' IN c.client_domains "
        "OPTIONAL MATCH (c)-[:HAS_QUERY]->(q:Query) "
        "RETURN c.ref_num, q.status AS query_status, q.reason AS query_reason, "
        "q.created_date AS query_created_date"
    ),
    (
        "USER INPUT: \"For the organization with domain 'candelalabs.io': "
        "which broker submitted claim PRCL17012020000005?\" "
        "QUERY: MATCH (c:Claim {ref_num: 'PRCL17012020000005'}) "
        "WHERE 'candelalabs.io' IN c.client_domains "
        "OPTIONAL MATCH (b:Broker)-[:SUBMITTED]->(c) "
        "RETURN c.ref_num, b.name AS broker_name"
    ),
    (
        "USER INPUT: \"For the organization with domain 'candelalabs.io': "
        "has claim PRCL17012020000005 been paid, and what is its payment "
        "status?\" "
        "QUERY: MATCH (c:Claim {ref_num: 'PRCL17012020000005'}) "
        "WHERE 'candelalabs.io' IN c.client_domains "
        "OPTIONAL MATCH (c)-[:HAS_PAYMENT]->(p:Payment) "
        "RETURN c.ref_num, c.payment_status AS claim_payment_status, "
        "p.status AS payment_record_status, p.amount AS amount, "
        "p.original_currency AS amount_currency"
    ),
    (
        "USER INPUT: \"For the organization with domain 'candelalabs.io': "
        "what is the case ID and process status for claim "
        "PRCL17012020000005?\" "
        "QUERY: MATCH (c:Claim {ref_num: 'PRCL17012020000005'}) "
        "WHERE 'candelalabs.io' IN c.client_domains "
        "OPTIONAL MATCH (c)-[:HAS_PROCESS]->(pi:ProcessInstance) "
        "RETURN c.ref_num, pi.case_id AS case_id, "
        "pi.portal_status AS process_portal_status, "
        "pi.proceed_decision AS process_proceed_decision"
    ),
    (
        "USER INPUT: \"For the organization with domain 'candelalabs.io': "
        "where did the loss for claim PRCL17012020000005 occur?\" "
        "QUERY: MATCH (c:Claim {ref_num: 'PRCL17012020000005'}) "
        "WHERE 'candelalabs.io' IN c.client_domains "
        "OPTIONAL MATCH (c)-[:OCCURRED_IN]->(l:Location) "
        "RETURN c.ref_num, l.name AS loss_location"
    ),
]

EXAMPLE_QUESTION_PATTERN = re.compile(
    r"USER INPUT: \"(?:For the organization with domain '[^']+': )?(.*?)\""
)
DYNAMIC_EXAMPLE_COUNT = 3


def _extract_example_question(example: str) -> str:
    match = EXAMPLE_QUESTION_PATTERN.search(example)
    return match.group(1) if match else example[:80]


EXAMPLE_QUESTIONS = [_extract_example_question(example) for example in CLAIMS_EXAMPLES]


class _ExampleSelector:
    def __init__(self) -> None:
        self._embedder = SentenceTransformerEmbeddingProvider()
        vectors = [self._embedder.embed_sync(q) for q in EXAMPLE_QUESTIONS]
        self._question_vectors = np.array(vectors)

    async def select(self, query: str, k: int = DYNAMIC_EXAMPLE_COUNT) -> list[str]:
        query_vector = np.array(await self._embedder.embed(query))
        similarities = self._question_vectors @ query_vector
        top_k = np.argsort(-similarities)[:k]
        return [CLAIMS_EXAMPLES[i] for i in top_k]


_SCHEMA_LABEL_BLOCK_PATTERN = re.compile(r"^(\w+) \{([^}]*)\}", re.MULTILINE)
_SCHEMA_RELATIONSHIP_LINE_PATTERN = re.compile(r"\(:(\w+)\)-\[:(\w+)\]->\(:(\w+)\)")
_QUERY_VAR_LABEL_PATTERN = re.compile(r"\((\w+):(\w+)\b")
_QUERY_REL_TYPE_PATTERN = re.compile(r"\[[^\]]*:(\w+)[^\]]*\]")
_QUERY_PROPERTY_REF_PATTERN = re.compile(r"\b([a-zA-Z_]\w*)\.(\w+)")


def _parse_schema(schema_text: str) -> tuple[dict[str, set[str]], set[str]]:
    props_by_label: dict[str, set[str]] = {}
    for match in _SCHEMA_LABEL_BLOCK_PATTERN.finditer(schema_text):
        label, props_blob = match.group(1), match.group(2)
        props_by_label[label] = {
            decl.strip().split(":")[0].strip()
            for decl in props_blob.split(",")
            if decl.strip()
        }

    relationships = {
        match.group(2)
        for match in _SCHEMA_RELATIONSHIP_LINE_PATTERN.finditer(schema_text)
    }
    return props_by_label, relationships


_VALID_PROPS_BY_LABEL, _VALID_RELATIONSHIPS = _parse_schema(CLAIMS_SCHEMA)


def validate_cypher_against_schema(cypher: str) -> list[str]:
    # Regex-based, not a real Cypher parser -- skips anything ambiguous.
    problems: list[str] = []

    var_to_label = {
        match.group(1): match.group(2)
        for match in _QUERY_VAR_LABEL_PATTERN.finditer(cypher)
    }

    for match in _QUERY_REL_TYPE_PATTERN.finditer(cypher):
        rel_type = match.group(1)
        if rel_type.isupper() and rel_type not in _VALID_RELATIONSHIPS:
            problems.append(f"unknown relationship type :{rel_type}")

    for match in _QUERY_PROPERTY_REF_PATTERN.finditer(cypher):
        var, prop = match.groups()
        label = var_to_label.get(var)
        if label is None:
            continue
        known_props = _VALID_PROPS_BY_LABEL.get(label)
        if known_props is not None and prop not in known_props:
            problems.append(f"{var}:{label} has no property '{prop}'")

    return problems


NOT_TRACKED_CONCEPTS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(r"\b(adjuster|claims officer|team lead|approver)\b", re.IGNORECASE),
        "who is assigned as adjuster, claims officer, team lead, or approver on a claim",
    ),
    (
        re.compile(
            r"\bsettle(?:d|ment)\b(?:\s+\w+){0,3}\s+\bdate\b"
            r"|\bdate\b(?:\s+\w+){0,3}\s+\bsettle(?:d|ment)\b"
            r"|\bwhen\b(?:\s+\w+){0,6}\s+\bsettled\b",
            re.IGNORECASE,
        ),
        "when a claim was settled (a settlement date)",
    ),
    (
        re.compile(r"\btype of business\b", re.IGNORECASE),
        "the claim's type of business (class of business is tracked, type of business is not)",
    ),
    (
        re.compile(r"\boffset(?:ting|s)?\b", re.IGNORECASE),
        "why a payment was offset or reduced",
    ),
    (
        re.compile(r"\b(broker|cedant|insurer)'?s?\s+(?:own\s+)?reference\b", re.IGNORECASE),
        "the broker's or cedant's own reference number for the claim",
    ),
    (
        # matches "country" and "cedant"/"insurer" anywhere, not just adjacent
        re.compile(r"(?=.*\bcountry\b)(?=.*\b(?:cedant|insurer)\b)", re.IGNORECASE | re.DOTALL),
        "the cedant's country",
    ),
    (
        re.compile(
            r"\b(broker|cedant|insurer)'?s?\s+(?:contact\s+)?email\b|\bemail address\b",
            re.IGNORECASE,
        ),
        "a broker's or cedant's contact email address",
    ),
]


def detect_not_tracked_concept(query: str) -> str | None:
    for pattern, description in NOT_TRACKED_CONCEPTS:
        if pattern.search(query):
            return description
    return None


TRACKED_FIELD_HINTS_PATTERN = re.compile(
    r"\b(reserve|class of business|settlement|status|insurer|insured|cedant|"
    r"broker|payment|paid|quer(?:y|ies)|process|case|location|date of loss|"
    r"cause of loss|claim reason|base amount|amount|currency)\b",
    re.IGNORECASE,
)


VERIFY_CLAIMS_QUERY = """
MATCH (c:Claim) WHERE c.ref_num IN $refs AND ($domain IS NULL OR $domain IN c.client_domains)
RETURN c.ref_num AS authorized
"""

MULTI_CLAIM_QUERY = """
MATCH (c:Claim) WHERE c.ref_num IN $refs AND ($domain IS NULL OR $domain IN c.client_domains)
OPTIONAL MATCH (c)-[:HAS_PAYMENT]->(p:Payment)
RETURN c.ref_num AS ref_num, c.status AS status, c.portal_status AS portal_status,
       c.proceed_decision AS proceed_decision, c.date_of_loss AS date_of_loss,
       c.request_type AS request_type, c.class_of_business AS class_of_business,
       c.cause_of_loss AS cause_of_loss, c.claim_reason AS claim_reason,
       c.reserves AS reserves, c.base_amount AS base_amount,
       c.payment_status AS payment_status, p.original_currency AS amount_currency
"""


class LookupClaimTool:

    def __init__(self, driver: neo4j.Driver, database: str, model_name: str = "granite4.1:3b"):
        self._driver = driver
        self._database = database
        llm = OllamaLLM(
            model_name=model_name,
            model_params={"options": {"temperature": 0}},
            host=os.environ.get("LLM_API_BASE", "http://localhost:11434"),
        )
        self._retriever = Text2CypherRetriever(
            driver=driver,
            llm=llm,
            neo4j_schema=CLAIMS_SCHEMA,
            examples=CLAIMS_EXAMPLES,
            neo4j_database=database,
        )
        self._example_selector: _ExampleSelector | None = None

    async def _ensure_example_selector(self) -> _ExampleSelector:
        if self._example_selector is None:
            self._example_selector = await asyncio.to_thread(_ExampleSelector)
        return self._example_selector

    async def __call__(
        self,
        query: str,
        *,
        domain: str | None,
        memory_client: Any,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        refs = extract_claim_references(query)
        if len(refs) >= 2:
            return await self._lookup_multiple_refs(
                refs, domain=domain, memory_client=memory_client, request_id=request_id
            )

        not_tracked = detect_not_tracked_concept(query)
        if not_tracked is not None and not TRACKED_FIELD_HINTS_PATTERN.search(query):
            # nothing else answerable in the query -- skip Text2Cypher entirely
            log_event(
                logger, "lookup_claim_not_tracked_shortcircuit",
                request_id=request_id, concept=not_tracked,
            )
            return {
                "not_tracked": True,
                "explanation": f"This system does not track {not_tracked}.",
                "cypher": "<deterministic not-tracked short-circuit>",
            }

        scoped_query_text = (
            f"For the organization with domain '{domain}': {query}"
            if domain is not None
            else query
        )

        prompt_params: dict[str, Any] | None = None
        if len(CLAIMS_EXAMPLES) > DYNAMIC_EXAMPLE_COUNT:
            selector = await self._ensure_example_selector()
            selected_examples = await selector.select(query)
            prompt_params = {"examples": "\n".join(selected_examples)}

        with StageTimer() as timer:
            try:
                raw = await asyncio.to_thread(
                    self._retriever.get_search_results, scoped_query_text, prompt_params
                )
            except Exception as exc:
                log_event(
                    logger, "lookup_claim_error", level=logging.WARNING,
                    request_id=request_id, error=str(exc),
                )
                return {"error": "lookup_failed"}

            generated_cypher = raw.metadata.get("cypher", "") if raw.metadata else ""
            problems = validate_cypher_against_schema(generated_cypher)
            if problems:
                log_event(
                    logger, "lookup_claim_schema_violation", level=logging.WARNING,
                    request_id=request_id, cypher=generated_cypher, problems=problems,
                )
                retry_query_text = (
                    f"{scoped_query_text}\n\nCORRECTION NEEDED: a previous attempt "
                    f"produced an invalid query ({'; '.join(problems)}). Only use "
                    "properties, labels, and relationships listed in the schema above."
                )
                try:
                    retry_raw = await asyncio.to_thread(
                        self._retriever.get_search_results, retry_query_text, prompt_params
                    )
                    retry_cypher = retry_raw.metadata.get("cypher", "") if retry_raw.metadata else ""
                    if not validate_cypher_against_schema(retry_cypher):
                        raw, generated_cypher = retry_raw, retry_cypher
                        log_event(
                            logger, "lookup_claim_schema_violation_retry_fixed",
                            request_id=request_id, cypher=retry_cypher,
                        )
                except Exception:
                    pass  # keep the original (already-executed) result

        rows = [dict(record) for record in raw.records]

        verified_rows, dropped, any_row_confirmed = await self._verify_domain_scope(
            rows, domain=domain, memory_client=memory_client
        )

        unverifiable_and_unscoped = (
            domain is not None
            and not any_row_confirmed
            and rows
            and "client_domains" not in generated_cypher
        )

        log_event(
            logger, "lookup_claim_complete",
            request_id=request_id, duration_ms=timer.duration_ms,
            row_count=len(rows), verified_count=len(verified_rows),
            dropped_count=dropped, cypher=generated_cypher,
        )

        if unverifiable_and_unscoped:
            return {"error": "could_not_verify_scope"}

        result: dict[str, Any] = {"rows": verified_rows, "cypher": generated_cypher}
        if not_tracked is not None:
            result["not_tracked_note"] = f"This system does not track {not_tracked}."
        return result

    async def _lookup_multiple_refs(
        self,
        refs: list[str],
        *,
        domain: str | None,
        memory_client: Any,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        with StageTimer() as timer:
            rows = await memory_client.query.cypher(
                MULTI_CLAIM_QUERY, {"refs": refs, "domain": domain}
            )

        found_refs = {row["ref_num"] for row in rows}
        not_found = [ref for ref in refs if ref not in found_refs]

        log_event(
            logger, "lookup_claim_complete",
            request_id=request_id, duration_ms=timer.duration_ms,
            row_count=len(rows), verified_count=len(rows),
            dropped_count=0, cypher="<direct multi-ref lookup>",
        )

        return {"rows": rows, "not_found": not_found, "cypher": "<direct multi-ref lookup>"}

    async def _verify_domain_scope(
        self,
        rows: list[dict[str, Any]],
        *,
        domain: str | None,
        memory_client: Any,
    ) -> tuple[list[dict[str, Any]], int, bool]:
        if domain is None or not rows:
            return rows, 0, domain is None


        refs_by_row: list[set[str]] = []
        all_refs: set[str] = set()
        for row in rows:
            found = set()
            for value in row.values():
                if isinstance(value, str):
                    match = CLAIM_REFERENCE_PATTERN.search(value)
                    if match:
                        found.add(match.group(0).upper())
            refs_by_row.append(found)
            all_refs |= found

        if not all_refs:
            return rows, 0, False

        authorized_rows = await memory_client.query.cypher(
            VERIFY_CLAIMS_QUERY, {"refs": list(all_refs), "domain": domain}
        )
        authorized = {r["authorized"] for r in authorized_rows}

        verified: list[dict[str, Any]] = []
        dropped = 0
        for row, refs in zip(rows, refs_by_row):
            if not refs:
                dropped += 1
                continue
            if refs <= authorized:
                verified.append(row)
            else:
                dropped += 1

        return verified, dropped, bool(verified)


_tool_instance: LookupClaimTool | None = None


def get_lookup_claim_tool() -> LookupClaimTool:
    global _tool_instance
    if _tool_instance is None:
        driver = neo4j.GraphDatabase.driver(
            os.environ["NEO4J_URI"],
            auth=(
                os.environ["NEO4J_READONLY_USERNAME"],
                os.environ["NEO4J_READONLY_PASSWORD"],
            ),
        )
        _tool_instance = LookupClaimTool(driver, os.environ["NEO4J_DATABASE"])
    return _tool_instance
