
from __future__ import annotations
from typing import Any

ASPECT_QUERIES: dict[str, str] = {
    "status": """
        MATCH (claim:Claim {ref_num: $ref_num})
        WHERE $domain IS NULL OR $domain IN claim.client_domains
        RETURN
            claim.ref_num AS ref_num,
            claim.status AS status,
            claim.portal_status AS portal_status,
            claim.proceed_decision AS proceed_decision,
            claim.modified_date AS modified_date
    """,

    "settlement": """
        MATCH (claim:Claim {ref_num: $ref_num})
        WHERE $domain IS NULL OR $domain IN claim.client_domains
        OPTIONAL MATCH (claim)-[:HAS_PAYMENT]->(payment:Payment)
        RETURN
            claim.ref_num AS ref_num,
            claim.payment_status AS claim_payment_status,
            payment.amount AS amount,
            payment.currency AS currency,
            payment.status AS status,
            payment.type AS payment_type,
            payment.pay_to AS pay_to,
            payment.finance_ref_num AS finance_ref_num
    """,

    "reserve": """
        MATCH (claim:Claim {ref_num: $ref_num})
        WHERE $domain IS NULL OR $domain IN claim.client_domains
        RETURN
            claim.ref_num AS ref_num,
            claim.reserves AS reserves,
            claim.base_amount AS base_amount,
            claim.class_of_business AS class_of_business
    """,


    "overview": """
        MATCH (claim:Claim {ref_num: $ref_num})
        WHERE $domain IS NULL OR $domain IN claim.client_domains
        OPTIONAL MATCH (broker:Broker)-[:SUBMITTED]->(claim)
        OPTIONAL MATCH (cedant:Cedant)-[:HAS_CLAIM]->(claim)
        OPTIONAL MATCH (insured:Insured)-[:HAS_CLAIM]->(claim)
        OPTIONAL MATCH (claim)-[:HAS_PAYMENT]->(payment:Payment)
        OPTIONAL MATCH (claim)-[:HAS_QUERY]->(query:Query)
        OPTIONAL MATCH (claim)-[:OCCURRED_IN]->(location:Location)
        RETURN
            claim.ref_num AS ref_num,
            claim.status AS status,
            claim.portal_status AS portal_status,
            claim.proceed_decision AS proceed_decision,
            claim.modified_date AS modified_date,
            claim.created_date AS created_date,
            claim.request_type AS request_type,
            claim.class_of_business AS class_of_business,
            claim.cause_of_loss AS cause_of_loss,
            claim.date_of_loss AS date_of_loss,
            claim.claim_reason AS claim_reason,
            claim.reserves AS reserves,
            claim.base_amount AS base_amount,
            claim.payment_status AS claim_payment_status,
            payment.amount AS settlement_amount,
            payment.currency AS settlement_currency,
            payment.status AS settlement_status,
            payment.type AS payment_type,
            payment.pay_to AS pay_to,
            payment.finance_ref_num AS finance_ref_num,
            broker.name AS broker_name,
            broker.broker_code AS broker_code,
            cedant.name AS cedant_name,
            cedant.cedant_code AS cedant_code,
            insured.name AS insured_name,
            query.reason AS query_reason,
            query.status AS query_status,
            query.comments AS query_comments,
            location.name AS location
    """,
}

ASPECT_ALIASES: dict[str, str] = {
    "settlement status": "settlement",
    "payment status": "settlement",
    "payment": "settlement",
    "claim status": "status",
    "reserve amount": "reserve",
    "reserves": "reserve",
}


def _normalize_aspect(aspect: str | None) -> str | None:
    if not aspect:
        return aspect
    normalized = aspect.strip().lower()
    return ASPECT_ALIASES.get(normalized, normalized)


async def execute(client, aspect: str | None, claim_reference: str | None, domain: str | None) -> dict[str, Any]:
    if not claim_reference:
        return {"error": "missing_claim_reference"}

    normalized = _normalize_aspect(aspect)
    resolved_aspect = normalized if normalized in ASPECT_QUERIES else "overview"
    query = ASPECT_QUERIES[resolved_aspect]

    rows = await client.query.cypher(
        query,
        {"ref_num": claim_reference, "domain": domain},
    )

    if not rows or rows[0].get("ref_num") is None:
        return {
            "error": "not_found_or_not_authorized",
            "claim_reference": claim_reference,
        }

    return {"aspect": resolved_aspect, "claim_reference": claim_reference, "rows": rows}


