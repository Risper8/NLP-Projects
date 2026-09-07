
from __future__ import annotations
from typing import Any

RECENT_CLAIMS_QUERY = """
MATCH (claim:Claim)
WHERE $domain IS NULL OR $domain IN claim.client_domains
RETURN
    claim.ref_num AS ref_num,
    claim.status AS status,
    claim.portal_status AS portal_status,
    claim.modified_date AS modified_date
ORDER BY claim.modified_date DESC
LIMIT $limit
"""


async def list_recent_claims(client, domain: str | None, limit: int = 5) -> list[dict[str, Any]]:
    return await client.query.cypher(
        RECENT_CLAIMS_QUERY,
        {"domain": domain, "limit": limit},
    )


PORTFOLIO_BREAKDOWN_QUERY = """
MATCH (claim:Claim)
WHERE $domain IS NULL OR $domain IN claim.client_domains
WITH claim.status AS status,
     claim.portal_status AS portal_status,
     claim.payment_status AS payment_status,
     count(*) AS count
RETURN collect({
    status: status,
    portal_status: portal_status,
    payment_status: payment_status,
    count: count
}) AS breakdown
"""


async def get_portfolio_summary(
    client,
    domain: str | None,
    claims_limit: int = 25,
) -> dict[str, Any]:
    breakdown_rows = await client.query.cypher(
        PORTFOLIO_BREAKDOWN_QUERY,
        {"domain": domain},
    )

    breakdown = breakdown_rows[0]["breakdown"] if breakdown_rows else []
    total_claims = sum(item["count"] for item in breakdown)

    claims = await list_recent_claims(client, domain, limit=claims_limit)

    return {
        "total_claims": total_claims,
        "breakdown": breakdown,
        "claims": claims,
        "claims_shown": len(claims),
    }
