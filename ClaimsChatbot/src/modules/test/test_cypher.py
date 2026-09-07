
from __future__ import annotations

import pytest

from src.modules.retrieval import cypher
from src.modules.test.conftest import (
    KNOWN_CLAIM_PORTAL_STATUS,
    KNOWN_CLAIM_REF,
    KNOWN_CLAIM_STATUS,
    KNOWN_DOMAIN,
    OTHER_TENANT_CLAIM_REF,
)

pytestmark_integration = pytest.mark.integration

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("status", "status"),
        ("Status", "status"),
        ("  settlement  ", "settlement"),
        ("settlement status", "settlement"),  
        ("payment status", "settlement"),  
        ("payment", "settlement"),  
        ("claim status", "status"),  
        ("reserve amount", "reserve"),  
        ("reserves", "reserve"),  
        (None, None),
        ("", ""),
    ],
)
def test_normalize_aspect(raw, expected):
    assert cypher._normalize_aspect(raw) == expected


async def test_execute_returns_error_when_claim_reference_missing():
    result = await cypher.execute(
        client=None, 
        aspect="status",
        claim_reference=None,
        domain=KNOWN_DOMAIN,
    )
    assert result == {"error": "missing_claim_reference"}


@pytest.mark.integration
async def test_execute_overview_returns_real_claim_data(memory):
    result = await cypher.execute(
        memory.client,
        aspect="overview",
        claim_reference=KNOWN_CLAIM_REF,
        domain=KNOWN_DOMAIN,
    )
    assert "error" not in result
    row = result["rows"][0]
    assert row["ref_num"] == KNOWN_CLAIM_REF
    assert row["status"] == KNOWN_CLAIM_STATUS
    assert row["portal_status"] == KNOWN_CLAIM_PORTAL_STATUS


@pytest.mark.integration
async def test_execute_unrecognized_aspect_falls_back_to_overview(memory):
    result = await cypher.execute(
        memory.client,
        aspect="insured name please",
        claim_reference=KNOWN_CLAIM_REF,
        domain=KNOWN_DOMAIN,
    )
    assert result["aspect"] == "overview"


@pytest.mark.integration
async def test_execute_denies_a_claim_outside_the_domain(memory):
    result = await cypher.execute(
        memory.client,
        aspect="overview",
        claim_reference=OTHER_TENANT_CLAIM_REF,
        domain=KNOWN_DOMAIN,
    )
    assert result["error"] == "not_found_or_not_authorized"


@pytest.mark.integration
async def test_execute_admin_domain_none_sees_any_claim(memory):
    result = await cypher.execute(
        memory.client,
        aspect="overview",
        claim_reference=OTHER_TENANT_CLAIM_REF,
        domain=None,
    )
    assert "error" not in result
    assert result["rows"][0]["ref_num"] == OTHER_TENANT_CLAIM_REF


